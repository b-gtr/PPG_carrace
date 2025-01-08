import os
import math
import random
import threading

import numpy as np
import torch

# Wir verwenden Gymnasium, NICHT gym
import gymnasium as gym
from gymnasium import spaces

# Shimmy-Wrapper, um Gymnasium-Env in "altes Gym" für SB3 zu übersetzen
from shimmy import GymV26CompatibilityV0

# Stable-Baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

try:
    import carla
except ImportError:
    raise RuntimeError("Fehler: CARLA-Python-API nicht gefunden. Bitte Carla installieren bzw. PYTHONPATH setzen.")

# Trainingsparameter
MAX_STEPS_PER_EPISODE = 1000
LOAD_MODEL = False               # Falls du bereits trainierte Parameter laden möchtest
MODEL_PATH = "ppo_carla_model.zip"
TOTAL_TIMESTEPS = 50_000         # Anzahl Trainingsschritte (anpassbar)

# Gerät initialisieren
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()
print(f"Number of GPUs available: {num_gpus}")

# -------------------------------------------------------
# 1) Native CarlaEnv (ohne Gymnasium) -- Kapselt Fahrzeug & Sensorik
# -------------------------------------------------------
class Sensor:
    """
    Stark vereinfachte Basisklasse für Sensoren.
    """
    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.sensor = None
        self.history = []

    def destroy(self):
        if self.sensor is not None:
            try:
                self.sensor.destroy()
            except RuntimeError as e:
                print(f"Fehler beim Zerstören des Sensors: {e}")


class CollisionSensor(Sensor):
    def __init__(self, vehicle, blueprint_library, world):
        super().__init__(vehicle)
        collision_bp = blueprint_library.find('sensor.other.collision')
        self.sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)

    def listen(self):
        self.sensor.listen(lambda event: self.history.append(event))


class LaneInvasionSensor(Sensor):
    def __init__(self, vehicle, blueprint_library, world):
        super().__init__(vehicle)
        lane_invasion_bp = blueprint_library.find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(lane_invasion_bp, carla.Transform(), attach_to=self.vehicle)

    def listen(self):
        self.sensor.listen(lambda event: self.history.append(event))


class GnssSensor(Sensor):
    def __init__(self, vehicle, blueprint_library, world):
        super().__init__(vehicle)
        gnss_bp = blueprint_library.find('sensor.other.gnss')
        self.sensor = world.spawn_actor(gnss_bp, carla.Transform(), attach_to=self.vehicle)
        self.current_gnss = None

    def listen(self):
        self.sensor.listen(lambda event: setattr(self, 'current_gnss', event))


class CameraSensor(Sensor):
    def __init__(self, vehicle, blueprint_library, world, image_lock):
        super().__init__(vehicle)
        camera_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        camera_bp.set_attribute('image_size_x', '640')
        camera_bp.set_attribute('image_size_y', '480')
        camera_bp.set_attribute('fov', '110')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.sensor = world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.latest_image = None
        self.image_lock = image_lock

    def listen(self):
        def callback(image):
            """
            Verarbeitet das semantische Segmentierungsbild.
            Wir nehmen den R-Kanal (image[:, :, 2]),
            da Carla die Segmentation-Labels dort speichert.
            Normalisiert auf [0,1].
            """
            image.convert(carla.ColorConverter.Raw)
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))
            labels = array[:, :, 2]

            # Normalisieren auf [0, 1], in Carla ~22 Label-Klassen
            with self.image_lock:
                self.latest_image = labels / 22.0

        self.sensor.listen(callback)


class CarlaEnv:
    """
    Nicht-Gym-Umgebung, rein zum Steuern von Carla im synchronen Modus.
    """
    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)

        # Lade Town10HD_Opt (oder z.B. Town01)
        self.client.load_world('Town10HD_Opt')
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.spawn_points = self.world.get_map().get_spawn_points()

        self.vehicle = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_sensor = None

        self.image_lock = threading.Lock()
        self.original_settings = self.world.get_settings()

        # Synchroner Modus
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 FPS
        self.world.apply_settings(settings)

        # Environment zurücksetzen
        self.reset_environment()

    def reset_environment(self):
        self._clear_sensors()
        if self.vehicle is not None:
            self.vehicle.destroy()
            self.vehicle = None

        # Spawn
        vehicle_bp = self.blueprint_library.filter('vehicle.lincoln.mkz_2017')[0]
        self.spawn_point = random.choice(self.spawn_points)
        self.vehicle = self.world.spawn_actor(vehicle_bp, self.spawn_point)

        self.setup_sensors()

        # Ein paar Ticks
        for _ in range(10):
            self.world.tick()

    def setup_sensors(self):
        self.collision_sensor = CollisionSensor(self.vehicle, self.blueprint_library, self.world)
        self.collision_sensor.listen()

        self.lane_invasion_sensor = LaneInvasionSensor(self.vehicle, self.blueprint_library, self.world)
        self.lane_invasion_sensor.listen()

        self.gnss_sensor = GnssSensor(self.vehicle, self.blueprint_library, self.world)
        self.gnss_sensor.listen()

        self.camera_sensor = CameraSensor(self.vehicle, self.blueprint_library, self.world, self.image_lock)
        self.camera_sensor.listen()

    def _clear_sensors(self):
        if self.collision_sensor is not None:
            self.collision_sensor.destroy()
        if self.lane_invasion_sensor is not None:
            self.lane_invasion_sensor.destroy()
        if self.gnss_sensor is not None:
            self.gnss_sensor.destroy()
        if self.camera_sensor is not None:
            self.camera_sensor.destroy()

    def get_vehicle_speed(self):
        vel = self.vehicle.get_velocity()
        return math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

    def destroy(self):
        # Ursprüngliche Einstellungen wiederherstellen
        self.world.apply_settings(self.original_settings)
        if self.vehicle is not None:
            self.vehicle.destroy()
        self._clear_sensors()

    def get_lane_center_and_offset(self):
        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location

        map_ = self.world.get_map()
        waypoint = map_.get_waypoint(vehicle_location, project_to_road=True)
        if not waypoint:
            return None, None

        lane_center = waypoint.transform.location
        dx = vehicle_location.x - lane_center.x
        dy = vehicle_location.y - lane_center.y

        lane_heading = math.radians(waypoint.transform.rotation.yaw)
        lane_direction = carla.Vector3D(math.cos(lane_heading), math.sin(lane_heading), 0)
        perpendicular_direction = carla.Vector3D(-lane_direction.y, lane_direction.x, 0)

        lateral_offset = dx * perpendicular_direction.x + dy * perpendicular_direction.y
        return lane_center, lateral_offset


# -------------------------------------------------------
# 2) Gymnasium-Env, die CarlaEnv umschließt
#    => (obs, reward, done, truncated, info)
# -------------------------------------------------------
class CarlaGymEnv(gym.Env):
    """
    Gymnasium-Env, die intern CarlaEnv nutzt.
    Beobachtung wird als (1, 480, 640) zurückgegeben => Channel-First
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super(CarlaGymEnv, self).__init__()
        self.env = CarlaEnv()

        # Action Space: 2 Dimensionen: Steer + Throttle ∈ [-1,1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

        # *** Wichtig: Channel-First => (1, 480, 640) ***
        # So kann Shimmy + SB3 das ohne Probleme verarbeiten.
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1, 480, 640),  
            dtype=np.float32
        )

        self.max_step = MAX_STEPS_PER_EPISODE
        self.current_step = 0
        self.previous_distance = None

        # Ziel (wird in reset() überschrieben)
        direction_vector = self.env.spawn_point.rotation.get_forward_vector()
        self.destination = self.env.spawn_point.location + direction_vector * 40

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.env.reset_environment()
        direction_vector = self.env.spawn_point.rotation.get_forward_vector()
        self.destination = self.env.spawn_point.location + direction_vector * 40
        self.previous_distance = None
        self.current_step = 0

        obs = self._get_observation()
        # Gymnasium: reset() => (obs, info)
        return obs, {}

    def step(self, action):
        """
        Rückgabe: (obs, reward, done, truncated, info)
        """
        steer = float(action[0])      # ∈ [-1,1]
        throttle = float(action[1])   # ∈ [-1,1]

        # Skaliere / clamp
        steer = steer / 2.0
        throttle = 0.5 * (1 + throttle)
        control = carla.VehicleControl(
            steer=np.clip(steer, -1.0, 1.0),
            throttle=np.clip(throttle, 0.0, 1.0)
        )
        self.env.vehicle.apply_control(control)

        # Simulations-Tick
        self.env.world.tick()
        self.current_step += 1

        # Reward
        reward, done, info = self._compute_reward_done_info()
        truncated = False
        if self.current_step >= self.max_step and not done:
            truncated = True

        obs = self._get_observation()
        return obs, reward, done, truncated, info

    def close(self):
        self.env.destroy()
        super().close()

    def _get_observation(self):
        """
        Liefert ein 3D-Array mit shape (1,480,640).
        """
        with self.env.image_lock:
            if self.env.camera_sensor and self.env.camera_sensor.latest_image is not None:
                # Carla-Bild: (480,640)
                img = self.env.camera_sensor.latest_image
                # => channel-first => (1,480,640)
                img = np.expand_dims(img, axis=0)
                return img.astype(np.float32)
            else:
                return np.zeros((1, 480, 640), dtype=np.float32)

    def _compute_reward_done_info(self):
        # Kollision?
        if len(self.env.collision_sensor.history) > 0:
            return -30.0, True, {"termination_reason": "collision"}

        # Fahrzeug-Position
        location = self.env.vehicle.get_transform().location
        distance_to_destination = location.distance(self.destination)
        if self.previous_distance is None:
            self.previous_distance = distance_to_destination

        # Lane-Abweichung
        lane_center, lateral_offset = self.env.get_lane_center_and_offset()
        if lane_center is None:
            # Kein Waypoint
            return -1.0, False, {}

        deviation_threshold = 0.7
        deviation_penalty_scale = 4.0
        if abs(lateral_offset) <= deviation_threshold:
            r_lane_centering = 1.0 / (abs(lateral_offset) + 0.1)
        else:
            r_lane_centering = -deviation_penalty_scale * (abs(lateral_offset) - deviation_threshold)

        # Speed
        speed = self.env.get_vehicle_speed() * 3.6
        v_target = 20
        r_speed = 1.0 - min(1.0, abs(speed - v_target) / 5)

        # Heading
        transform = self.env.vehicle.get_transform()
        yaw = math.radians(transform.rotation.yaw)
        waypoint = self.env.world.get_map().get_waypoint(location)
        next_waypoints = waypoint.next(2.0)
        if next_waypoints:
            next_wp = next_waypoints[0]
        else:
            next_wp = waypoint

        wp_location = next_wp.transform.location
        dx = wp_location.x - location.x
        dy = wp_location.y - location.y
        desired_yaw = math.atan2(dy, dx)
        epsilon = desired_yaw - yaw
        epsilon = (epsilon + math.pi) % (2 * math.pi) - math.pi
        r_heading = -(abs(epsilon) / 3.0)**2

        # Travel
        if distance_to_destination < self.previous_distance:
            r_traveled = 1.0
        else:
            r_traveled = -0.1

        # Overspeed
        r_overspeed = -5.0 if speed > 25 else 0.0

        reward = r_lane_centering + r_speed + r_heading + r_traveled + r_overspeed
        done = False
        self.previous_distance = distance_to_destination

        return reward, done, {}


# -------------------------------------------------------
# 3) Hauptprogramm mit Shimmy-Wrapper + SB3 (PPO)
# -------------------------------------------------------
def main():
    # 1) Erstelle unsere Gymnasium-Env
    env = CarlaGymEnv()

    # 2) Kompatibilitäts-Wrapper (shimmy), NUR wenn SB3 das alte Gym-API erwartet:
    env = GymV26CompatibilityV0(env=env)

    # 3) Monitor-Wrapper für Logging
    env = Monitor(env)

    # 4) DummyVecEnv (1 Environment)
    vec_env = DummyVecEnv([lambda: env])

    # Wir brauchen KEIN VecTransposeImage mehr, da wir schon (C,H,W) zurückgeben!
    # => Mit (1,480,640) ist bereits channel-first.

    ppo_hyperparams = dict(
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        gae_lambda=0.95,
        max_grad_norm=0.5,
        verbose=1,
        device=device
    )

    model = PPO(
        policy="CnnPolicy",
        env=vec_env,
        **ppo_hyperparams
    )

    if LOAD_MODEL and os.path.exists(MODEL_PATH):
        model = PPO.load(MODEL_PATH, env=vec_env)
        print("Modellparameter geladen.")

    # Training
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    # Speichern
    model.save(MODEL_PATH)
    print(f"Modell unter '{MODEL_PATH}' gespeichert.")

    # Optional: Testen
    obs, _ = env.reset()
    for _ in range(500):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            obs, _ = env.reset()

    env.close()


if __name__ == "__main__":
    main()
