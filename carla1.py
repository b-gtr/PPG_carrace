import os
import math
import random
import threading

import numpy as np
import torch

# WICHTIG: Gym (alte Version), NICHT gymnasium
import gym
from gym import spaces

# Stable-Baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor

try:
    import carla
except ImportError:
    raise RuntimeError("Fehler: CARLA-Python-API nicht gefunden. Bitte Carla installieren und PYTHONPATH setzen.")

# Trainingsparameter
MAX_STEPS_PER_EPISODE = 1000
LOAD_MODEL = False
MODEL_PATH = "ppo_carla_model.zip"
TOTAL_TIMESTEPS = 50_000

# Gerät initialisieren
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()
print(f"Number of GPUs available: {num_gpus}")


# -------------------------------------------------------
# 1) Native CarlaEnv (ohne Gym) -- Kapselt Fahrzeug & Sensorik
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
            Verarbeitet das semantische Segmentierungsbild (Labels im R-Kanal).
            Normalisierung auf [0,1].
            """
            image.convert(carla.ColorConverter.Raw)
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))
            labels = array[:, :, 2]

            with self.image_lock:
                self.latest_image = labels / 22.0  # ~22 Klassen in Carla

        self.sensor.listen(callback)


class CarlaEnv:
    """
    Klasse, die die CARLA-Umgebung im synchronen Modus startet
    und ein Fahrzeug + Sensoren aufbaut.
    """
    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)

        # Lade Town10HD_Opt (oder Town01 etc.)
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

        self.reset_environment()

    def reset_environment(self):
        self._clear_sensors()
        if self.vehicle is not None:
            self.vehicle.destroy()
            self.vehicle = None

        # Zufälliger Spawnpoint
        vehicle_bp = self.blueprint_library.filter('vehicle.lincoln.mkz_2017')[0]
        self.spawn_point = random.choice(self.spawn_points)
        self.vehicle = self.world.spawn_actor(vehicle_bp, self.spawn_point)

        self.setup_sensors()

        # Ein paar Ticks warten, damit Sensoren initialisiert sind
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
# 2) Gym-Wrapper, der CarlaEnv in eine gym.Env verwandelt
# -------------------------------------------------------
class CarlaGymEnv(gym.Env):
    """
    Aus der CarlaEnv machen wir jetzt eine gym.Env mit reset() und step().
    Wir geben das Bild in (H, W, C) => (480, 640, 1) an; SB3 kann das transponieren.
    """
    def __init__(self):
        super(CarlaGymEnv, self).__init__()
        self.env = CarlaEnv()

        # Action Space: 2 Dim (Steer, Throttle)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

        # Observation Space: (480, 640, 1)
        # => wir werden später VecTransposeImage benutzen, um (C,H,W) für SB3 zu bekommen.
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(480, 640, 1),
            dtype=np.float32
        )

        self.max_steps = MAX_STEPS_PER_EPISODE
        self.current_step = 0
        self.previous_distance = None

        direction_vector = self.env.spawn_point.rotation.get_forward_vector()
        self.destination = self.env.spawn_point.location + direction_vector * 40

    def reset(self):
        self.env.reset_environment()
        direction_vector = self.env.spawn_point.rotation.get_forward_vector()
        self.destination = self.env.spawn_point.location + direction_vector * 40
        self.previous_distance = None
        self.current_step = 0

        obs = self._get_observation()
        return obs

    def step(self, action):
        steer = float(action[0])  # ∈ [-1,1]
        throttle = float(action[1])  # ∈ [-1,1]

        steer = steer / 2.0
        throttle = 0.5 * (1 + throttle)

        control = carla.VehicleControl(
            steer=np.clip(steer, -1.0, 1.0),
            throttle=np.clip(throttle, 0.0, 1.0)
        )
        self.env.vehicle.apply_control(control)

        self.env.world.tick()
        self.current_step += 1

        reward, done = self._compute_reward_and_done()

        obs = self._get_observation()
        info = {}

        return obs, reward, done, info

    def _get_observation(self):
        with self.env.image_lock:
            if self.env.camera_sensor and self.env.camera_sensor.latest_image is not None:
                # shape (480,640)
                img = self.env.camera_sensor.latest_image
                # => (480,640,1)
                img = np.expand_dims(img, axis=-1)
                return img.astype(np.float32)
            else:
                return np.zeros((480,640,1), dtype=np.float32)

    def _compute_reward_and_done(self):
        # Kollision?
        if len(self.env.collision_sensor.history) > 0:
            return -30.0, True

        # Zeit abgelaufen?
        if self.current_step >= self.max_steps:
            return -10.0, True

        location = self.env.vehicle.get_transform().location
        distance_to_destination = location.distance(self.destination)
        if self.previous_distance is None:
            self.previous_distance = distance_to_destination

        lane_center, lateral_offset = self.env.get_lane_center_and_offset()
        if lane_center is None:
            # Kein Waypoint
            return -1.0, False

        # Belohnung aus deinem SAC-Beispiel
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
        map_ = self.env.world.get_map()
        waypoint = map_.get_waypoint(location)
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
        r_heading = -(abs(epsilon) / 3)**2

        # Travel
        if distance_to_destination < self.previous_distance:
            r_traveled = 1.0
        else:
            r_traveled = -0.1

        # Overspeed
        r_overspeed = -5.0 if speed > 25 else 0.0

        reward = r_lane_centering + r_speed + r_heading + r_traveled + r_overspeed
        self.previous_distance = distance_to_destination
        return reward, False

    def close(self):
        self.env.destroy()
        super().close()

# -------------------------------------------------------
# 3) Hauptprogramm: PPO mit SB3
# -------------------------------------------------------
def main():
    # 1) Instanziiere unser Gym-Env
    env = CarlaGymEnv()

    # 2) Monitor-Wrapper (Logging)
    env = Monitor(env)

    # 3) Für SB3 brauchen wir VecEnv
    vec_env = DummyVecEnv([lambda: env])

    # 4) Transponiere (H,W,C)->(C,H,W) für CnnPolicy
    vec_env = VecTransposeImage(vec_env)

    # PPO-Hyperparameter
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

    # 5) PPO instanziieren
    model = PPO(
        policy="CnnPolicy",
        env=vec_env,
        **ppo_hyperparams
    )

    if LOAD_MODEL and os.path.exists(MODEL_PATH):
        model = PPO.load(MODEL_PATH, env=vec_env)
        print("Modellparameter geladen.")

    # 6) Training
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    # 7) Speichern
    model.save(MODEL_PATH)
    print(f"Modell unter '{MODEL_PATH}' gespeichert.")

    # 8) Testen (optional)
    obs = env.reset()
    for _ in range(500):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()

    env.close()


if __name__ == "__main__":
    main()
