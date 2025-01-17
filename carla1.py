import os
import math
import threading
import random
import numpy as np
import time

import carla

# Gymnasium (nicht das alte "gym")
import gymnasium as gym
from gymnasium import spaces

# Stable Baselines 3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor


class Sensor:
    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.sensor = None
        self.history = []

    def listen(self):
        raise NotImplementedError

    def clear_history(self):
        self.history.clear()

    def destroy(self):
        if self.sensor:
            try:
                self.sensor.destroy()
            except RuntimeError as e:
                print(f"Fehler beim Zerstören des Sensors: {e}")

    def get_history(self):
        return self.history


class CollisionSensor(Sensor):
    def __init__(self, vehicle, blueprint_library, world):
        super().__init__(vehicle)
        collision_bp = blueprint_library.find('sensor.other.collision')
        self.sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)

    def _on_collision(self, event):
        self.history.append(event)

    def listen(self):
        self.sensor.listen(self._on_collision)


class LaneInvasionSensor(Sensor):
    def __init__(self, vehicle, blueprint_library, world):
        super().__init__(vehicle)
        lane_invasion_bp = blueprint_library.find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(lane_invasion_bp, carla.Transform(), attach_to=self.vehicle)

    def _on_lane_invasion(self, event):
        self.history.append(event)

    def listen(self):
        self.sensor.listen(self._on_lane_invasion)


class GnssSensor(Sensor):
    def __init__(self, vehicle, blueprint_library, world):
        super().__init__(vehicle)
        gnss_bp = blueprint_library.find('sensor.other.gnss')
        self.sensor = world.spawn_actor(gnss_bp, carla.Transform(), attach_to=self.vehicle)
        self.current_gnss = None

    def _on_gnss_event(self, event):
        self.current_gnss = event

    def listen(self):
        self.sensor.listen(self._on_gnss_event)

    def get_current_gnss(self):
        return self.current_gnss


class CameraSensor(Sensor):
    def __init__(self, vehicle, blueprint_library, world, image_processor_callback):
        super().__init__(vehicle)
        camera_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        camera_bp.set_attribute('image_size_x', '640')
        camera_bp.set_attribute('image_size_y', '480')
        camera_bp.set_attribute('fov', '110')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.sensor = world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.image_processor_callback = image_processor_callback

    def listen(self):
        self.sensor.listen(self.image_processor_callback)


class CarlaEnv(gym.Env):
    """
    Gymnasium-Umgebung (>=0.26):
      - reset() -> (obs, info)
      - step() -> (obs, reward, done, truncated, info)

    Reward-Struktur:
      - Meistens Werte im Bereich -1..+1
      - Kollision -> -5 (größerer negativer Wert)
      - Ziel erreicht -> +5 (größerer positiver Wert)
    """

    def __init__(self, render_mode=None):
        super(CarlaEnv, self).__init__()

        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.client.load_world('Town01')

        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.spawn_points = self.world.get_map().get_spawn_points()

        self.vehicle = None
        self.camera_sensor = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None

        self.image_lock = threading.Lock()
        self.running = True

        # Synchroner Modus
        self.original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 FPS
        self.world.apply_settings(settings)

        self.latest_image = None
        self.agent_image = None

        # Episodenbegrenzung
        self.max_episode_steps = 300
        self.current_step = 0

        # Aktionen: [Steer, Throttle] in [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Beobachtung: 480x640-Bild in 3 Kanälen
        self.observation_shape = (480, 640, 3)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=self.observation_shape, dtype=np.uint8
        )

        # Spawn / Ziel
        self.spawn_point = None
        self.spawn_rotation = None
        self.destination = None
        self.collision_occured = False

        # Distanz-Messung
        self.previous_distance = None
        self.distance_threshold = 3.0  # Wenn man näher als 3m am Ziel ist, gilt "Ziel erreicht"

        self.reset_environment()

    def reset_environment(self):
        self._clear_sensors()

        if self.vehicle is not None:
            try:
                self.vehicle.destroy()
            except:
                pass
            self.vehicle = None

        self.collision_occured = False

        # Feste Spawn-Location, z.B. Index 0
        self.spawn_point = self.spawn_points[0]
        self.spawn_rotation = self.spawn_point.rotation

        vehicle_bp = self.blueprint_library.filter('vehicle.lincoln.mkz_2017')[0]
        self.vehicle = self.world.spawn_actor(vehicle_bp, self.spawn_point)

        self.setup_sensors()

        # Ziel: 40 Meter geradeaus
        direction_vector = self.spawn_rotation.get_forward_vector()
        self.destination = self.spawn_point.location + direction_vector * 40

        self.current_step = 0
        self.previous_distance = None

        # Ein paar Ticks warten für erstes Bild
        for _ in range(5):
            self.world.tick()

    def setup_sensors(self):
        self.camera_sensor = CameraSensor(
            self.vehicle, self.blueprint_library, self.world, self.process_image
        )
        self.camera_sensor.listen()

        self.collision_sensor = CollisionSensor(
            self.vehicle, self.blueprint_library, self.world
        )
        self.collision_sensor.listen()

        self.lane_invasion_sensor = LaneInvasionSensor(
            self.vehicle, self.blueprint_library, self.world
        )
        self.lane_invasion_sensor.listen()

        self.gnss_sensor = GnssSensor(
            self.vehicle, self.blueprint_library, self.world
        )
        self.gnss_sensor.listen()

    def _clear_sensors(self):
        if self.camera_sensor:
            self.camera_sensor.destroy()
        if self.collision_sensor:
            self.collision_sensor.destroy()
        if self.lane_invasion_sensor:
            self.lane_invasion_sensor.destroy()
        if self.gnss_sensor:
            self.gnss_sensor.destroy()

        self.camera_sensor = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None

        self.latest_image = None
        self.agent_image = None

    def process_image(self, image):
        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        labels = array[:, :, 2]  # roter Kanal

        with self.image_lock:
            # 1-Kanal in 3-Kanäle duplizieren
            labels_3ch = np.stack([labels, labels, labels], axis=-1)
            self.agent_image = labels_3ch

    def get_vehicle_speed(self):
        vel = self.vehicle.get_velocity()
        return math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

    def get_lane_center_and_offset(self):
        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location

        map_carla = self.world.get_map()
        waypoint = map_carla.get_waypoint(vehicle_location, project_to_road=True)
        if not waypoint:
            return None, 0.0

        lane_center = waypoint.transform.location
        dx = vehicle_location.x - lane_center.x
        dy = vehicle_location.y - lane_center.y

        lane_heading = math.radians(waypoint.transform.rotation.yaw)
        lane_direction = carla.Vector3D(math.cos(lane_heading), math.sin(lane_heading), 0)
        perpendicular_direction = carla.Vector3D(-lane_direction.y, lane_direction.x, 0)

        lateral_offset = dx * perpendicular_direction.x + dy * perpendicular_direction.y
        return lane_center, lateral_offset

    def get_distance_to_destination(self):
        if self.destination is None:
            return None
        vehicle_location = self.vehicle.get_transform().location
        return vehicle_location.distance(self.destination)

    def step(self, action):
        """
        Gymnasium: step -> (obs, reward, done, truncated, info)
        """
        self.current_step += 1

        steer = float(action[0])
        throttle = float(action[1])

        # Steering / Throttle anpassen
        steer_scaled = np.clip(steer / 2.0, -1.0, 1.0)
        throttle_scaled = np.clip(0.5 * (1 + throttle), 0.0, 1.0)

        control = carla.VehicleControl(
            steer=steer_scaled,
            throttle=throttle_scaled
        )
        self.vehicle.apply_control(control)

        self.world.tick()

        if len(self.collision_sensor.get_history()) > 0:
            self.collision_occured = True

        obs = self._get_observation()

        # Check terminal conditions
        done = False
        truncated = False

        # Kollision -> Ende
        if self.collision_occured:
            done = True

        # Zeit/Schritte abgelaufen
        elif self.current_step >= self.max_episode_steps:
            truncated = True

        # Check, ob wir das Ziel erreicht haben (Abstand < threshold => Belohnung, done = True)
        distance = self.get_distance_to_destination()
        reached_destination = False
        if distance is not None and distance < self.distance_threshold:
            # Ziel erreicht => Extra-Reward + Ende
            reached_destination = True
            done = True

        reward = self._compute_reward(done, truncated, reached_destination)
        info = {}

        return obs, reward, done, truncated, info

    def _compute_reward(self, done, truncated, reached_destination):
        """
        Rewardskalierung:
          - Normale Rewards zwischen -1 und +1
          - Bei Kollision -> -5
          - Bei Zielerreichung -> +5
        """
        # 1) Kollision
        if done and self.collision_occured:
            return -5.0

        # 2) Ziel erreicht
        if reached_destination:
            return 5.0

        # 3) Timeout
        if truncated:
            return -1.0

        # 4) Normaler Step-Reward
        total_reward = 0.0

        # 4a) Lane-Kontrolle in [-0.5..+0.5]
        _, lateral_offset = self.get_lane_center_and_offset()
        deviation_threshold = 0.7  # ab hier Strafe
        if abs(lateral_offset) < 0.1:
            # sehr gut in der Spur -> +0.5
            r_lane = 0.5
        elif abs(lateral_offset) < deviation_threshold:
            # mittel -> +0.2
            r_lane = 0.2
        else:
            # stark abgewichen -> -0.5
            r_lane = -0.5
        total_reward += r_lane

        # 4b) Speed-Kontrolle in [-0.5..+0.5]
        speed = self.get_vehicle_speed() * 3.6
        v_target = 20.0
        speed_diff = abs(speed - v_target)
        if speed_diff < 2:
            # sehr nah an 20 => +0.5
            r_speed = 0.5
        elif speed_diff < 5:
            # okay => +0.2
            r_speed = 0.2
        else:
            # zu schnell/zu langsam => -0.5
            r_speed = -0.5
        total_reward += r_speed

        # 4c) Distanz-Fortschritt: +0.2, wenn wir uns nähern, -0.1, wenn wir uns entfernen
        current_distance = self.get_distance_to_destination()
        if current_distance is not None:
            if self.previous_distance is None:
                # Erstes Step => kein Fortschritt
                r_distance = 0.0
            else:
                if current_distance < self.previous_distance:
                    r_distance = 0.2
                else:
                    r_distance = -0.1
            self.previous_distance = current_distance
            total_reward += r_distance

        # 4d) Heading / Wegpunkte in [-0.2..+0.2]
        r_heading = self._heading_reward()
        total_reward += r_heading

        # 4e) Begrenzung, dass Reward maximal ~1 oder minimal ~-1 (außer Kollision/Ziel/Timeout)
        total_reward = max(-1.0, min(1.0, total_reward))

        return total_reward

    def _heading_reward(self):
        # Heading in [-0.2..+0.2]
        transform = self.vehicle.get_transform()
        yaw = math.radians(transform.rotation.yaw)

        map_carla = self.world.get_map()
        waypoint = map_carla.get_waypoint(transform.location)
        if waypoint:
            next_waypoints = waypoint.next(2.0)
        else:
            next_waypoints = []

        if next_waypoints:
            next_wp = next_waypoints[0]
            wp_loc = next_wp.transform.location
            dx = wp_loc.x - transform.location.x
            dy = wp_loc.y - transform.location.y
            desired_yaw = math.atan2(dy, dx)
            epsilon = desired_yaw - yaw
            epsilon = (epsilon + math.pi) % (2 * math.pi) - math.pi
            # kleine Abweichung => besserer Reward
            if abs(epsilon) < 0.1:
                return 0.2
            elif abs(epsilon) < 0.3:
                return 0.1
            else:
                return -0.2
        else:
            return 0.0

    def _get_observation(self):
        with self.image_lock:
            if self.agent_image is None:
                return np.zeros(self.observation_shape, dtype=np.uint8)
            return self.agent_image.astype(np.uint8)

    def reset(self, seed=None, options=None):
        """
        Gymnasium reset-Signatur:
        -> (obs, info)
        """
        super().reset(seed=seed)
        self.reset_environment()
        obs = self._get_observation()
        info = {}
        return obs, info

    def render(self):
        pass

    def close(self):
        self.running = False
        self.world.apply_settings(self.original_settings)

        if self.camera_sensor:
            self.camera_sensor.destroy()
        if self.collision_sensor:
            self.collision_sensor.destroy()
        if self.lane_invasion_sensor:
            self.lane_invasion_sensor.destroy()
        if self.gnss_sensor:
            self.gnss_sensor.destroy()
        if self.vehicle:
            self.vehicle.destroy()


def main():
    # Environment erstellen
    env = CarlaEnv()
    env = Monitor(env)

    # Vektor-Umgebung für SB3
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecTransposeImage(vec_env)

    # PPO-Parameter
    ppo_hyperparams = dict(
        n_steps=1024,
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
        device="cuda",  # Falls PyTorch mit CUDA installiert
    )

    model = PPO(
        policy="CnnPolicy",
        env=vec_env,
        **ppo_hyperparams
    )

    total_timesteps = 10_000
    model.learn(total_timesteps=total_timesteps)
    model.save("ppo_carla_model_small_rewards")

    # Test-Episode
    print("Training abgeschlossen. Starte Testepisode ...")
    test_env = CarlaEnv()
    test_env = Monitor(test_env)
    test_env = DummyVecEnv([lambda: test_env])
    test_env = VecTransposeImage(test_env)

    # Da Gymnasium: reset() -> (obs, info)
    obs, info = test_env.reset()

    for _ in range(300):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(action)

        # Optional: test_env.render() oder logging
        if done or truncated:
            obs, info = test_env.reset()

    test_env.close()


if __name__ == "__main__":
    main()
