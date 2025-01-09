import os
import math
import threading
import random
import numpy as np
import time

import carla

# Gymnasium statt altem Gym
import gymnasium as gym
from gymnasium import spaces

# Stable Baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import torch
import torch.nn as nn


# -----------------------------------------------------
#   Sensor-Klassen
# -----------------------------------------------------
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
        if self.sensor is not None:
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
        # Beispiel: größere Auflösung, z. B. 800x600
        camera_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '110')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.sensor = world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.image_processor_callback = image_processor_callback

    def listen(self):
        self.sensor.listen(self.image_processor_callback)


# -----------------------------------------------------
#   Custom Features Extractor (Optional)
# -----------------------------------------------------
class CustomCNNFeaturesExtractor(BaseFeaturesExtractor):
    """
    Beispielhafter Custom CNN, angelehnt an dein verlinktes Repo:
    Evtl. tiefer oder breiter je nach Bedarf.
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        # observation_space.shape = (H, W, C)
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[2]  # i. d. R. 3

        # Baue dein CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Dummy Forward-Pass zum Dim-Bestimmen
        with torch.no_grad():
            sample_input = torch.zeros((1, n_input_channels, observation_space.shape[0], observation_space.shape[1]))
            n_flatten = self.cnn(sample_input).shape[1]

        # Anschließend lineares Mapping auf features_dim
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # [B, C, H, W]
        x = self.cnn(observations)
        return self.linear(x)


# -----------------------------------------------------
#   CarlaEnv (Gymnasium)
# -----------------------------------------------------
class CarlaEnv(gym.Env):
    """
    Gymnasium-Umgebung mit veränderter Reward-Struktur,
    größerer Bildauflösung und expansionsfähigem Setup.
    """

    def __init__(self, render_mode=None):
        super(CarlaEnv, self).__init__()

        # Carla initialisieren
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

        # Episode/Step
        self.max_episode_steps = 600  # Erhöht, wenn du längere Episoden willst
        self.current_step = 0

        # Action Space: [Steer, Throttle] in [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation Space:
        # Da wir 800x600 in 3-Kanälen haben:
        self.observation_shape = (600, 800, 3)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=self.observation_shape, dtype=np.uint8
        )

        # Zieldefinition
        self.spawn_point = None
        self.spawn_rotation = None
        self.destination = None
        self.collision_occured = False
        self.lane_invasion_count = 0

        # Distanz-Tracking
        self.previous_distance = None
        self.distance_threshold = 3.0  # Wie nah man ans Ziel muss, um als "erreicht" zu gelten

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
        self.lane_invasion_count = 0
        self.previous_distance = None

        # Feste Startposition
        self.spawn_point = random.choice(self.spawn_points)  # Oder self.spawn_points[0]
        self.spawn_rotation = self.spawn_point.rotation

        vehicle_bp = self.blueprint_library.filter('vehicle.lincoln.mkz_2017')[0]
        self.vehicle = self.world.spawn_actor(vehicle_bp, self.spawn_point)

        self.setup_sensors()

        # Beispiel: 60 Meter nach vorn als Ziel
        direction_vector = self.spawn_rotation.get_forward_vector()
        self.destination = self.spawn_point.location + direction_vector * 60

        self.current_step = 0

        # Erste Ticks
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
        self.lane_invasion_count = 0

        self.latest_image = None
        self.agent_image = None

    def process_image(self, image):
        # Semantic Segmentation
        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        labels = array[:, :, 2]

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
        vehicle_loc = self.vehicle.get_transform().location
        return vehicle_loc.distance(self.destination)

    def step(self, action):
        self.current_step += 1

        steer = float(action[0])
        throttle = float(action[1])

        # Steering und Throttle anpassen
        steer_scaled = np.clip(steer / 2.0, -1.0, 1.0)
        throttle_scaled = np.clip(0.5 * (1 + throttle), 0.0, 1.0)

        control = carla.VehicleControl(
            steer=steer_scaled,
            throttle=throttle_scaled
        )
        self.vehicle.apply_control(control)

        self.world.tick()

        # Check Sensors
        if len(self.collision_sensor.get_history()) > 0:
            self.collision_occured = True
        # Lane Invasion
        if len(self.lane_invasion_sensor.get_history()) > 0:
            self.lane_invasion_count += 1
            self.lane_invasion_sensor.clear_history()

        obs = self._get_observation()

        done = False
        truncated = False

        # Endbedingung: Kollision
        if self.collision_occured:
            done = True
        # Timeout
        elif self.current_step >= self.max_episode_steps:
            truncated = True

        distance = self.get_distance_to_destination()
        reached_destination = False
        if distance is not None and distance < self.distance_threshold:
            reached_destination = True
            done = True

        reward = self._compute_reward(done, truncated, reached_destination)
        info = {}

        return obs, reward, done, truncated, info

    def _compute_reward(self, done, truncated, reached_destination):
        """
        Neue, erweiterte Reward-Struktur:
          - Große negative Strafe bei Kollision -> -100
          - Große positive Belohnung bei Ziel -> +100
          - Timeout -> -5
          - Lane Invasion -> Strafe pro Vorfall
          - Speed-Kontrolle, Distanz-Fortschritt, etc.
        """
        # 1) Kollision
        if done and self.collision_occured:
            return -100.0

        # 2) Ziel erreicht
        if reached_destination:
            return 100.0

        # 3) Timeout
        if truncated:
            return -5.0

        # 4) Normaler Step-Reward
        reward = 0.0

        # 4a) Lane Invasion Strafe: z. B. -0.2 pro Invasion
        #    -> lane_invasion_count wird in step() hochgezählt, also belohnen wir negative je event
        if self.lane_invasion_count > 0:
            reward -= 0.2 * self.lane_invasion_count
            # Wenn du möchtest, dass wir pro Step nur 1x zählen,
            # kannst du hier self.lane_invasion_count = 0 wieder auf 0 setzen.
            self.lane_invasion_count = 0

        # 4b) Speed-Kontrolle: wir wollen ~30 km/h
        speed = self.get_vehicle_speed() * 3.6
        v_target = 30.0
        diff = abs(speed - v_target)
        if diff < 5.0:
            reward += 0.3  # gut
        elif diff < 10.0:
            reward += 0.1  # OK
        else:
            reward -= 0.3  # zu schnell/zu langsam

        # 4c) Distanz-Fortschritt: +0.5, wenn wir uns nähern, -0.2, wenn wir uns entfernen
        current_distance = self.get_distance_to_destination()
        if current_distance is not None:
            if self.previous_distance is not None:
                if current_distance < self.previous_distance:
                    reward += 0.5
                else:
                    reward -= 0.2
            self.previous_distance = current_distance

        # 4d) Spurhaltung (lateral offset)
        _, lateral_offset = self.get_lane_center_and_offset()
        if abs(lateral_offset) < 0.3:
            reward += 0.3
        else:
            reward -= 0.3

        # 4e) Bonus, wenn wir in einem "vernünftigen" Heading sind
        r_heading = self._heading_bonus()
        reward += r_heading

        # 4f) Reward-Clamp (optional, um extreme Werte zu vermeiden)
        reward = max(-10.0, min(10.0, reward))

        return reward

    def _heading_bonus(self):
        """
        Kleiner Heading-Bonus, z. B. +/- 0.2
        """
        transform = self.vehicle.get_transform()
        yaw = math.radians(transform.rotation.yaw)

        map_carla = self.world.get_map()
        waypoint = map_carla.get_waypoint(transform.location)
        if not waypoint:
            return 0.0

        next_waypoints = waypoint.next(2.0)
        if not next_waypoints:
            return 0.0

        next_wp = next_waypoints[0]
        wp_loc = next_wp.transform.location
        dx = wp_loc.x - transform.location.x
        dy = wp_loc.y - transform.location.y

        desired_yaw = math.atan2(dy, dx)
        epsilon = desired_yaw - yaw
        epsilon = (epsilon + math.pi) % (2 * math.pi) - math.pi

        # Kleiner Bonus: je kleiner Epsilon
        if abs(epsilon) < 0.2:
            return 0.2
        elif abs(epsilon) < 0.5:
            return 0.1
        else:
            return -0.2

    def _get_observation(self):
        with self.image_lock:
            if self.agent_image is None:
                return np.zeros(self.observation_shape, dtype=np.uint8)
            return self.agent_image.astype(np.uint8)

    def reset(self, seed=None, options=None):
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


# -----------------------------------------------------
#   Hauptprogramm / Trainings-Loop
# -----------------------------------------------------
def main():
    # 1) Environment erstellen
    env = CarlaEnv()
    env = Monitor(env)

    # 2) Vektor-Umgebung
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecTransposeImage(vec_env)  # (B,H,W,C)->(B,C,H,W)

    # 3) PPO-Konfiguration
    #    Größere net_arch, wie in deinem verlinkten Repo
    #    Zusätzlich: CustomCNN-Extractor, falls du es nutzen willst.
    policy_kwargs = dict(
        # Netzwerkkonfiguration (größeres MLP in Policy + Value Head)
        net_arch=[512, 256, 128],
        # Optionaler, eigener CNN-Feature-Extraktor
        features_extractor_class=CustomCNNFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=256),
    )

    ppo_hyperparams = dict(
        n_steps=2048,
        batch_size=128,  # Größerer Batch
        n_epochs=10,
        gamma=0.99,
        learning_rate=1e-4,  # ggf. kleinere LR, da größeres Netz
        clip_range=0.2,
        ent_coef=0.001,      # Du kannst ent_coef senken oder erhöhen
        vf_coef=0.5,
        gae_lambda=0.95,
        max_grad_norm=0.5,
        verbose=1,
        device="cuda",       # "cuda" wenn du eine GPU verwenden kannst
        policy_kwargs=policy_kwargs
    )

    # 4) PPO anlegen
    model = PPO(
        policy="CnnPolicy",  # "CnnPolicy" wird durch unseren CustomExtractor überschrieben
        env=vec_env,
        **ppo_hyperparams
    )

    # 5) Training
    total_timesteps = 20_000
    model.learn(total_timesteps=total_timesteps)

    # 6) Modell speichern
    model.save("ppo_carla_model_large_net")

    # Testepisode
    print("Training abgeschlossen. Starte Testepisode ...")

    test_env = CarlaEnv()
    test_env = Monitor(test_env)
    test_env = DummyVecEnv([lambda: test_env])
    test_env = VecTransposeImage(test_env)

    obs, info = test_env.reset()

    for _ in range(600):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(action)

        if done or truncated:
            obs, info = test_env.reset()

    test_env.close()


if __name__ == "__main__":
    main()
