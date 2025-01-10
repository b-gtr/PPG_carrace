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
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecTransposeImage
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
        # Höhere Auflösung: 800x600
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
#   Custom CNN Features Extractor
# -----------------------------------------------------
class CustomCNNFeaturesExtractor(BaseFeaturesExtractor):
    """
    NOTE: After using VecTransposeImage, the observation shape is (C, H, W).
    For a semantic segmentation camera, we expect C=3, H=600, W=800.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        c, h, w = observation_space.shape

        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Größe nach CNN bestimmen
        with torch.no_grad():
            dummy_input = torch.zeros((1, c, h, w), dtype=torch.float32)
            n_flatten = self.cnn(dummy_input).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.cnn(observations)
        return self.linear(x)


# -----------------------------------------------------
#   CarlaEnv (Gymnasium)
# -----------------------------------------------------
class CarlaEnv(gym.Env):
    """
    Gymnasium-Umgebung mit:
      - größerem Bild (800x600),
      - Waypoints zur Routenführung (ohne Endlosschleife),
      - erweitertem Reward: Lane Deviations brechen nicht die Episode ab,
        nur Strafe - außer man ist sehr weit von der Fahrbahn.
    """

    def __init__(self, host='localhost', port=2000, render_mode=None):
        super(CarlaEnv, self).__init__()

        # Carla init
        self.client = carla.Client(host, port)
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

        # Enable synchronous mode
        self.original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # ~20 FPS
        self.world.apply_settings(settings)

        self.latest_image = None
        self.agent_image = None

        # Max steps per episode
        self.max_episode_steps = 600
        self.current_step = 0

        # Action Space: [Steer, Throttle] in [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation Space: (H=600, W=800, C=3)
        self.observation_shape = (600, 800, 3)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=self.observation_shape,
            dtype=np.uint8
        )

        # Ziel, Kollisions-Flags etc.
        self.spawn_point = None
        self.spawn_rotation = None
        self.destination = None
        self.collision_occured = False
        self.lane_invasion_count = 0

        # Distance threshold
        self.previous_distance = None
        self.distance_threshold = 3.0

        # Route (Liste von Waypoints)
        self.route = []

        # Umgebung initialisieren
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

        # Zufälliger Spawn-Punkt
        self.spawn_point = random.choice(self.spawn_points)
        self.spawn_rotation = self.spawn_point.rotation

        vehicle_bp = self.blueprint_library.filter('vehicle.lincoln.mkz_2017')[0]
        self.vehicle = self.world.spawn_actor(vehicle_bp, self.spawn_point)

        self.setup_sensors()

        # Beispiel: Ziel 60m in Fahrtrichtung
        direction_vector = self.spawn_rotation.get_forward_vector()
        self.destination = self.spawn_point.location + direction_vector * 60

        # Route mit Waypoints (alle 2m) erstellen (ohne Endlosschleife)
        self._generate_route()

        self.current_step = 0

        # World ein paar Ticks vorlaufen lassen
        for _ in range(5):
            self.world.tick()

    def _generate_route(self, max_waypoints=1000):
        """
        Erzeugt eine Liste von Waypoints anhand der Spawn-Position
        bis (ungefähr) zur Destination, ohne Endlosschleife.
        """
        self.route = []
        start_waypoint = self.world.get_map().get_waypoint(
            self.spawn_point.location,
            project_to_road=True
        )
        if not start_waypoint:
            return

        current_wp = start_waypoint
        self.route.append(current_wp)
        last_distance = current_wp.transform.location.distance(self.destination)

        # Anzahl Schritte ohne Fortschritt
        no_progress_count = 0
        max_no_progress = 50

        for _ in range(max_waypoints):
            next_wps = current_wp.next(2.0)
            if not next_wps:
                # Keine weiteren Waypoints
                break

            # Nimm den ersten Vorschlag (alternativ könnte man hier den "besten" wählen)
            next_wp = next_wps[0]
            dist = next_wp.transform.location.distance(self.destination)

            # Abbruchbedingung: Nahe genug am Ziel
            if dist < 2.0:
                self.route.append(next_wp)
                break

            # Fortschrittskontrolle
            if dist >= last_distance:
                no_progress_count += 1
            else:
                no_progress_count = 0
                last_distance = dist

            if no_progress_count > max_no_progress:
                print("[WARN] _generate_route(): Keine Annäherung an das Ziel – Abbruch.")
                break

            self.route.append(next_wp)
            current_wp = next_wp

        else:
            # Falls man aus der for-Schleife kommt, ohne break => max_waypoints erreicht
            print("[WARN] _generate_route(): Zu viele Waypoints erzeugt – Abbruch.")

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
        # Convert semantic segmentation image to raw
        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))

        # Hier nutzen wir den Blau-Kanal als "Label" und machen 3 Kanäle draus
        labels = array[:, :, 2]

        with self.image_lock:
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

        # Steering "dämpfen", Throttle auf [0..1] abbilden
        steer_scaled = np.clip(steer / 2.0, -1.0, 1.0)
        throttle_scaled = np.clip(0.5 * (1 + throttle), 0.0, 1.0)

        control = carla.VehicleControl(
            steer=steer_scaled,
            throttle=throttle_scaled
        )
        self.vehicle.apply_control(control)

        self.world.tick()

        # Kollisionen prüfen
        if len(self.collision_sensor.get_history()) > 0:
            self.collision_occured = True

        # Lane Invasion
        if len(self.lane_invasion_sensor.get_history()) > 0:
            self.lane_invasion_count += len(self.lane_invasion_sensor.get_history())
            self.lane_invasion_sensor.clear_history()

        obs = self._get_observation()

        done = False
        truncated = False

        # Falls sehr weit von der Spurmitte -> Episode Ende (z.B. Offroad)
        _, lateral_offset = self.get_lane_center_and_offset()
        if abs(lateral_offset) > 3.0:  # Grenze anpassen
            done = True

        # Kollision -> Done
        if self.collision_occured:
            done = True

        # Zeitlimit
        if self.current_step >= self.max_episode_steps:
            truncated = True

        # Ziel erreicht?
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
        Beispielhafte erweiterte Reward-Struktur:
          - Kollision -> -100
          - Ziel -> +100
          - Timeout -> -5
          - LaneInvasion -> -0.2 pro Invasion
          - Geschwindigkeits-Kontrolle (Ziel: 30 km/h)
          - Distanz-Fortschritt
          - Spurhaltung (Lateral Offset) => Penalty, aber nicht sofort done
          - Heading-Bonus via waypoint.next(2.0)
        """
        if done and self.collision_occured:
            return -100.0
        if reached_destination:
            return 100.0
        if truncated:
            return -5.0

        reward = 0.0

        # Lane Invasion penalty
        if self.lane_invasion_count > 0:
            reward -= 0.2 * self.lane_invasion_count
            self.lane_invasion_count = 0

        # Geschwindigkeits-Kontrolle
        speed = self.get_vehicle_speed() * 3.6  # m/s -> km/h
        v_target = 30.0
        diff = abs(speed - v_target)
        if diff < 5:
            reward += 0.3
        elif diff < 10:
            reward += 0.1
        else:
            reward -= 0.3

        # Distanz-Fortschritt
        current_distance = self.get_distance_to_destination()
        if current_distance is not None:
            if self.previous_distance is not None:
                if current_distance < self.previous_distance:
                    reward += 0.5
                else:
                    reward -= 0.2
            self.previous_distance = current_distance

        # Lateral Offset-Strafe, solange man nicht offroad (> 3.0) ist
        _, lateral_offset = self.get_lane_center_and_offset()
        if abs(lateral_offset) < 0.3:
            reward += 0.3
        else:
            reward -= 0.3

        # Heading Bonus
        r_heading = self._heading_bonus()
        reward += r_heading

        # Optional clamp
        reward = max(-10.0, min(10.0, reward))
        return reward

    def _heading_bonus(self):
        transform = self.vehicle.get_transform()
        yaw = math.radians(transform.rotation.yaw)

        map_carla = self.world.get_map()
        waypoint = map_carla.get_waypoint(transform.location, project_to_road=True)
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
#   Hauptprogramm - Mehrere Agents gleichzeitig
# -----------------------------------------------------
def main():
    """
    Beispiel, um mehrere CarlaEnv-Instanzen parallel zu starten.
    Für jeden Agenten wird eine eigene Environment erzeugt.
    Achtung: Falls du lokal mehrere Carla-Server starten willst,
             musst du ggf. verschiedene Ports verwenden.
    """
    
    # Anzahl paralleler Environments/Agenten:
    num_envs = 2  # z.B. 2 Agents

    # Falls du verschiedene Ports brauchst, könnte das in einer Liste stehen:
    # ports = [2000, 2002]  # Sofern du mehrere CARLA-Server laufen hast
    # und entsprechend host/port übergeben:
    # def make_env(i):
    #     def _init():
    #         return Monitor(CarlaEnv(host='localhost', port=ports[i]))
    #     return _init

    # -- Vereinfachte Variante (alle auf demselben CARLA, Achtung Kollision möglich) --
    def make_env():
        def _init():
            env = CarlaEnv(host='localhost', port=2000)
            env = Monitor(env)
            return env
        return _init

    env_fns = [make_env() for _ in range(num_envs)]

    # Du kannst auch DummyVecEnv verwenden, wenn du nur testweise lokal ausführst.
    # SubprocVecEnv entkoppelt die Environments per Prozess.
    vec_env = SubprocVecEnv(env_fns)  
    # (B, H, W, C) -> (B, C, H, W)
    vec_env = VecTransposeImage(vec_env)

    # PPO-Konfiguration mit Custom CNN + größerer net_arch
    policy_kwargs = dict(
        net_arch=[512, 256, 128],
        features_extractor_class=CustomCNNFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=256),
    )

    ppo_hyperparams = dict(
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        learning_rate=1e-4,
        clip_range=0.2,
        ent_coef=0.001,
        vf_coef=0.5,
        gae_lambda=0.95,
        max_grad_norm=0.5,
        verbose=1,
        device="cuda",  # "cuda" wenn verfügbar
        policy_kwargs=policy_kwargs,
    )

    # PPO anlegen
    model = PPO(
        policy="CnnPolicy",
        env=vec_env,
        **ppo_hyperparams
    )

    # Training
    total_timesteps = 10_000  # Beispielhafter kleiner Wert zum Testen
    model.learn(total_timesteps=total_timesteps)

    # Model speichern
    model.save("ppo_carla_model_large_net_multiagent")

    # --- Testepisode (nur 1 Env, hier Dummy) ---
    print("Training abgeschlossen. Starte Testepisode ...")

    test_env = CarlaEnv()
    test_env = Monitor(test_env)
    test_env = DummyVecEnv([lambda: test_env])
    test_env = VecTransposeImage(test_env)

    obs, info = test_env.reset()
    for _ in range(300):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(action)
        if done or truncated:
            obs, info = test_env.reset()

    test_env.close()


if __name__ == "__main__":
    main()
