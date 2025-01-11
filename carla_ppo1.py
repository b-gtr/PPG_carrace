"""
carla_seg_ppo_agent_extended.py

Ein erweitertes Beispielskript, das:
1) Ein CARLA-Environment mit semantischer Segmentierungskamera nutzt,
2) Fahrzeugzustand + Segmentierungsbild als gemeinsame Beobachtung ausgibt,
3) Eine rudimentäre PPO-Rollout-Speicherung implementiert,
4) Reward-Shaping demonstriert (z. B. Berücksichtigung von Kollisionen).

Voraussetzungen:
- CARLA-Server läuft auf host="localhost", port=2000 (Standard-Setup)
- Python-Pakete: carla, torch, numpy, gym, requests, etc.
- Dein GPU-Setup ist korrekt (falls device="cuda").
"""

import carla
import gym
import numpy as np
import time
import requests
import xml.etree.ElementTree as ET
import torch
import torch.nn.functional as F
from typing import Union

# ----------------------------------------------------------
# Original CarlaRouteEnv
# ----------------------------------------------------------
class CarlaRouteEnv(gym.Env):
    """
    Originales Gym-Environment aus carla.py, leicht gekürzt für die Übersicht.
    Lädt eine Route aus routes_training.xml und steuert ein Fahrzeug.
    """
    def __init__(self,
                 host="localhost",
                 port=2000,
                 town="Town01",
                 route_id="1",
                 fixed_delta_seconds=0.05,
                 distance_threshold=10.0,
                 max_steps=10000):
        super(CarlaRouteEnv, self).__init__()
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world(town)

        settings = self.world.get_settings()
        settings.fixed_delta_seconds = fixed_delta_seconds
        self.world.apply_settings(settings)

        blueprint_library = self.world.get_blueprint_library()
        self.vehicle_bp = blueprint_library.find('vehicle.tesla.model3')

        self.raw_route = self._load_route_from_leaderboard(route_id, town)
        self.track_waypoints = []
        for (x, y, z) in self.raw_route:
            loc = carla.Location(x=x, y=y, z=z)
            self.track_waypoints.append(loc)

        self.distance_threshold = distance_threshold
        self.max_steps = max_steps
        self.current_step = 0

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32)
        )
        self.vehicle = None

    def _load_route_from_leaderboard(self, route_id, town):
        url = (
            "https://raw.githubusercontent.com/carla-simulator/leaderboard/"
            "master/data/routes_training.xml"
        )
        response = requests.get(url)
        if response.status_code != 200:
            raise IOError(
                f"Fehler beim Herunterladen von routes_training.xml (Status: {response.status_code})"
            )
        root = ET.fromstring(response.text)

        for route_elem in root.findall("route"):
            rid = route_elem.get("id")
            rtown = route_elem.get("town")
            if rid == str(route_id) and rtown == town:
                waypoints_elem = route_elem.find("waypoints")
                if waypoints_elem is None:
                    continue
                route_waypoints = []
                for wp in waypoints_elem.findall("waypoint"):
                    x = float(wp.get("x"))
                    y = float(wp.get("y"))
                    z = float(wp.get("z"))
                    route_waypoints.append((x, y, z))
                return route_waypoints

        raise ValueError(f"Keine passende Route mit id={route_id} und town={town} gefunden.")

    def reset(self):
        self._destroy_actors()
        start_wp = self.track_waypoints[0]
        spawn_transform = carla.Transform(
            carla.Location(x=start_wp.x, y=start_wp.y, z=start_wp.z + 1.0),
            carla.Rotation(yaw=0)
        )
        self.vehicle = self.world.try_spawn_actor(self.vehicle_bp, spawn_transform)
        if self.vehicle is None:
            raise RuntimeError("Fahrzeug konnte nicht gespawnt werden.")
        time.sleep(1.0)

        self.current_step = 0
        obs = self._get_observation()
        return obs

    def step(self, action):
        self.current_step += 1
        steer, throttle = action
        control = carla.VehicleControl()
        control.steer = float(steer)
        control.throttle = float(throttle)
        control.brake = 0.0
        self.vehicle.apply_control(control)
        self.world.tick()

        obs = self._get_observation()
        reward, done = self._compute_reward_and_done(obs)
        if self.current_step >= self.max_steps:
            done = True
        info = {"step": self.current_step}

        return obs, reward, done, info

    def _get_observation(self):
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()

        x = transform.location.x
        y = transform.location.y
        yaw = transform.rotation.yaw
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        dist = self._get_distance_to_closest_waypoint(x, y)
        return np.array([x, y, yaw, speed, dist], dtype=np.float32)

    def _get_distance_to_closest_waypoint(self, x, y):
        closest_dist = float('inf')
        for loc in self.track_waypoints:
            dx = loc.x - x
            dy = loc.y - y
            dist_sq = dx*dx + dy*dy
            if dist_sq < closest_dist:
                closest_dist = dist_sq
        return np.sqrt(closest_dist)

    def _compute_reward_and_done(self, obs):
        dist_to_route = obs[4]
        speed = obs[3]
        reward = -dist_to_route + 0.05 * speed
        done = dist_to_route > self.distance_threshold
        return reward, done

    def _destroy_actors(self):
        if self.vehicle is not None:
            self.vehicle.destroy()
            self.vehicle = None

    def close(self):
        self._destroy_actors()


# ----------------------------------------------------------
# Environment mit segmentierter Kamera
# ----------------------------------------------------------
class CarlaRouteEnvSeg(CarlaRouteEnv):
    """
    Environment, das zusätzlich eine semantische Segmentierungs-Kamera anbringt
    und als Observation (C,H,W) liefert, in der Standardversion NUR das Kamerabild.
    """
    def __init__(self,
                 img_width=128,
                 img_height=128,
                 in_channels=3,
                 **kwargs):
        super().__init__(**kwargs)

        self.img_width = img_width
        self.img_height = img_height
        self.in_channels = in_channels

        # Beobachtung ist hier nur das Bild (3,H,W)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.in_channels, self.img_height, self.img_width), dtype=np.uint8
        )

        self.latest_image = None
        
        blueprint_library = self.world.get_blueprint_library()
        self.camera_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        self.camera_bp.set_attribute('image_size_x', str(self.img_width))
        self.camera_bp.set_attribute('image_size_y', str(self.img_height))
        self.camera_bp.set_attribute('fov', '100')
        self.camera_bp.set_attribute('sensor_tick', '0.0')

        self.camera_actor = None

    def reset(self):
        super().reset()
        self._attach_camera()

        # Warte kurz, bis erste Bilder eintrudeln
        for _ in range(5):
            self.world.tick()
            time.sleep(0.05)

        return self._get_image_observation()

    def step(self, action):
        _, reward, done, info = super().step(action)
        image_obs = self._get_image_observation()
        return image_obs, reward, done, info

    def _attach_camera(self):
        if self.vehicle is None:
            return
        camera_transform = carla.Transform(
            carla.Location(x=1.6, y=0.0, z=1.7),
            carla.Rotation(pitch=0.0)
        )
        self.camera_actor = self.world.try_spawn_actor(
            self.camera_bp, camera_transform, attach_to=self.vehicle
        )
        if self.camera_actor:
            self.camera_actor.listen(lambda image: self._on_camera_image(image))

    def _on_camera_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))  # BGRA
        # Wir extrahieren nur die ersten 3 Kanäle (BGR):
        array = array[:, :, :3]
        # Optional: In RGB konvertieren:
        array = array[..., ::-1]  # => RGB
        self.latest_image = array

    def _get_image_observation(self):
        if self.latest_image is None:
            obs = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
        else:
            obs = self.latest_image
        # (H,W,3) -> (3,H,W)
        obs = np.transpose(obs, (2, 0, 1))
        return obs

    def _destroy_actors(self):
        if self.camera_actor is not None:
            self.camera_actor.destroy()
            self.camera_actor = None
        super()._destroy_actors()


# ----------------------------------------------------------
# Erweiterung: kombiniert Bild + Fahrzeugzustand
# ----------------------------------------------------------
class CarlaRouteEnvSegExtended(CarlaRouteEnvSeg):
    """
    Wie CarlaRouteEnvSeg, aber zusätzlich (x, y, yaw, speed, dist) als Teil der Observation.
    Output = dict{'img':..., 'state':...} oder in einem concatenated array. Hier nehmen wir ein Dict.
    
    Zusätzlich: Dummy-Kollisionssensor, um z. B. Kollisionen im Reward zu berücksichtigen.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Wir definieren nun eine Observation, die ein Dictionary enthält:
        # {"img": Box(3,H,W), "state": Box(5,)} 
        # Im RL-Algorithmus muss man das verarbeiten können (z. B. via Multi-Input-Netz).
        self.observation_space = gym.spaces.Dict({
            "img": gym.spaces.Box(low=0, high=255, shape=(self.in_channels, self.img_height, self.img_width), dtype=np.uint8),
            "state": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        })

        # Kollisionssensor optional
        self.collision_sensor = None
        self.collision_flag = False  # Wenn Kollision erkannt wurde

    def reset(self):
        super().reset()
        self._attach_collision_sensor()

        # Wie zuvor warten, bis Kamera initialisiert:
        for _ in range(5):
            self.world.tick()
            time.sleep(0.05)

        return self._get_dict_observation()

    def step(self, action):
        # Rufe das step() von CarlaRouteEnvSeg auf (das nur das Bild als Obs liefern würde)
        # => wir ignorieren dort das image_obs, weil wir hier ein Dictionary zurückgeben.
        super_obs, reward, done, info = super().step(action)
        
        # Bisher: super_obs = (3,H,W). Wir wollen aber + "state".
        # und Reward-Shaping wg. Kollision:
        if self.collision_flag:
            # Beispiel: negative Belohnung
            reward -= 50.0
            # Du könntest done=True setzen oder max. anpassen

        return self._get_dict_observation(), reward, done, info

    def _attach_collision_sensor(self):
        blueprint_library = self.world.get_blueprint_library()
        collision_bp = blueprint_library.find('sensor.other.collision')
        if self.vehicle is None:
            return
        self.collision_sensor = self.world.try_spawn_actor(
            collision_bp,
            carla.Transform(),
            attach_to=self.vehicle
        )
        if self.collision_sensor:
            self.collision_sensor.listen(lambda event: self._on_collision(event))

    def _on_collision(self, event):
        # Dummy-Handling: Wir merken uns einfach, dass eine Kollision passiert ist
        self.collision_flag = True

    def _get_dict_observation(self):
        """
        Gibt ein Dictionary zurück: 
        {
           "img": (3,H,W) uint8,
           "state": (5,) float32
        }
        """
        # Hole das Bild (aus CarlaRouteEnvSeg):
        img_obs = self._get_image_observation()

        # Hole den Fahrzeugstatus (aus CarlaRouteEnv):
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        x = transform.location.x
        y = transform.location.y
        yaw = transform.rotation.yaw
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        dist = self._get_distance_to_closest_waypoint(x, y)
        state_obs = np.array([x, y, yaw, speed, dist], dtype=np.float32)

        return {
            "img": img_obs,
            "state": state_obs
        }

    def _destroy_actors(self):
        if self.collision_sensor is not None:
            self.collision_sensor.destroy()
            self.collision_sensor = None
        super()._destroy_actors()
        self.collision_flag = False


# ----------------------------------------------------------
# PPO-Klassen aus ppo.py (unverändert)
# ----------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def kaiming_init(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

class CNNLSTMActor(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        action_dim: int = 2,
        hidden_dim: int = 256,
        lstm_hidden: int = 128,
        dropout_prob: float = 0.1,
        dummy_img_size: int = 64,
        num_lstm_layers: int = 1
    ):
        super().__init__()
        # Nur CNN auf das Bild. Wenn du "state" (x, y, yaw, speed, dist) verarbeiten willst,
        # müsstest du hier eine zweite Input-Branch hinzufügen oder Feature-Fusion anpassen.
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten()
        )

        self.feature_dim = self._get_feature_dim(in_channels, dummy_img_size, dummy_img_size)
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_param = nn.Parameter(torch.zeros(action_dim))

        kaiming_init(self)

    def _get_feature_dim(self, in_channels, dummy_img_height, dummy_img_width):
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, dummy_img_height, dummy_img_width)
            out = self.cnn(dummy_input)
            return out.shape[1]

    def forward(self, x, lstm_state=None):
        """
        x: (B, T, C, H, W) - nur Bilddaten
        Gibt (mu, log_std, (h, c)) zurück.
        """
        B, T, C, H, W = x.shape
        x_reshaped = x.view(B * T, C, H, W)
        feats = self.cnn(x_reshaped)
        feats_lstm_in = feats.view(B, T, -1)

        if lstm_state is None:
            lstm_out, (h, c) = self.lstm(feats_lstm_in)
        else:
            lstm_out, (h, c) = self.lstm(feats_lstm_in, lstm_state)

        fc_out = self.fc(lstm_out)
        mu = self.mu_head(fc_out)
        log_std = self.log_std_param.expand_as(mu)

        return mu, log_std, (h, c)

class CNNLSTMCritic(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 256,
        lstm_hidden: int = 128,
        dropout_prob: float = 0.1,
        dummy_img_size: int = 64,
        num_lstm_layers: int = 1
    ):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten()
        )
        self.feature_dim = self._get_feature_dim(in_channels, dummy_img_size, dummy_img_size)
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, 1)
        )
        kaiming_init(self)

    def _get_feature_dim(self, in_channels, dummy_img_height, dummy_img_width):
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, dummy_img_height, dummy_img_width)
            out = self.cnn(dummy_input)
            return out.shape[1]

    def forward(self, x, lstm_state=None):
        B, T, C, H, W = x.shape
        x_reshaped = x.view(B * T, C, H, W)
        feats = self.cnn(x_reshaped)
        feats_lstm_in = feats.view(B, T, -1)

        if lstm_state is None:
            lstm_out, (h, c) = self.lstm(feats_lstm_in)
        else:
            lstm_out, (h, c) = self.lstm(feats_lstm_in, lstm_state)

        values = self.fc(lstm_out).squeeze(-1)
        return values, (h, c)

class PPO:
    def __init__(
        self,
        in_channels: int = 3,
        action_dim: int = 2,
        init_learning_rate: float = 3e-4,
        lr_konstant: float = 1.0,
        n_steps: int = 2048,
        n_epochs: int = 10,
        n_envs: int = 1,
        batch_size: int = 64,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        clip_range_vf: Union[None, float] = None,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        device: str = "cpu",
        num_lstm_layers: int = 1
    ):
        self.device = torch.device(device)
        self.init_learning_rate = init_learning_rate
        self.learning_rate = init_learning_rate
        self.lr_konstant = lr_konstant
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.n_envs = n_envs
        self.batch_size = batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef

        self.actor = CNNLSTMActor(
            in_channels=in_channels,
            action_dim=action_dim,
            num_lstm_layers=num_lstm_layers
        ).to(self.device)

        self.critic = CNNLSTMCritic(
            in_channels=in_channels,
            num_lstm_layers=num_lstm_layers
        ).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate)

    def update_learning_rate(self, epoch):
        self.learning_rate = self.init_learning_rate * np.exp(-self.lr_konstant * epoch)
        for param_group in self.actor_optimizer.param_groups:
            param_group["lr"] = self.learning_rate
        for param_group in self.critic_optimizer.param_groups:
            param_group["lr"] = self.learning_rate

    def get_action_and_value(self, obs, actor_lstm_state=None, critic_lstm_state=None):
        """
        obs: (B,T,C,H,W) Bild. 
        state: Normalerweise wird "state" (x,y,yaw,...) ebenfalls verarbeitet. 
               In diesem Demo-Code NICHT integriert. 
               => Du müsstest es am Netz-Eingang fusionieren.
        """
        obs = obs.to(self.device)
        mu, log_std, new_actor_state = self.actor(obs, actor_lstm_state)
        values, new_critic_state = self.critic(obs, critic_lstm_state)
        return mu, log_std, values, new_actor_state, new_critic_state

    def _compute_gae(self, rewards, values, dones, next_value):
        T = len(rewards)
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        gae = 0.0
        for step in reversed(range(T)):
            next_non_terminal = 1.0 - dones[step]
            delta = rewards[step] + self.gamma * next_value * next_non_terminal - values[step]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[step] = gae
            next_value = values[step]
        returns = advantages + values
        return returns, advantages

    def train_on_batch(self, rollouts):
        obs_all       = rollouts["obs"]       # [N, T, C, H, W]
        actions_all   = rollouts["actions"]   # [N, T, action_dim]
        old_logp_all  = rollouts["log_probs"] # [N, T]
        old_values_all= rollouts["values"]    # [N, T]
        rewards_all   = rollouts["rewards"]   # [N, T]
        dones_all     = rollouts["dones"]     # [N, T]
        next_value_all= rollouts["next_value"]# [N]

        N, T = old_logp_all.shape

        returns_all    = []
        advantages_all = []
        for i in range(N):
            ret_i, adv_i = self._compute_gae(
                rewards_all[i], 
                old_values_all[i], 
                dones_all[i], 
                next_value_all[i]
            )
            returns_all.append(ret_i)
            advantages_all.append(adv_i)

        returns_all    = torch.stack(returns_all)
        advantages_all = torch.stack(advantages_all)
        advantages_all = (advantages_all - advantages_all.mean()) / (advantages_all.std() + 1e-8)

        flat_obs       = obs_all.view(N * T, *obs_all.shape[2:])       
        flat_actions   = actions_all.view(N * T, -1)                   
        flat_old_logp  = old_logp_all.view(N * T)
        flat_old_values= old_values_all.view(N * T)
        flat_returns   = returns_all.view(N * T)
        flat_advantage = advantages_all.view(N * T)

        total_steps = N * T
        indices = np.arange(total_steps)

        for epoch in range(self.n_epochs):
            np.random.shuffle(indices)

            for start in range(0, total_steps, self.batch_size):
                end = start + self.batch_size
                mb_inds = indices[start:end]

                mb_obs       = flat_obs[mb_inds]
                mb_actions   = flat_actions[mb_inds]
                mb_old_logp  = flat_old_logp[mb_inds]
                mb_old_values= flat_old_values[mb_inds]
                mb_returns   = flat_returns[mb_inds]
                mb_advantage = flat_advantage[mb_inds]

                mb_obs_2D = mb_obs.unsqueeze(1)  # => (B,1,C,H,W)

                mu, log_std, _ = self.actor(mb_obs_2D)
                values, _      = self.critic(mb_obs_2D)

                mu      = mu.squeeze(1)        
                log_std = log_std.squeeze(1)   
                values  = values.squeeze(1)    

                dist = torch.distributions.Normal(mu, log_std.exp())
                new_logp = dist.log_prob(mb_actions).sum(dim=-1)
                
                ratio = torch.exp(new_logp - mb_old_logp)
                surr1 = ratio * mb_advantage
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * mb_advantage
                policy_loss = -torch.min(surr1, surr2).mean()

                entropy = dist.entropy().sum(dim=-1).mean()
                ent_loss = -self.ent_coef * entropy

                if self.clip_range_vf is not None:
                    clipped_values = mb_old_values + torch.clamp(values - mb_old_values,
                                                                 -self.clip_range_vf, 
                                                                 self.clip_range_vf)
                    vf_losses1 = (values - mb_returns)**2
                    vf_losses2 = (clipped_values - mb_returns)**2
                    vf_loss = 0.5 * torch.max(vf_losses1, vf_losses2).mean()
                else:
                    vf_loss = 0.5 * (mb_returns - values).pow(2).mean()

                loss = policy_loss + self.vf_coef * vf_loss + ent_loss

                self.actor_optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.actor_optimizer.step()

                critic_loss = vf_loss
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()


# ----------------------------------------------------------
# Rudimentärer Trainings-Loop mit Rollout-Speicherung
# ----------------------------------------------------------
def gather_rollout(env, ppo_agent, n_steps=200):
    """
    Führt n_steps lang Aktionen aus und sammelt (Obs, Actions, Rewards, Values, LogProbs, Dones).
    Hier nur Single-Environment.
    """
    obs_buffer = []
    actions_buffer = []
    rewards_buffer = []
    dones_buffer = []
    values_buffer = []
    log_probs_buffer = []

    # Aktuelle LSTM-States
    actor_lstm_state = None
    critic_lstm_state = None

    obs = env.reset()  
    # Wandle die Observation in das Format um, das dein Netz erwartet.
    # Hier: Wir nehmen an, wir extrahieren das Bild aus obs und ignorieren state.
    # ACHTUNG: In "CarlaRouteEnvSegExtended" ist obs ein Dict.
    if isinstance(obs, dict):
        img_obs = obs["img"]
    else:
        img_obs = obs

    # => (B=1, T=1, C,H,W)
    obs_tensor = torch.from_numpy(img_obs).float().unsqueeze(0).unsqueeze(0)  # [1,1,3,H,W]
    steps = 0

    while True:
        with torch.no_grad():
            mu, log_std, value, actor_lstm_state, critic_lstm_state = ppo_agent.get_action_and_value(
                obs_tensor,
                actor_lstm_state,
                critic_lstm_state
            )
            dist = torch.distributions.Normal(mu, log_std.exp())
            action = dist.sample()[0, 0].cpu().numpy()  # [action_dim]
            log_prob = dist.log_prob(torch.tensor(action)).sum().item()
            value = value[0, 0].item()

        # Environment-Schritt
        obs_new, reward, done, info = env.step(action)

        obs_buffer.append(img_obs)
        actions_buffer.append(action)
        rewards_buffer.append(reward)
        dones_buffer.append(done)
        values_buffer.append(value)
        log_probs_buffer.append(log_prob)

        obs = obs_new
        if isinstance(obs, dict):
            img_obs = obs["img"]
        else:
            img_obs = obs

        obs_tensor = torch.from_numpy(img_obs).float().unsqueeze(0).unsqueeze(0)

        steps += 1
        if done or steps >= n_steps:
            break

    # Nächstes Value schätzen
    if done:
        next_value = 0.0
    else:
        with torch.no_grad():
            mu, log_std, value, _, _ = ppo_agent.get_action_and_value(obs_tensor, actor_lstm_state, critic_lstm_state)
            next_value = value[0, 0].item()

    # In ein Dictionary packen
    rollouts = {
        "obs":        torch.from_numpy(np.array(obs_buffer)).float(),   # => [T,3,H,W]
        "actions":    torch.from_numpy(np.array(actions_buffer)).float(),  
        "rewards":    torch.from_numpy(np.array(rewards_buffer)).float(),
        "dones":      torch.from_numpy(np.array(dones_buffer)).float(),
        "values":     torch.from_numpy(np.array(values_buffer)).float(),
        "log_probs":  torch.from_numpy(np.array(log_probs_buffer)).float(),
        "next_value": torch.tensor([next_value], dtype=torch.float32)
    }

    # Da PPO train_on_batch erwartet: (N,T,...) => wir machen N=1
    for k in ["obs", "actions", "rewards", "dones", "values", "log_probs"]:
        rollouts[k] = rollouts[k].unsqueeze(0)  # => [1,T,...]

    return rollouts


if __name__ == "__main__":
    # Beispiel-Environment
    env = CarlaRouteEnvSegExtended(
        host="localhost",
        port=2000,
        town="Town01",
        route_id="1",
        distance_threshold=15.0,
        max_steps=3000,
        img_width=128,
        img_height=128,
        in_channels=3
    )

    # PPO-Agent
    ppo_agent = PPO(
        in_channels=3,
        action_dim=2,
        device="cuda" if torch.cuda.is_available() else "cpu",
        n_steps=1800,
        n_epochs=5
    )

    # Einfacher Trainingsloop über mehrere Episoden
    n_episodes = 1000
    for ep in range(n_episodes):
        # Rollout generieren
        rollouts = gather_rollout(env, ppo_agent, n_steps=1800)

        # PPO-Update
        ppo_agent.train_on_batch(rollouts)

        # Statistik
        ep_rew = rollouts["rewards"].sum().item()
        print(f"[Episode {ep+1}] Reward: {ep_rew:.2f}")

    env.close()
