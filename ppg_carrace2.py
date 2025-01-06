import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data import Dataset, DataLoader

import cv2  # Für Graustufen-Konvertierung

###############################################
# RL-Zoo / PPO-ähnliche Hyperparameter
###############################################
GAMMA = 0.99
LAMBDA = 0.95
LR = 1e-4            # etwas kleinere LR für mehr Stabilität
ROLLOUT_STEPS = 2048
BATCH_SIZE = 64
EPOCHS_PER_UPDATE = 10
CLIP_EPS = 0.2
ENT_COEF = 0.01      # Etwas Entropie-Koeff. für bessere Exploration
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5

# PPG-spezifische
AUX_EPOCHS = 2
MAX_ITERATIONS = 1000

# VAE-Hyperparameter
LATENT_DIM = 32          # Latent-Space-Größe
VAE_BUFFER_SIZE = 30000  # Max Frames für VAE
VAE_TRAIN_EPOCHS = 5     # Epochen pro VAE-Train
VAE_LR = 3e-4
BETA_KL = 0.5            # Weniger strenges Beta -> weniger starker KL-Drang

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###################################################
# 1) VAE-Modell (1-Kanal statt 3)
###################################################
class VAE(nn.Module):
    """
    VAE für 96x96 Graustufenbilder (1 Kanal).
    """
    def __init__(self, latent_dim=32, img_channels=1, img_size=96):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.enc_conv = nn.Sequential(
            nn.Conv2d(img_channels, 32, 4, 2, 1),   # [32, 48, 48]
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),             # [64, 24, 24]
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),            # [128, 12, 12]
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),           # [256, 6, 6]
            nn.ReLU(),
        )
        self.enc_fc_mu = nn.Linear(256 * 6 * 6, latent_dim)
        self.enc_fc_logvar = nn.Linear(256 * 6 * 6, latent_dim)

        # Decoder
        self.dec_fc = nn.Linear(latent_dim, 256 * 6 * 6)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # [128, 12, 12]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # [64, 24, 24]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # [32, 48, 48]
            nn.ReLU(),
            nn.ConvTranspose2d(32, img_channels, 4, 2, 1),  # [1, 96, 96]
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.enc_conv(x)
        h = h.view(h.size(0), -1)
        mu = self.enc_fc_mu(h)
        logvar = self.enc_fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.dec_fc(z)
        h = h.view(h.size(0), 256, 6, 6)
        return self.dec_conv(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


###################################################
# 2) VAE-Dataset + Trainer
###################################################
class VAEDataset(Dataset):
    """
    Enthält 96x96-Graustufenframes: [96, 96, 1].
    """
    def __init__(self, frames):
        self.frames = frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        # frame shape: [96, 96, 1], Werte 0..255
        frame = self.frames[idx]
        frame_tensor = torch.from_numpy(frame).float() / 255.0
        # [96, 96, 1] -> [1, 96, 96]
        frame_tensor = frame_tensor.permute(2, 0, 1)
        return frame_tensor


class VAETrainer:
    def __init__(self, vae: VAE, lr=3e-4, beta_kl=1.0, device=DEVICE):
        self.vae = vae.to(device)
        self.optimizer = optim.Adam(self.vae.parameters(), lr=lr)
        self.beta_kl = beta_kl
        self.device = device

    def loss_function(self, x, x_recon, mu, logvar):
        """
        Rekonstruktionsloss + Beta-KL
        """
        recon_loss = nn.functional.mse_loss(x_recon, x, reduction="sum")
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.beta_kl * kl_loss

    def train_vae(self, frames, epochs=5, batch_size=64):
        dataset = VAEDataset(frames)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.vae.train()
        for ep in range(epochs):
            total_loss = 0
            for batch in dataloader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()

                x_recon, mu, logvar = self.vae(batch)
                loss = self.loss_function(batch, x_recon, mu, logvar)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader.dataset)
            print(f"[VAE] Epoch={ep+1}/{epochs} Loss={avg_loss:.2f}")


###################################################
# 3) ActorCritic (1-Kanal VAE latent)
###################################################
class ActorCritic(nn.Module):
    def __init__(self, latent_dim, action_dim=3, hidden_size=256):
        super().__init__()
        self.latent_dim = latent_dim

        # Policy
        self.policy_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        # Value
        self.value_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        self.mu = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        self.v = nn.Linear(hidden_size, 1)

    def forward_policy(self, latent):
        p = self.policy_net(latent)
        mu = self.mu(p)
        return mu

    def forward_value(self, latent):
        v_ = self.value_net(latent)
        val = self.v(v_)
        return val

    def get_logp_value(self, latent, action):
        with torch.no_grad():
            self.log_std.data = torch.clamp(self.log_std.data, -2.0, 1.0)
        mu = self.forward_policy(latent)
        std = torch.exp(self.log_std)
        dist = Normal(mu, std)

        logp = dist.log_prob(action).sum(dim=-1)
        value = self.forward_value(latent).squeeze(-1)
        return logp, value


###################################################
# 4) GAE-Funktion
###################################################
def compute_gae_returns(rewards, values, dones, gamma=0.99, lam=0.95, last_val=0.0):
    T = len(rewards)
    gae = 0.0
    advantages = np.zeros_like(rewards, dtype=np.float32)
    returns = np.zeros_like(rewards, dtype=np.float32)

    for t in reversed(range(T)):
        if t == T - 1:
            next_value = last_val
        else:
            next_value = values[t + 1]
        td_error = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = td_error + gamma * lam * gae * (1 - dones[t])
        advantages[t] = gae
        returns[t] = advantages[t] + values[t]

    return torch.tensor(returns, dtype=torch.float32), torch.tensor(advantages, dtype=torch.float32)


###################################################
# 5) PPG-Trainer
###################################################
class PPGTrainer:
    def __init__(
        self,
        net: ActorCritic,
        optimizer: optim.Optimizer,
        clip_eps=CLIP_EPS,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        ent_coef=ENT_COEF,   # <-- jetzt 0.01
        epochs_per_update=EPOCHS_PER_UPDATE,
        aux_epochs=AUX_EPOCHS,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        lam=LAMBDA,
        device=DEVICE,
    ):
        self.net = net.to(device)
        self.optimizer = optimizer
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.ent_coef = ent_coef
        self.epochs_per_update = epochs_per_update
        self.aux_epochs = aux_epochs
        self.batch_size = batch_size
        self.gamma = gamma
        self.lam = lam
        self.device = device

    def ppg_update(self, latent_t, act_t, old_logp_t, returns_t, advantages_t):
        dataset_size = len(latent_t)

        # Phase 1: Policy+Value
        for _ in range(self.epochs_per_update):
            idxs = np.random.permutation(dataset_size)
            start = 0
            while start < dataset_size:
                end = start + self.batch_size
                batch_idxs = idxs[start:end]

                b_latent = latent_t[batch_idxs].to(self.device)
                b_act = act_t[batch_idxs].to(self.device)
                b_old_logp = old_logp_t[batch_idxs].to(self.device)
                b_ret = returns_t[batch_idxs].to(self.device)
                b_adv = advantages_t[batch_idxs].to(self.device)

                logp_new, v_pred = self.net.get_logp_value(b_latent, b_act)

                ratio = torch.exp(logp_new - b_old_logp)
                clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                pg_loss = -torch.min(ratio * b_adv, clipped_ratio * b_adv).mean()

                value_loss = (b_ret - v_pred).pow(2).mean()

                # Optional: Entropie
                # Dist ermitteln:
                mu = self.net.forward_policy(b_latent)
                std = torch.exp(self.net.log_std)
                dist = Normal(mu, std)
                entropy = dist.entropy().sum(dim=-1).mean()

                loss = pg_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                start = end

        # Phase 2: Aux-Value (Policy einfrieren)
        actor_mean_params = list(self.net.mu.parameters())
        actor_logstd_params = [self.net.log_std]
        for p in actor_mean_params + actor_logstd_params:
            p.requires_grad = False

        for _ in range(self.aux_epochs):
            idxs = np.random.permutation(dataset_size)
            start = 0
            while start < dataset_size:
                end = start + self.batch_size
                batch_idxs = idxs[start:end]

                b_latent = latent_t[batch_idxs].to(self.device)
                b_ret = returns_t[batch_idxs].to(self.device)

                v_pred_aux = self.net.forward_value(b_latent).squeeze(-1)
                aux_value_loss = (b_ret - v_pred_aux).pow(2).mean()

                self.optimizer.zero_grad()
                aux_value_loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                start = end

        # Auftauen
        for p in actor_mean_params + actor_logstd_params:
            p.requires_grad = True

    def train_one_iteration(self, rollouts):
        latent_obs = rollouts["latent_obs"]
        act_array = rollouts["actions"]
        rew_array = rollouts["rewards"]
        done_array = rollouts["dones"]
        val_array = rollouts["values"]
        old_logp_array = rollouts["logp"]
        last_val = rollouts["last_val"]

        returns_t, advantages_t = compute_gae_returns(
            rew_array, val_array, done_array,
            gamma=self.gamma, lam=self.lam, last_val=last_val
        )
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        latent_t = torch.as_tensor(latent_obs, dtype=torch.float32)
        act_t = torch.as_tensor(act_array, dtype=torch.float32)
        old_logp_t = torch.as_tensor(old_logp_array, dtype=torch.float32)

        self.ppg_update(latent_t, act_t, old_logp_t, returns_t, advantages_t)


###################################################
# 6) ReplayBuffer für VAE
###################################################
class VAEFrameBuffer:
    """
    Speichert die letzten max_size Graustufen-Frames.
    """
    def __init__(self, max_size=30000):
        self.max_size = max_size
        self.frames = []

    def add_frame(self, frame):
        # frame: [96, 96, 1] Graustufen
        if len(self.frames) >= self.max_size:
            self.frames.pop(0)
        self.frames.append(frame)

    def get_all_frames(self):
        return self.frames

    def size(self):
        return len(self.frames)


###################################################
# 7) Rollout-Funktionen
###################################################
def to_grayscale(frame_rgb):
    """
    frame_rgb: [96, 96, 3], Werte 0..255
    => Rückgabe: [96, 96, 1], Graustufe
    """
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)  # [96,96]
    gray = np.expand_dims(gray, axis=-1)               # [96,96,1]
    return gray


def collect_rollout_for_vae(env, vae_frame_buffer, steps=1000):
    """
    Sammelt Frames per Zufallsaktion in Graustufen.
    """
    obs = env.reset()[0]
    for _ in range(steps):
        # Convert to grayscale
        gray_frame = to_grayscale(obs)
        vae_frame_buffer.add_frame(gray_frame)

        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            obs = env.reset()[0]


def collect_car_racing_rollout(env, vae, net, rollout_steps=2048, device=DEVICE):
    """
    Sammelt Rollout für PPG, Graustufen + VAE -> latenter State.
    """
    obs_buffer = []
    latent_buffer = []
    act_buffer = []
    logp_buffer = []
    val_buffer = []
    rew_buffer = []
    done_buffer = []

    obs = env.reset()[0]
    total_reward = 0.0

    for t in range(rollout_steps):
        # 1) Graustufen
        gray_frame = to_grayscale(obs)

        # 2) VAE-Encoden
        obs_tensor = torch.tensor(gray_frame, dtype=torch.float32).permute(2,0,1).unsqueeze(0).to(device) / 255.0
        with torch.no_grad():
            mu, logvar = vae.encode(obs_tensor)
            z = vae.reparameterize(mu, logvar)  # [1, latent_dim]

        # 3) Aktion via Policy
        with torch.no_grad():
            net.log_std.data = torch.clamp(net.log_std.data, -2.0, 1.0)
            mu_ac = net.forward_policy(z)
            std_ac = torch.exp(net.log_std)
            dist_ac = Normal(mu_ac, std_ac)
            action = dist_ac.sample()
            logp = dist_ac.log_prob(action).sum(dim=-1)
            value = net.forward_value(z).squeeze(-1)

        action_np = action.cpu().numpy()[0]
        # Clamping [steer, gas, brake]
        steer = np.clip(action_np[0], -1.0, 1.0)
        gas   = np.clip(action_np[1],  0.0, 1.0)
        brake = np.clip(action_np[2], 0.0, 1.0)
        action_clamped = np.array([steer, gas, brake], dtype=np.float32)

        obs_new, reward, done, truncated, info = env.step(action_clamped)
        total_reward += reward

        # 4) Speichern
        obs_buffer.append(gray_frame)  # optional, falls wir Frames speichern wollen
        latent_buffer.append(z.squeeze(0).cpu().numpy())
        act_buffer.append(action_np)
        logp_buffer.append(logp.item())
        val_buffer.append(value.item())
        rew_buffer.append(reward)
        done_buffer.append(float(done))

        obs = obs_new
        if done or truncated:
            obs = env.reset()[0]

    # Bootstrap
    if done:
        last_val = 0.0
    else:
        gray_frame = to_grayscale(obs)
        obs_tensor = torch.tensor(gray_frame, dtype=torch.float32).permute(2,0,1).unsqueeze(0).to(device) / 255.0
        with torch.no_grad():
            mu, logvar = vae.encode(obs_tensor)
            z = vae.reparameterize(mu, logvar)
            last_val = net.forward_value(z).item()

    rollouts = {
        "obs": np.array(obs_buffer),  # optional
        "latent_obs": np.array(latent_buffer),
        "actions": np.array(act_buffer),
        "rewards": np.array(rew_buffer, dtype=np.float32),
        "dones": np.array(done_buffer, dtype=np.float32),
        "values": np.array(val_buffer, dtype=np.float32),
        "logp": np.array(logp_buffer, dtype=np.float32),
        "last_val": last_val
    }

    return rollouts, total_reward


###################################################
# 8) Haupt-Trainingsfunktion
###################################################
def train_ppg_with_vae():
    env = gym.make("CarRacing-v2", continuous=True)
    env.action_space.seed(42)
    obs = env.reset(seed=42)[0]

    # 1) VAE + Trainer
    #    => Graustufen => 1 Kanal
    vae = VAE(latent_dim=LATENT_DIM, img_channels=1, img_size=96).to(DEVICE)
    vae_trainer = VAETrainer(vae, lr=VAE_LR, beta_kl=BETA_KL, device=DEVICE)
    vae_buffer = VAEFrameBuffer(max_size=VAE_BUFFER_SIZE)

    # 2) Buffer füllen mit Zufallsaktionen
    print("[Init] Sammle erste Frames für VAE-Puffer ...")
    collect_rollout_for_vae(env, vae_buffer, steps=5000)
    print(f"[Init] VAE-Puffer gefüllt mit {vae_buffer.size()} Graustufen-Frames.")

    # 3) Erstes VAE-Training
    print("[Init] Trainiere VAE (erster Durchgang) ...")
    frames_init = vae_buffer.get_all_frames()
    vae_trainer.train_vae(frames_init, epochs=VAE_TRAIN_EPOCHS, batch_size=BATCH_SIZE)

    # 4) ActorCritic + PPG
    action_dim = env.action_space.shape[0]  # 3
    net = ActorCritic(latent_dim=LATENT_DIM, action_dim=action_dim, hidden_size=256)
    optimizer = optim.Adam(net.parameters(), lr=LR)

    ppg_trainer = PPGTrainer(
        net,
        optimizer,
        clip_eps=CLIP_EPS,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        ent_coef=ENT_COEF,        # 0.01
        epochs_per_update=EPOCHS_PER_UPDATE,
        aux_epochs=AUX_EPOCHS,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        lam=LAMBDA,
        device=DEVICE
    )

    # 5) Haupt-Trainingsschleife
    for iteration in range(MAX_ITERATIONS):
        print(f"=== Iteration {iteration+1}/{MAX_ITERATIONS} ===")

        # a) Rollout mit aktuellem Policy+VAE
        rollouts, ep_reward = collect_car_racing_rollout(env, vae, net, rollout_steps=ROLLOUT_STEPS, device=DEVICE)
        print(f"[Iteration {iteration+1}] Rollout Steps: {len(rollouts['rewards'])}, Reward={ep_reward:.2f}")

        # b) Frames in VAE-Puffer
        for frame_gray in rollouts["obs"]:
            vae_buffer.add_frame(frame_gray)

        # c) VAE-Training erneut
        frames_now = vae_buffer.get_all_frames()
        vae_trainer.train_vae(frames_now, epochs=VAE_TRAIN_EPOCHS, batch_size=BATCH_SIZE)

        # d) PPG-Update
        ppg_trainer.train_one_iteration(rollouts)

        print(f"[Iteration {iteration+1}] Fertig. EpisodeReward={ep_reward:.2f}")

    env.close()


###################################################
# 9) Entry Point
###################################################
if __name__ == "__main__":
    train_ppg_with_vae()
