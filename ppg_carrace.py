import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

##########################################################################
# 1) VAE-Modell für Dimensionality Reduction
##########################################################################
class VAE(nn.Module):
    """
    Einfaches VAE (Encoder+Decoder), um CarRacing-Bilder
    (z.B. 96x96x3) in einen niedrigdimensionalen Latent Space zu projizieren.
    """
    def __init__(self, latent_dim=32, img_channels=3, img_size=96):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_channels = img_channels
        self.img_size = img_size

        # -------- Encoder: conv -> fc mu, logvar
        self.enc_conv = nn.Sequential(
            nn.Conv2d(img_channels, 32, 4, 2, 1),  # [32, 48, 48]
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),            # [64, 24, 24]
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),           # [128, 12, 12]
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),          # [256, 6, 6]
            nn.ReLU(),
        )
        # Letzte Convolution-Feature-Größe: 256 * 6 * 6 = 9216
        self.enc_fc_mu = nn.Linear(256*6*6, latent_dim)
        self.enc_fc_logvar = nn.Linear(256*6*6, latent_dim)

        # -------- Decoder: fc -> deconv
        self.dec_fc = nn.Linear(latent_dim, 256*6*6)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # [128, 12, 12]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # [64, 24, 24]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # [32, 48, 48]
            nn.ReLU(),
            nn.ConvTranspose2d(32, img_channels, 4, 2, 1),  # [3, 96, 96]
            nn.Sigmoid(),  # Werte in [0, 1]
        )

    def encode(self, x):
        """
        x: [B, 3, 96, 96]  -> mu, logvar
        """
        h = self.enc_conv(x)
        h = h.view(h.size(0), -1)  # Flatten
        mu = self.enc_fc_mu(h)
        logvar = self.enc_fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Z ~ mu + std * epsilon
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """
        z: [B, latent_dim] -> Rekonstruiertes Bild
        """
        h = self.dec_fc(z)
        h = h.view(h.size(0), 256, 6, 6)
        x_recon = self.dec_conv(h)
        return x_recon

    def forward(self, x):
        """
        VAE-Gesamtforward: x -> mu, logvar -> z -> x_recon
        Return: x_recon, mu, logvar
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


##########################################################################
# 2) Trainer für den VAE
##########################################################################
class VAETrainer:
    def __init__(
        self,
        vae: VAE,
        lr=1e-3,
        beta_kl=1.0,  # Gewichtung KL
    ):
        """
        Einfacher Trainer für VAE.
        :param vae: Instanz des VAE
        :param lr: Lernrate
        :param beta_kl: Gewichtung der KL-Divergenz im Loss
        """
        self.vae = vae
        self.optimizer = optim.Adam(self.vae.parameters(), lr=lr)
        self.beta_kl = beta_kl

    def loss_function(self, x, x_recon, mu, logvar):
        """
        Rekonstruktions-Loss + Beta-KL.
        MSE oder BCE als Reconstruction Loss. Hier z.B. MSE.
        """
        recon_loss = nn.functional.mse_loss(x_recon, x, reduction="sum")
        # KL-Divergenz
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.beta_kl * kl_loss

    def train_epoch(self, dataloader, device="cpu"):
        """
        Training für einen Durchgang über das DataLoader.
        """
        self.vae.train()
        total_loss = 0.0
        for i, batch in enumerate(dataloader):
            x = batch.to(device)  # [B, 3, 96, 96]
            self.optimizer.zero_grad()
            x_recon, mu, logvar = self.vae(x)
            loss = self.loss_function(x, x_recon, mu, logvar)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader.dataset)


##########################################################################
# 3) ActorCritic: nutzt latent_dim statt roher Pixel
##########################################################################
class ActorCritic(nn.Module):
    def __init__(self, latent_dim, action_dim=3, hidden_size=256):
        """
        Für CarRacing sind die Actions standardmäßig 3-dimensional:
          [steering, gas, brake].
        """
        super().__init__()
        self.latent_dim = latent_dim

        # Policy-Backbone (MLP)
        self.policy_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        # Value-Backbone (MLP)
        self.value_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Actor-Kopf (Gaussian Policy, hier ggf. Steering in [-1,1], Gas/Brake in [0,1])
        # Wir machen es für alle 3 Actions Gauss-Parameter:
        self.mu = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic-Kopf
        self.v = nn.Linear(hidden_size, 1)

    def forward_policy(self, latent):
        """
        latent: [B, latent_dim]
        """
        p = self.policy_net(latent)
        mu = self.mu(p)
        return mu

    def forward_value(self, latent):
        """
        latent: [B, latent_dim]
        """
        v_ = self.value_net(latent)
        val = self.v(v_)
        return val

    def get_logp_value(self, latent, action):
        """
        Berechne Log-Prob und Value anhand des latenten Zustands.
        """
        with torch.no_grad():
            self.log_std.data = torch.clamp(self.log_std.data, -2.0, 1.0)

        mu = self.forward_policy(latent)
        std = torch.exp(self.log_std)
        dist = Normal(mu, std)

        logp = dist.log_prob(action).sum(dim=-1)
        value = self.forward_value(latent).squeeze(-1)
        return logp, value


##########################################################################
# 4) Compute GAE für PPG
##########################################################################
def compute_gae_returns(rewards, values, dones, gamma=0.99, lam=0.95, last_val=0.0):
    """
    GAE-like Vorteile + Returns.
    """
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


##########################################################################
# 5) PPG-Trainer
##########################################################################
class PPGTrainer:
    def __init__(
        self,
        net: ActorCritic,
        optimizer: optim.Optimizer,
        clip_eps=0.2,
        vf_coef=0.5,
        max_grad_norm=0.5,
        ent_coef=0.0,
        epochs_per_update=10,
        aux_epochs=2,
        batch_size=64,
        gamma=0.99,
        lam=0.95,
    ):
        self.net = net
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

    def ppg_update(self, latent_t, act_t, old_logp_t, returns_t, advantages_t):
        """
        Phase 1: Policy+Value Update (ähnlich PPO)
        Phase 2: Aux-Update (nur Value)
        """
        dataset_size = len(latent_t)

        # Phase 1: Policy+Value
        for _ in range(self.epochs_per_update):
            idxs = np.random.permutation(dataset_size)
            start = 0
            while start < dataset_size:
                end = start + self.batch_size
                batch_idxs = idxs[start:end]

                b_latent = latent_t[batch_idxs]
                b_act = act_t[batch_idxs]
                b_old_logp = old_logp_t[batch_idxs]
                b_ret = returns_t[batch_idxs]
                b_adv = advantages_t[batch_idxs]

                logp_new, v_pred = self.net.get_logp_value(b_latent, b_act)

                ratio = torch.exp(logp_new - b_old_logp)
                clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)

                # Policy Loss
                pg_loss = -torch.min(ratio * b_adv, clipped_ratio * b_adv).mean()

                # Value Loss
                value_loss = (b_ret - v_pred).pow(2).mean()

                loss = pg_loss + self.vf_coef * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                start = end

        # Phase 2: Aux-Update (Value-only)
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

                b_latent = latent_t[batch_idxs]
                b_ret = returns_t[batch_idxs]

                v_pred_aux = self.net.forward_value(b_latent).squeeze(-1)
                aux_value_loss = (b_ret - v_pred_aux).pow(2).mean()

                self.optimizer.zero_grad()
                aux_value_loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                start = end

        # Policy wieder "auftauen"
        for p in actor_mean_params + actor_logstd_params:
            p.requires_grad = True

    def train_one_iteration(self, rollouts):
        """
        Führt GAE + ppg_update durch.
        """
        # rollouts enthält bspw.:
        # "latent_obs", "actions", "rewards", "dones", "values", "logp", "last_val"
        latent_obs = rollouts["latent_obs"]
        act_array = rollouts["actions"]
        rew_array = rollouts["rewards"]
        done_array = rollouts["dones"]
        val_array = rollouts["values"]
        old_logp_array = rollouts["logp"]
        last_val = rollouts["last_val"]

        # 1) Returns & Vorteile
        returns_t, advantages_t = compute_gae_returns(
            rew_array, val_array, done_array,
            gamma=self.gamma, lam=self.lam,
            last_val=last_val
        )
        # Normalisierung
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # 2) Tensors
        latent_t = torch.as_tensor(latent_obs, dtype=torch.float32)
        act_t = torch.as_tensor(act_array, dtype=torch.float32)
        old_logp_t = torch.as_tensor(old_logp_array, dtype=torch.float32)

        # 3) Update
        self.ppg_update(latent_t, act_t, old_logp_t, returns_t, advantages_t)


##########################################################################
# 6) Beispiel-Integration mit CarRacing-v2
##########################################################################
def collect_car_racing_rollout(env, vae, net, rollout_steps=1024):
    """
    Beispielhafte Rollout-Funktion für CarRacing-v2:
      - Nimmt rohes Bild, encodet es über den VAE in latent_obs
      - Wählt Aktion über net (z.B. stochastisch -> Normal)
      - Speichert (latent_obs, action, reward, done, value, logp)
    """
    obs_buffer = []
    latent_buffer = []
    act_buffer = []
    logp_buffer = []
    val_buffer = []
    rew_buffer = []
    done_buffer = []

    obs = env.reset()[0]  # CarRacing-v2 liefert (obs, info)
    total_reward = 0.0

    for t in range(rollout_steps):
        # obs: [96, 96, 3], float in [0, 255], optional: /255.0 Normalisierung
        # -> B, C, H, W => [1, 3, 96, 96]
        obs_tensor = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0

        # 1) In latent space encoden
        with torch.no_grad():
            mu, logvar = vae.encode(obs_tensor)
            latent = vae.reparameterize(mu, logvar)  # [1, latent_dim]
            # Alternativ: Nur mu als State => latente Deterministik

        # 2) Aktion aus ActorCritic
        with torch.no_grad():
            net.log_std.data = torch.clamp(net.log_std.data, -2.0, 1.0)
            mu_ac = net.forward_policy(latent)
            std_ac = torch.exp(net.log_std)
            dist_ac = Normal(mu_ac, std_ac)
            action = dist_ac.sample()
            logp = dist_ac.log_prob(action).sum(dim=-1)
            value = net.forward_value(latent).squeeze(-1)

        # 3) Aktion ausführen
        action_np = action.numpy()[0]
        # CarRacing erwartet [steer, gas, brake] jeweils in [-1..1] oder [0..1].
        # Evtl. clampen: steer in [-1,1], gas/brake in [0,1]
        steer = np.clip(action_np[0], -1.0, 1.0)
        gas = np.clip(action_np[1], 0.0, 1.0)
        brake = np.clip(action_np[2], 0.0, 1.0)
        action_clamped = np.array([steer, gas, brake], dtype=np.float32)

        obs_new, reward, done, truncated, info = env.step(action_clamped)
        total_reward += reward

        # 4) Daten fürs Training speichern
        obs_buffer.append(obs)
        latent_buffer.append(latent.squeeze(0).numpy())
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
        obs_tensor = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
        with torch.no_grad():
            mu, logvar = vae.encode(obs_tensor)
            latent = vae.reparameterize(mu, logvar)
            last_val = net.forward_value(latent).item()

    rollouts = {
        "obs": np.array(obs_buffer),             # Nur zur Demo
        "latent_obs": np.array(latent_buffer),   # <- wichtig fürs Training
        "actions": np.array(act_buffer),
        "rewards": np.array(rew_buffer, dtype=np.float32),
        "dones": np.array(done_buffer, dtype=np.float32),
        "values": np.array(val_buffer, dtype=np.float32),
        "logp": np.array(logp_buffer, dtype=np.float32),
        "last_val": last_val
    }

    return rollouts, total_reward


def train_ppg_with_vae():
    """
    End-to-End-Beispiel:
      - 1) VAE bauen und offline vortrainieren (bzw. laden)
      - 2) PPG-Training auf CarRacing mit latent states
    """
    # 1) Environment
    env = gym.make("CarRacing-v2", continuous=True)
    action_dim = env.action_space.shape[0]  # 3

    # 2) VAE initialisieren & laden (oder selbst trainieren)
    latent_dim = 32
    vae = VAE(latent_dim=latent_dim, img_channels=3, img_size=96)
    # Hier: Laden aus Datei oder offline trainieren => 
    # vae.load_state_dict(torch.load("car_racing_vae.pth"))

    # 3) ActorCritic mit latent_dim
    net = ActorCritic(latent_dim=latent_dim, action_dim=action_dim, hidden_size=256)
    optimizer = optim.Adam(net.parameters(), lr=3e-4)
    trainer = PPGTrainer(net, optimizer, clip_eps=0.2, vf_coef=0.5, epochs_per_update=10, aux_epochs=2)

    # Training-Loop
    max_iterations = 100
    for it in range(max_iterations):
        # Rollouts sammeln (env + vae)
        rollouts, episode_reward = collect_car_racing_rollout(env, vae, net, rollout_steps=1024)
        
        # Update
        trainer.train_one_iteration(rollouts)

        print(f"Iteration {it+1}/{max_iterations}, Episode Reward: {episode_reward:.2f}")

    env.close()


if __name__ == "__main__":
    # Beispielhafter Aufruf
    train_ppg_with_vae()
