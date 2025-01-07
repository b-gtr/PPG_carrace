import gymnasium as gym
from shimmy import GymV26CompatibilityV0
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

def make_eval_env():
    """
    Erstellt ein CarRacing-v3-Environment im alten Gym-API-Stil,
    damit Stable Baselines3 (SB3) problemlos damit umgehen kann.
    Wird auf 'human' gerendert, damit man das Spielgeschehen sieht.
    """
    old_env = GymV26CompatibilityV0(
        "CarRacing-v3",
        make_kwargs={"render_mode": "human"}
    )
    return Monitor(old_env)

if __name__ == "__main__":
    # Ein bereits trainiertes PPO-Modell laden (Pfad anpassen, falls nötig)
    model = PPO.load("ppo_car_racing_v3")

    # Evaluierungs-Environment erstellen
    eval_env = make_eval_env()
    obs, _ = eval_env.reset()

    # Anzahl der Schritte oder Episoden kann beliebig erhöht werden
    total_eval_steps = 10_000

    for step in range(total_eval_steps):
        # Aktion aus dem Modell vorhersagen
        action, _states = model.predict(obs, deterministic=True)

        # Schritt im Environment ausführen
        obs, reward, done, truncated, info = eval_env.step(action)

        # Rendern (Ausgabe auf Bildschirm)
        eval_env.render()

        # Falls Episode beendet oder getruncated -> Environment neu starten
        if done or truncated:
            obs, _ = eval_env.reset()

    eval_env.close()
