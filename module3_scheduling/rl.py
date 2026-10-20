"""Reinforcement-learning scheduler (PPO), after An (2025).

Each day the agent sees what the other policies see (Module 2's smoothed loss
estimate, the learned soiling rate, days since the last wash, recent energy,
a 3-day rain forecast and the season) and chooses: wait or wash.
Reward = -(revenue lost to soiling today + washing cost today), in units of
one wash, so maximising reward minimises the total cost.

Trained on the training years of all five sites (random site, random
one-year window per episode) and tested on the held-out years.

Usage:  python -m module3_scheduling.rl
"""
import gymnasium as gym
import numpy as np

from . import config
from .policies import Policy
from .sim import LossEstimate, Module2Sensor, load_site, rain_clean_fraction, calibrate_rate_constant

N_FEATURES = 10


def features(obs, since):
    doy = obs["date"].dayofyear / 365.25 * 2 * np.pi
    rain = np.clip(np.pad(obs["rain_forecast_mm"], (0, config.RAIN_FORECAST_DAYS))[:config.RAIN_FORECAST_DAYS] / 10, 0, 3)
    value = ((obs["estimate"] - config.RESIDUAL_AFTER_CLEANING_PCT) / 100 * obs["mean_energy_kwh"]
             * config.TARIFF_INR_PER_KWH * 30 / config.CLEANING_COST_INR)
    return np.array([
        obs["estimate"] / 20, obs["rate"], min(since, 90) / 30, obs["mean_energy_kwh"] / 6000,
        *rain, np.sin(doy), np.cos(doy), value,
    ], dtype=np.float32)


class SoilingEnv(gym.Env):
    def __init__(self, sites, episode_days=365, seed=0):
        self.sites, self.episode_days = sites, episode_days
        self.observation_space = gym.spaces.Box(-10, 10, (N_FEATURES,), np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self.rng = np.random.default_rng(seed)

    def _obs(self):
        s, t = self.site, self.t
        reading = self.sensor.read(self.loss)
        est = self.est.update(reading, self.rained)
        f = config.RAIN_FORECAST_DAYS
        self.o = {
            "date": s.dates[t], "estimate": est, "rate": self.est.rate,
            "mean_energy_kwh": float(s.energy_kwh[max(0, t - 30):t + 1].mean()),
            "rain_forecast_mm": s.rain_mm[t:t + f].copy(),
        }
        return features(self.o, self.since)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.site = self.sites[self.rng.integers(len(self.sites))]
        self.t = int(self.rng.integers(0, len(self.site.dates) - self.episode_days))
        self.end = self.t + self.episode_days
        self.sensor, self.est = Module2Sensor(self.rng), LossEstimate()
        self.loss = float(self.rng.uniform(0, 5))
        self.since, self.rained = int(self.rng.integers(0, 20)), False
        return self._obs(), {}

    def step(self, action):
        s, t, C = self.site, self.t, config.CLEANING_COST_INR
        cost = 0.0
        if action == 1:
            self.loss = config.RESIDUAL_AFTER_CLEANING_PCT
            self.est.cleaned()
            cost += C
            self.since = 0
        else:
            self.since += 1
        cost += s.energy_kwh[t] * self.loss / 100 * config.TARIFF_INR_PER_KWH
        self.rained = s.rain_mm[t] >= config.RAIN_PARTIAL_MM
        dry = not self.rained
        self.loss = min(self.loss + (s.rate_pct[t] if dry else 0), config.MAX_SOILING_LOSS_PCT)
        self.loss *= 1 - rain_clean_fraction(s.rain_mm[t])
        self.t += 1
        done = self.t >= self.end
        obs = np.zeros(N_FEATURES, np.float32) if done else self._obs()
        return obs, -cost / C, done, False, {}


class PPOPolicy(Policy):
    name = "PPO reinforcement learning"

    def __init__(self, model):
        self.model = model

    def _decide(self, obs):
        a, _ = self.model.predict(features(obs, self.since), deterministic=True)
        return bool(a)


MODEL_PATH = config.OUT_DIR / "ppo_scheduler.zip"


def train():
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env

    k = calibrate_rate_constant()
    sites = [load_site(key, k).years(config.TRAIN_YEARS) for key in config.SITES]
    env = make_vec_env(lambda: SoilingEnv(sites, seed=np.random.randint(1 << 30)), n_envs=8, seed=config.SEED)
    model = PPO("MlpPolicy", env, n_steps=1024, batch_size=512, gamma=0.995, gae_lambda=0.95,
                learning_rate=3e-4, ent_coef=0.005, policy_kwargs={"net_arch": [64, 64]},
                seed=config.SEED, verbose=0)
    step = config.PPO_TIMESTEPS // 8
    for i in range(1, 9):
        model.learn(step, reset_num_timesteps=False)
        ep = [e["r"] for e in model.ep_info_buffer]
        print(f"{i * step:7d} steps | mean episode cost {-np.mean(ep):6.2f} washes-worth", flush=True)
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Saved {MODEL_PATH}")


def load_policy():
    from stable_baselines3 import PPO
    return PPOPolicy(PPO.load(MODEL_PATH, device="cpu"))


if __name__ == "__main__":
    train()
