"""
supply_chain_agent.py

PPO-based procurement agent for data centre hardware.

Trying to learn: given current state (inventory, utilisation trends,
open orders, budget), what's the optimal order quantity for each
hardware category?

This is v2 — v1 had a broken reward function where the agent just
never ordered anything because stockout penalties were too delayed
relative to holding cost penalties. Fixed that with a proper
'inventory shortage cost' that applies from the moment utilisation
exceeds available capacity.

Still lots of problems:
- The environment is too clean compared to real procurement
- Budget gates (quarterly purchasing windows) aren't modelled
- Vendor relationships / negotiation isn't modelled
- The agent can't "call a vendor" — it just submits orders

But: it does learn to pre-order GPU servers ~90 days before
projected shortage, which is roughly right given current lead times.
That's the core behaviour we want.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# try to import sb3, warn if not available
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import EvalCallback
    SB3_AVAILABLE = True
except ImportError:
    logger.warning("stable-baselines3 not installed. Training unavailable. `pip install stable-baselines3`")
    SB3_AVAILABLE = False


# Hardware categories with their characteristics
HARDWARE_TYPES = {
    "server_standard": {"mean_lead_days": 45, "holding_cost_per_unit_per_day": 2.0,  "stockout_cost_per_day": 50.0},
    "server_gpu":      {"mean_lead_days": 120, "holding_cost_per_unit_per_day": 15.0, "stockout_cost_per_day": 500.0},
    "networking":      {"mean_lead_days": 35,  "holding_cost_per_unit_per_day": 0.5,  "stockout_cost_per_day": 200.0},
}
N_HARDWARE = len(HARDWARE_TYPES)
HW_NAMES = list(HARDWARE_TYPES.keys())

# Discrete action space: how many units to order per hardware type
ORDER_OPTIONS = [0, 1, 2, 4, 8, 16]  # units per order


@dataclass
class ProcurementEnvConfig:
    episode_days: int = 365
    initial_inventory: dict = None        # units of each type
    daily_demand_mean: dict = None        # expected daily consumption rate
    lead_time_noise_std_frac: float = 0.25  # lead time noise as fraction of mean
    budget_per_day: float = 10000.0       # daily procurement budget (soft constraint)
    seed: Optional[int] = None

    def __post_init__(self):
        if self.initial_inventory is None:
            self.initial_inventory = {hw: 20 for hw in HW_NAMES}
        if self.daily_demand_mean is None:
            self.daily_demand_mean = {
                "server_standard": 0.3,
                "server_gpu": 0.1,
                "networking": 0.05,
            }


class ProcurementEnv(gym.Env):
    """
    Gymnasium environment for hardware procurement decisions.

    State (observation):
        For each hardware type:
        - current inventory (normalised)
        - units in transit (open orders)
        - days until earliest delivery
        - 30-day utilisation rate (normalised)
        - days since last order (normalised)
    Plus:
        - day of year (cyclically encoded: sin/cos)
        - days until quarter end (normalised)

    Action:
        Discrete: for each hardware type, choose order quantity from ORDER_OPTIONS
        Total action space: len(ORDER_OPTIONS)^N_HARDWARE
        (factorised: one discrete choice per hardware type — would use MultiDiscrete)

    Reward:
        Negative cost = -(holding_cost + stockout_cost + expedite_premium)
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[ProcurementEnvConfig] = None):
        super().__init__()
        self.config = config or ProcurementEnvConfig()
        self.rng = np.random.default_rng(self.config.seed)

        # Observation space: 5 features per hardware type + 4 global
        n_obs = N_HARDWARE * 5 + 4
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_obs,), dtype=np.float32
        )

        # Action space: one discrete choice per hardware type
        self.action_space = spaces.MultiDiscrete([len(ORDER_OPTIONS)] * N_HARDWARE)

        self._inventory = {}
        self._in_transit = {}  # {hw: [(units, eta_day), ...]}
        self._day = 0
        self._episode_cost = 0.0

    def _get_obs(self) -> np.ndarray:
        obs = []
        for hw in HW_NAMES:
            inv = self._inventory.get(hw, 0)
            in_transit_units = sum(u for u, _ in self._in_transit.get(hw, []))
            days_to_delivery = min(
                (eta - self._day for _, eta in self._in_transit.get(hw, [])),
                default=365
            )
            demand = self.config.daily_demand_mean.get(hw, 0.1)
            days_of_supply = inv / max(demand, 0.01)

            # normalise to roughly [-1, 1]
            obs.extend([
                np.clip(inv / 50.0, 0, 1),
                np.clip(in_transit_units / 50.0, 0, 1),
                np.clip(days_to_delivery / 180.0, 0, 1),
                np.clip(days_of_supply / 180.0, 0, 1),
                np.clip(demand / 0.5, 0, 1),
            ])

        # global features
        day_of_year = self._day % 365
        quarter_day = self._day % 91  # days into current quarter
        obs.extend([
            np.sin(2 * np.pi * day_of_year / 365),
            np.cos(2 * np.pi * day_of_year / 365),
            np.clip(quarter_day / 91.0, 0, 1),
            np.clip((91 - quarter_day) / 91.0, 0, 1),
        ])

        return np.array(obs, dtype=np.float32)

    def _compute_step_cost(self) -> float:
        cost = 0.0
        for hw, props in HARDWARE_TYPES.items():
            inv = self._inventory.get(hw, 0)
            demand = self.config.daily_demand_mean.get(hw, 0.1)

            # holding cost
            cost += inv * props["holding_cost_per_unit_per_day"]

            # stockout cost: if inventory < 7-day demand buffer, penalty
            # (this is the fix from v1 — applied every step, not just when it hits zero)
            buffer_threshold = demand * 7
            if inv < buffer_threshold:
                shortage_frac = 1 - inv / max(buffer_threshold, 0.01)
                cost += shortage_frac * props["stockout_cost_per_day"]

        return cost

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._inventory = dict(self.config.initial_inventory)
        self._in_transit = {hw: [] for hw in HW_NAMES}
        self._day = 0
        self._episode_cost = 0.0

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        # place orders
        for i, hw in enumerate(HW_NAMES):
            order_qty = ORDER_OPTIONS[int(action[i])]
            if order_qty > 0:
                props = HARDWARE_TYPES[hw]
                lead_mean = props["mean_lead_days"]
                lead_std = lead_mean * self.config.lead_time_noise_std_frac
                lead_days = max(7, int(self.rng.normal(lead_mean, lead_std)))
                eta = self._day + lead_days
                self._in_transit[hw].append((order_qty, eta))

        # deliver due orders
        for hw in HW_NAMES:
            delivered = []
            remaining = []
            for units, eta in self._in_transit.get(hw, []):
                if eta <= self._day:
                    delivered.append(units)
                else:
                    remaining.append((units, eta))
            self._inventory[hw] = self._inventory.get(hw, 0) + sum(delivered)
            self._in_transit[hw] = remaining

        # consume inventory based on demand
        for hw in HW_NAMES:
            demand = self.config.daily_demand_mean.get(hw, 0.1)
            actual_demand = max(0, self.rng.normal(demand, demand * 0.2))
            self._inventory[hw] = max(0, self._inventory.get(hw, 0) - actual_demand)

        step_cost = self._compute_step_cost()
        self._episode_cost += step_cost
        self._day += 1

        reward = -step_cost / 1000.0  # scale reward

        terminated = self._day >= self.config.episode_days
        obs = self._get_obs()
        info = {"step_cost": step_cost, "episode_cost": self._episode_cost, "day": self._day}

        return obs, reward, terminated, False, info


def train(total_timesteps: int = 500_000, checkpoint_dir: str = "./models/"):
    """Train the PPO procurement agent."""
    if not SB3_AVAILABLE:
        raise ImportError("stable-baselines3 required for training")

    import os
    os.makedirs(checkpoint_dir, exist_ok=True)

    env = make_vec_env(ProcurementEnv, n_envs=4, seed=42)
    eval_env = ProcurementEnv(ProcurementEnvConfig(seed=999))

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.995,       # long horizon — procurement decisions have delayed consequences
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,     # some entropy encourages exploration of order timing
        verbose=1,
        tensorboard_log="./logs/",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=checkpoint_dir,
        log_path="./logs/",
        eval_freq=10_000,
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save(f"{checkpoint_dir}/ppo_procurement_v2")
    logger.info("Training complete")
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--checkpoint", type=str, default="./models/best_model.zip")
    args = parser.parse_args()

    if args.train:
        train(total_timesteps=args.timesteps)

    elif args.eval:
        if not SB3_AVAILABLE:
            print("stable-baselines3 required")
            exit(1)
        model = PPO.load(args.checkpoint)
        env = ProcurementEnv(ProcurementEnvConfig(seed=1234))
        obs, _ = env.reset()
        total_reward = 0
        for _ in range(365):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            total_reward += reward
            if done:
                break
        print(f"Episode reward: {total_reward:.2f}, total cost: {info['episode_cost']:.0f}")
