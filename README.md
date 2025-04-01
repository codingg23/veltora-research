# veltora-research

Early research experiments for Veltora's multi-agent intelligence layer.

This is where I dump experimental work — things that might become core product features, or might get thrown away. Not production code. Lots of things in here are half-finished or failed experiments that I'm keeping for reference.

Current focus: can multi-agent RL systems make better procurement and capacity planning decisions than rules-based systems? Early results are... mixed. But interesting.

## What's Here

### Supply Chain Procurement Agent (`agents/supply_chain_agent.py`)

Reinforcement learning agent that learns procurement timing for data centre hardware.

The problem: when should you order new hardware given:
- Current utilisation trends
- Historical lead times (noisy, variable)
- Holding costs vs stockout risk
- Budget cycles / quarterly pressure

Framed as a Markov Decision Process:
- **State**: current inventory levels, open orders, utilisation forecast, days until quarter end
- **Actions**: order quantity for each hardware type (discrete: 0, 1, 2, 4, 8 units)
- **Reward**: negative cost (holding cost + stockout cost + expedite premium)

Algorithm: PPO (Proximal Policy Optimisation) via `stable-baselines3`.

Current status: converges on the training environment but overfit heavily to the synthetic data patterns. Real procurement has way more structure (budget gates, vendor negotiations, org politics) that isn't captured in the env. This is a research direction, not a product decision yet.

### Operations Orchestrator Agent (`agents/ops_agent.py`)

Prototype multi-agent system for coordinating operational responses to infrastructure events.

Basically: given a detected anomaly (e.g., "CRAC-04 efficiency degrading"), generate and prioritise a set of response actions across different teams (facilities, capacity, procurement, networking).

Current approach: LLM-based agent with tool use for querying the simulation model and generating action plans. This is more of a planning agent than a learning one — no RL yet.

### Experiments (`experiments/`)

- `ppo_procurement_v1.py` — first PPO attempt, bad reward shaping, didn't work
- `ppo_procurement_v2.py` — better, added holding cost properly
- `curriculum_learning_test.py` — trying to train on easy scenarios first
- `reward_shaping_ablation.ipynb` — comparing different reward formulations
- `lead_time_forecast_experiments.ipynb` — time-series forecasting for lead times specifically

## Setup

```bash
git clone https://github.com/aryan-veltora/veltora-research
cd veltora-research
pip install -r requirements.txt

# train the procurement agent
python agents/supply_chain_agent.py --train --timesteps 500000

# evaluate
python agents/supply_chain_agent.py --eval --checkpoint ./models/ppo_procurement_v2.zip
```

## Research Log

**2024-09-15**: First PPO attempt on procurement env. Reward function was wrong — wasn't penalising stockouts hard enough, agent learned to just never order anything.

**2024-10-20**: Fixed reward shaping. Agent now learns a sensible base policy but is very conservative (orders too early, high holding costs). This might actually be fine for GPU hardware where stockout cost is enormous.

**2024-11-08**: Tried curriculum learning — start with simple single-hardware scenarios, gradually add complexity. Training is more stable but still overfit to synthetic data distribution. Need real data.

**2025-01-14**: Shifted ops agent to LLM-based planning instead of RL. Much more interpretable, easier to debug, handles novel situations better. RL might still be useful for the procurement timing specifically.

**2025-03-02**: Started lead time forecasting experiments. Treating vendor lead time as a time-series forecasting problem. Interesting result: supplier-specific features matter a lot (some vendors are consistently faster/slower than their quotes).

## Disclaimer

This is research code. It's messy. It has experiments that didn't work. That's intentional — I want to keep the history of what was tried, not just the things that worked.

If you're looking at this trying to understand Veltora's current product: the research here feeds into product decisions but isn't the product itself.
