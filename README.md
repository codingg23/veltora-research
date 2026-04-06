# veltora-research

Early research for Veltora's multi agent intelligence layer.

This is where I dump experimental work. Some of it will become product features, a lot of it will get thrown away. Not production code - lots of half finished experiments that I'm keeping for reference.

Current focus: can multi agent RL systems make better procurement and capacity planning decisions than rules based systems? Early results are mixed but interesting.

## What's Here

### Supply Chain Procurement Agent (`agents/supply_chain_agent.py`)

RL agent that learns procurement timing for data centre hardware.

The problem: when should you order new hardware given current utilisation trends, historical lead times, holding costs vs stockout risk, and budget cycles?

Framed as an MDP:
- **State**: inventory levels, open orders, utilisation forecast, days until quarter end
- **Actions**: order quantity per hardware type (discrete: 0, 1, 2, 4, 8, 16 units)
- **Reward**: negative cost (holding cost + stockout cost + expedite premium)

Algorithm: PPO via stable-baselines3.

Current status: converges on the training env but overfit to the synthetic data patterns. Real procurement has way more structure (budget gates, vendor negotiations, org politics) that isn't in the env. Still a useful research direction.

### Operations Orchestrator Agent (`agents/ops_agent.py`)

Prototype multi agent system for coordinating operational responses to infrastructure events.

Given a detected anomaly (e.g. "CRAC-04 efficiency degrading"), it generates and prioritises response actions across different teams (facilities, capacity, procurement, networking).

Uses an LLM with tool calling for querying the simulation model and generating action plans. More of a planning agent than a learning one - no RL here.

### Experiments (`experiments/`)

- `ppo_procurement_v1.py` - first PPO attempt, reward function was wrong, didn't work
- `ppo_procurement_v2.py` - better, added holding cost properly
- `curriculum_learning_test.py` - trying to train on easy scenarios first
- `reward_shaping_ablation.ipynb` - comparing different reward formulations
- `lead_time_forecaster.py` - GBM model for predicting vendor lead times

## Setup

```bash
git clone https://github.com/codingg23/veltora-research
cd veltora-research
pip install -r requirements.txt

# train the procurement agent
python agents/supply_chain_agent.py --train --timesteps 500000

# evaluate
python agents/supply_chain_agent.py --eval --checkpoint ./models/ppo_procurement_v2.zip
```

## Research Log

**2024-09-15**: First PPO attempt. Reward function was wrong - wasn't penalising stockouts hard enough so the agent learned to just never order anything.

**2024-10-20**: Fixed reward shaping. Agent now learns a sensible base policy but is very conservative (orders too early, high holding costs). Probably fine for GPU hardware where stockout cost is massive.

**2024-11-08**: Tried curriculum learning - start with simple single-hardware scenarios, gradually add complexity. Training is more stable but still overfit to the synthetic distribution. Need real data.

**2025-01-14**: Shifted ops agent to LLM-based planning instead of RL. Much more interpretable, easier to debug, handles novel situations better. RL might still be worth it for the procurement timing problem specifically.

**2025-03-02**: Started lead time forecasting experiments. Treating vendor lead time as a time series problem. Main finding: supplier specific features matter a lot, some vendors are consistently faster or slower than their quotes.

## Disclaimer

This is research code. It is messy. It has experiments that didn't work. I'm keeping the history of what was tried, not just what worked.

If you're looking at this to understand Veltora's current product: the research here feeds into product decisions but isn't the product itself.
