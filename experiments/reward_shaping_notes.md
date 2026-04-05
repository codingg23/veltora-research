# Reward Shaping Experiments  -  Procurement Agent

Notes from trying different reward formulations for the PPO procurement agent.
Keeping this as markdown because it's more of a research diary than code.

## v1  -  Simple holding + stockout cost (failed)

```python
reward = -(holding_cost + stockout_cost)
```

Problem: stockout cost only applied when inventory hit zero.
By then it's too late  -  lead times are 90-120 days.
The agent learned to never order anything because the penalty was so delayed.

Result: agent always had zero inventory, got penalised constantly,
never learned to pre-order.

## v2  -  Buffer threshold penalty (current)

```python
buffer_threshold = demand * 7  # 7-day safety stock
if inv < buffer_threshold:
    shortage_frac = 1 - inv / buffer_threshold
    cost += shortage_frac * stockout_cost_per_day
```

This applies a graduated penalty as inventory approaches zero,
not just when it hits zero. The gradient is much smoother and
the agent can actually learn from it.

Result: agent learns to maintain a safety buffer. Pre-orders
GPU servers ~85 days out (slightly conservative vs optimal ~100 days).
Still overfit to synthetic data distribution.

## v3 ideas  -  not tried yet

- Add an "expedite cost" component: penalise orders placed within 30 days of projected need
  (forces the agent to plan ahead, not just react)
- Reward shaping via potential functions (Ng et al.) to guide early training
- Asymmetric cost: weight GPU stockout 10x over standard server (reflects business reality)

## Discount factor observations

gamma=0.99 was too high  -  agent discounted near-term holding costs too heavily
and learned to order massive quantities infrequently (which looks bad).

gamma=0.995 with longer rollouts (n_steps=2048) worked better.
The procurement horizon is long so you do need high gamma, just not too high.

## Data distribution problem

The synthetic data has very consistent lead times (normal distribution, fixed params).
Real procurement has:
- Supplier-specific lead time distributions (some always late, some always early)
- Correlated delays (supply crunches affect all GPU vendors simultaneously)
- Occasional windfalls (cancelled orders freed up early)

None of this is in the training env. The agent is essentially memorising
the synthetic distribution rather than learning general procurement strategy.

Next step: build a harder env with more variance, vendor-specific distributions,
and occasional exogenous shocks (supply crunch event mid-episode).
