# Flow AutoTTS Context Pack

Read this file first. It is the intended context budget for this round.

## Allowed First-Pass Reads

- `flow_tts_controller_implementation_spec.md`
- `flow_autotts/controllers/optimal.py`
- `flow_autotts/controllers/baselines.py`
- `flow_autotts/core/state.py`
- `flow_autotts/core/errors.py`
- `flow_autotts/experiments/pickscore_sd35/harness.py`
- `flow_autotts/experiments/pickscore_sd35/env.py`
- recent round summaries listed below

## Write Boundary

- Edit only `flow_autotts/controllers/optimal.py`.
- Do not edit the harness, environment, dataset loader, workflow, tests, logs, model directories, or datasets.
- Keep the controller self-contained. The workflow resets it from the template before every round.

## Context Discipline

- Do not run broad repository scans such as `find .` or unconstrained `rg` from repo root.
- Do not bulk-read raw `history.json`, raw event logs, datasets, `SD_3.5_med/`, `PickScore_v1/`, `flow_grpo/`, `.git/`, or `logs/`.
- If a compact summary points to a concrete anomaly, inspect only the relevant small snippet from that round.
- Prefer targeted reads of the files listed above.

## Template

- `flow_autotts/controllers/optimal.template.py`

## Baseline References

These compact baseline files are injected by the workflow so the proposer can compare by nearest NFE.

### `logs/flow_autotts/pickscore_sd35/train_bestof4_ode_retry2_clean_b128_compact_baseline/aggregate_summary.json`

```json
[
  {
    "action_statistics": {
      "answer": 4.0,
      "forward": 8.0,
      "mean_nfe": 8.0,
      "spawn": 4.0
    },
    "behavior_summary": "best-of-4 deterministic ODE (spawn=4.00, forward=8.00, nfe=8.00, single_ode_nfe=2)",
    "beta": 0.0,
    "nfe": 8.0,
    "reward": 0.6764771305322647,
    "reward_per_nfe": 0.08455964131653308
  },
  {
    "action_statistics": {
      "answer": 4.0,
      "forward": 24.0,
      "mean_nfe": 24.0,
      "spawn": 4.0
    },
    "behavior_summary": "best-of-4 deterministic ODE (spawn=4.00, forward=24.00, nfe=24.00, single_ode_nfe=6)",
    "beta": 0.25,
    "nfe": 24.0,
    "reward": 0.816731225013733,
    "reward_per_nfe": 0.03403046770890554
  },
  {
    "action_statistics": {
      "answer": 4.0,
      "forward": 44.0,
      "mean_nfe": 44.0,
      "spawn": 4.0
    },
    "behavior_summary": "best-of-4 deterministic ODE (spawn=4.00, forward=44.00, nfe=44.00, single_ode_nfe=11)",
    "beta": 0.5,
    "nfe": 44.0,
    "reward": 0.8422055500745773,
    "reward_per_nfe": 0.019141035228967665
  },
  {
    "action_statistics": {
      "answer": 4.0,
      "forward": 72.0,
      "mean_nfe": 72.0,
      "spawn": 4.0
    },
    "behavior_summary": "best-of-4 deterministic ODE (spawn=4.00, forward=72.00, nfe=72.00, single_ode_nfe=18)",
    "beta": 0.75,
    "nfe": 72.0,
    "reward": 0.8481007544994354,
    "reward_per_nfe": 0.011779177145825492
  },
  {
    "action_statistics": {
      "answer": 4.0,
      "forward": 100.0,
      "mean_nfe": 100.0,
      "spawn": 4.0
    },
    "behavior_summary": "best-of-4 deterministic ODE (spawn=4.00, forward=100.00, nfe=100.00, single_ode_nfe=25)",
    "beta": 1.0,
    "nfe": 100.0,
    "reward": 0.849547344326973,
    "reward_per_nfe": 0.00849547344326973
  }
]
```

## Recent Round Frontier Comparison

No prior rounds found.

## Recent History

No prior rounds found. Treat this as round 0.

