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

| round | beta | mean_nfe | reward | nearest_baseline_nfe | baseline_reward | delta | actions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| r0001 | 0.000 | 10.000 | 0.819160 | 8.000 | 0.676477 | 0.142682 | deterministic forward (spawn=1.00, preview=0.00, nfe=10.00) |
| r0001 | 0.250 | 26.100 | 0.834235 | 24.000 | 0.816731 | 0.017504 | preview-guided backward refinement (spawn=1.00, preview=7.01, backward=1.47, prune=0.46, nfe=26.10) |
| r0001 | 0.500 | 45.672 | 0.839438 | 44.000 | 0.842206 | -0.002768 | preview-guided backward refinement (spawn=1.00, preview=11.50, backward=2.83, prune=0.33, nfe=45.67) |
| r0001 | 0.750 | 64.828 | 0.836009 | 72.000 | 0.848101 | -0.012092 | preview-guided backward refinement (spawn=1.00, preview=16.74, backward=3.81, prune=0.20, nfe=64.83) |
| r0001 | 1.000 | 91.922 | 0.837993 | 100.000 | 0.849547 | -0.011554 | preview-guided backward refinement (spawn=1.00, preview=25.27, backward=4.20, prune=0.13, nfe=91.92) |
| r0000 | 0.000 | 10.000 | 0.819160 | 8.000 | 0.676477 | 0.142682 | deterministic forward (spawn=1.00, preview=0.00, nfe=10.00) |
| r0000 | 0.250 | 26.000 | 0.830616 | 24.000 | 0.816731 | 0.013884 | single-root preview (spawn=1.00, preview=6.00, nfe=26.00) |
| r0000 | 0.500 | 44.410 | 0.836373 | 44.000 | 0.842206 | -0.005832 | preview-guided backward refinement (spawn=1.00, preview=10.42, backward=1.71, prune=0.29, nfe=44.41) |
| r0000 | 0.750 | 56.566 | 0.840041 | 44.000 | 0.842206 | -0.002164 | preview-guided backward refinement (spawn=1.00, preview=13.43, backward=2.77, prune=1.06, nfe=56.57) |
| r0000 | 1.000 | 84.316 | 0.843048 | 72.000 | 0.848101 | -0.005053 | preview-guided backward refinement (spawn=1.00, preview=18.65, backward=3.90, prune=1.02, nfe=84.32) |

## Recent History

### `logs/flow_autotts/pickscore_sd35/train_codex_bestof4_ode_clean_b128/history/r0001_20260521_092626_8464683a`

- controller snapshot: `logs/flow_autotts/pickscore_sd35/train_codex_bestof4_ode_clean_b128/history/r0001_20260521_092626_8464683a/flow_autotts/controllers/optimal.py`
- compact summary: `logs/flow_autotts/pickscore_sd35/train_codex_bestof4_ode_clean_b128/history/r0001_20260521_092626_8464683a/proposal_results/summary.json`

```json
{
  "betas": [
    0.0,
    0.25,
    0.5,
    0.75,
    1.0
  ],
  "budget": 128,
  "evaluated_sample_size": 500,
  "experiment": "pickscore_sd35",
  "num_shards": 4,
  "rounds": [
    {
      "beta_sweep": [
        {
          "action_statistics": {
            "answer": 1.0,
            "forward": 10.0,
            "mean_nfe": 10.0,
            "spawn": 1.0
          },
          "behavior_summary": "deterministic forward (spawn=1.00, preview=0.00, nfe=10.00)",
          "beta": 0.0,
          "nfe": 10,
          "reward": 0.8191596091985702,
          "reward_per_nfe": 0.08191596091985702
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "backward": 1.47,
            "forward": 19.092,
            "mean_nfe": 26.1,
            "preview": 7.008,
            "prune": 0.462,
            "spawn": 1.0
          },
          "behavior_summary": "preview-guided backward refinement (spawn=1.00, preview=7.01, backward=1.47, prune=0.46, nfe=26.10)",
          "beta": 0.25,
          "nfe": 26.1,
          "reward": 0.8342354561090469,
          "reward_per_nfe": 0.03247220769090685
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "backward": 2.828,
            "forward": 34.172,
            "mean_nfe": 45.672,
            "preview": 11.5,
            "prune": 0.328,
            "spawn": 1.0
          },
          "behavior_summary": "preview-guided backward refinement (spawn=1.00, preview=11.50, backward=2.83, prune=0.33, nfe=45.67)",
          "beta": 0.5,
          "nfe": 45.672,
          "reward": 0.8394380246400833,
          "reward_per_nfe": 0.018440989523264957
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "backward": 3.806,
            "forward": 48.084,
            "mean_nfe": 64.828,
            "preview": 16.744,
            "prune": 0.202,
            "spawn": 1.0
          },
          "behavior_summary": "preview-guided backward refinement (spawn=1.00, preview=16.74, backward=3.81, prune=0.20, nfe=64.83)",
          "beta": 0.75,
          "nfe": 64.828,
          "reward": 0.8360091205835343,
          "reward_per_nfe": 0.012914798900774672
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "backward": 4.204,
            "forward": 66.648,
            "mean_nfe": 91.922,
            "preview": 25.274,
            "prune": 0.13,
            "spawn": 1.0
          },
          "behavior_summary": "preview-guided backward refinement (spawn=1.00, preview=25.27, backward=4.20, prune=0.13, nfe=91.92)",
          "beta": 1.0,
          "nfe": 91.922,
          "reward": 0.8379931790828705,
          "reward_per_nfe": 0.009128856082094844
        }
      ],
      "controller": "optimal",
      "controller_name": "OptimalController",
      "pareto_frontier": [
        {
          "action_statistics": {
            "answer": 1.0,
            "forward": 10.0,
            "mean_nfe": 10.0,
            "spawn": 1.0
          },
          "behavior_summary": "deterministic forward (spawn=1.00, preview=0.00, nfe=10.00)",
          "beta": 0.0,
          "nfe": 10,
          "reward": 0.8191596091985702,
          "reward_per_nfe": 0.08191596091985702
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "backward": 1.47,
            "forward": 19.092,
            "mean_nfe": 26.1,
            "preview": 7.008,
            "prune": 0.462,
            "spawn": 1.0
          },
          "behavior_summary": "preview-guided backward refinement (spawn=1.00, preview=7.01, backward=1.47, prune=0.46, nfe=26.10)",
          "beta": 0.25,
          "nfe": 26.1,
          "reward": 0.8342354561090469,
          "reward_per_nfe": 0.03247220769090685
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "backward": 2.828,
            "forward": 34.172,
            "mean_nfe": 45.672,
            "preview": 11.5,
            "prune": 0.328,
            "spawn": 1.0
          },
          "behavior_summary": "preview-guided backward refinement (spawn=1.00, preview=11.50, backward=2.83, prune=0.33, nfe=45.67)",
          "beta": 0.5,
          "nfe": 45.672,
          "reward": 0.8394380246400833,
          "reward_per_nfe": 0.018440989523264957
        }
      ],
      "round_id": 0
    }
  ],
  "sample_seed": 42,
  "sample_size": 500,
  "shard_index": null
}
```

### `logs/flow_autotts/pickscore_sd35/train_codex_bestof4_ode_clean_b128/history/r0000_20260521_092626_8464683a`

- controller snapshot: `logs/flow_autotts/pickscore_sd35/train_codex_bestof4_ode_clean_b128/history/r0000_20260521_092626_8464683a/flow_autotts/controllers/optimal.py`
- compact summary: `logs/flow_autotts/pickscore_sd35/train_codex_bestof4_ode_clean_b128/history/r0000_20260521_092626_8464683a/proposal_results/summary.json`

```json
{
  "betas": [
    0.0,
    0.25,
    0.5,
    0.75,
    1.0
  ],
  "budget": 128,
  "evaluated_sample_size": 500,
  "experiment": "pickscore_sd35",
  "num_shards": 4,
  "rounds": [
    {
      "beta_sweep": [
        {
          "action_statistics": {
            "answer": 1.0,
            "forward": 10.0,
            "mean_nfe": 10.0,
            "spawn": 1.0
          },
          "behavior_summary": "deterministic forward (spawn=1.00, preview=0.00, nfe=10.00)",
          "beta": 0.0,
          "nfe": 10,
          "reward": 0.8191596091985702,
          "reward_per_nfe": 0.08191596091985702
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "forward": 20.0,
            "mean_nfe": 26.0,
            "preview": 6.0,
            "spawn": 1.0
          },
          "behavior_summary": "single-root preview (spawn=1.00, preview=6.00, nfe=26.00)",
          "beta": 0.25,
          "nfe": 26,
          "reward": 0.8306155842542648,
          "reward_per_nfe": 0.031946753240548645
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "backward": 1.71,
            "forward": 33.986,
            "mean_nfe": 44.41,
            "preview": 10.424,
            "prune": 0.286,
            "spawn": 1.0
          },
          "behavior_summary": "preview-guided backward refinement (spawn=1.00, preview=10.42, backward=1.71, prune=0.29, nfe=44.41)",
          "beta": 0.5,
          "nfe": 44.41,
          "reward": 0.8363734068870544,
          "reward_per_nfe": 0.018985413484423865
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "backward": 2.77,
            "forward": 43.136,
            "mean_nfe": 56.566,
            "preview": 13.43,
            "prune": 1.06,
            "spawn": 1.0
          },
          "behavior_summary": "preview-guided backward refinement (spawn=1.00, preview=13.43, backward=2.77, prune=1.06, nfe=56.57)",
          "beta": 0.75,
          "nfe": 56.566,
          "reward": 0.840041455745697,
          "reward_per_nfe": 0.014946287974899788
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "backward": 3.898,
            "forward": 65.668,
            "mean_nfe": 84.316,
            "preview": 18.648,
            "prune": 1.024,
            "spawn": 1.0
          },
          "behavior_summary": "preview-guided backward refinement (spawn=1.00, preview=18.65, backward=3.90, prune=1.02, nfe=84.32)",
          "beta": 1.0,
          "nfe": 84.316,
          "reward": 0.8430481963157653,
          "reward_per_nfe": 0.010032911791097427
        }
      ],
      "controller": "optimal",
      "controller_name": "OptimalController",
      "pareto_frontier": [
        {
          "action_statistics": {
            "answer": 1.0,
            "forward": 10.0,
            "mean_nfe": 10.0,
            "spawn": 1.0
          },
          "behavior_summary": "deterministic forward (spawn=1.00, preview=0.00, nfe=10.00)",
          "beta": 0.0,
          "nfe": 10,
          "reward": 0.8191596091985702,
          "reward_per_nfe": 0.08191596091985702
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "forward": 20.0,
            "mean_nfe": 26.0,
            "preview": 6.0,
            "spawn": 1.0
          },
          "behavior_summary": "single-root preview (spawn=1.00, preview=6.00, nfe=26.00)",
          "beta": 0.25,
          "nfe": 26,
          "reward": 0.8306155842542648,
          "reward_per_nfe": 0.031946753240548645
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "backward": 1.71,
            "forward": 33.986,
            "mean_nfe": 44.41,
            "preview": 10.424,
            "prune": 0.286,
            "spawn": 1.0
          },
          "behavior_summary": "preview-guided backward refinement (spawn=1.00, preview=10.42, backward=1.71, prune=0.29, nfe=44.41)",
          "beta": 0.5,
          "nfe": 44.41,
          "reward": 0.8363734068870544,
          "reward_per_nfe": 0.018985413484423865
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "backward": 2.77,
            "forward": 43.136,
            "mean_nfe": 56.566,
            "preview": 13.43,
            "prune": 1.06,
            "spawn": 1.0
          },
          "behavior_summary": "preview-guided backward refinement (spawn=1.00, preview=13.43, backward=2.77, prune=1.06, nfe=56.57)",
          "beta": 0.75,
          "nfe": 56.566,
          "reward": 0.840041455745697,
          "reward_per_nfe": 0.014946287974899788
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "backward": 3.898,
            "forward": 65.668,
            "mean_nfe": 84.316,
            "preview": 18.648,
            "prune": 1.024,
            "spawn": 1.0
          },
          "behavior_summary": "preview-guided backward refinement (spawn=1.00, preview=18.65, backward=3.90, prune=1.02, nfe=84.32)",
          "beta": 1.0,
          "nfe": 84.316,
          "reward": 0.8430481963157653,
          "reward_per_nfe": 0.010032911791097427
        }
      ],
      "round_id": 0
    }
  ],
  "sample_seed": 42,
  "sample_size": 500,
  "shard_index": null
}
```

