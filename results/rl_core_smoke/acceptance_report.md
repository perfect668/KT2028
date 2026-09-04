# MS-RLSurv minimal trusted Dense sequential PPO smoke

- Acceptance: 28/28 PASS
- Real Train transitions: 2,008 (seed=17, 8 patients, both labels)
- STOP-after-transition count: 0
- First-action warm-start accuracy: 1.0
- Critic fixed-batch overfit explained variance: 0.9969919324
- Reward telescoping error: 4.44e-16
- Checkpoint reload output delta: 0
- Fixed E2 checkpoints: 15 strict OOF fold/seed checkpoints
- Calibration: outputs/thesis_rl_v7/r1_online_ppo/metrics/e2_calibration_parameters.csv
- No Validation/Test read; no MoE, beam, DROP, REFINE or formal PPO run.

Records: `KT/synced_reports/thesis_rl_v8/rl_core_smoke/`.
