# Myo-RSI

[All studies and V0.2a evaluation](results/INDEX.md)

**Matched-budget PPO study.** On Reach, direct PPO achieved 87.3% final success and K=16 synergy PPO achieved 84.0% after 18M interactions per seed. Projecting the official teacher into the same K=16 bank achieved 35/100 on the identical targets. All three trained action spaces had zero final Pose successes. [Results and figures](results/synergy_budget/RESULTS_REPORT.md) · [Code and protocol](experiments/synergy_budget/README.md) · [Verified PPO checkpoints (Git LFS)](checkpoints/synergy_budget)

**v0.1r rerun.** The official Reach teacher passed local validation (89/100). Across 12 NMF banks the best projected teacher achieved 30/100 against 95/100 on matched targets. The experiment stopped at the prespecified functional gate. [Measured report and figures](results/v0_1r/RESULTS_REPORT.md) · [Run protocol](results/v0_1r/preflight.md) · [Audit](results/v0_1r/verification.json)

Myo-RSI v0.1 compares direct muscle control with a fixed NMF synergy bank on MyoSuite random hand reach and pose, then tests how the Reach policy adapts to reduced muscle strength. The experiment uses mjlab on CUDA and RSL-RL PPO.

The three-seed final success rates on random Reach were 3.0% for direct PPO and 5.3% for synergy PPO. Both methods recorded zero final successes on random Pose. On sarcopenic Reach, frozen, full and local adaptation reached 4.7%, 5.3% and 2.3% respectively. Each final policy was evaluated on 100 held-out episodes per task and seed.

- [Environment setup, commands and evaluation protocol](README_MYO_RSI_V0_1.md)
- [Measured results and figures](results/RESULTS_REPORT.md)
- [Runtime compatibility and actuator checks](results/preflight.json)
- [Numerical and experimental limitations](results/limitations.md)
- [Checkpoint, frozen-weight and evaluation audit](results/verification.json)
