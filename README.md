# Myo-RSI

Myo-RSI v0.1 compares direct muscle control with a fixed NMF synergy bank on MyoSuite random hand reach and pose, then tests how the Reach policy adapts to reduced muscle strength. The experiment uses mjlab on CUDA and RSL-RL PPO.

The three-seed final success rates on random Reach were 3.0% for direct PPO and 5.3% for synergy PPO. Both methods recorded zero final successes on random Pose. On sarcopenic Reach, frozen, full and local adaptation reached 4.7%, 5.3% and 2.3% respectively. Each final policy was evaluated on 100 held-out episodes per task and seed.

- [Environment setup, commands and evaluation protocol](README_MYO_RSI_V0_1.md)
- [Measured results and figures](results/RESULTS_REPORT.md)
- [Runtime compatibility and actuator checks](results/preflight.json)
- [Numerical and experimental limitations](results/limitations.md)
- [Checkpoint, frozen-weight and evaluation audit](results/verification.json)
