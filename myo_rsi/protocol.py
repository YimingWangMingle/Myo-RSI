"""Locked V0.1 task and seed assignments. Evaluation seeds are never bank sources."""

TASKS = {
    "T0": "myoHandReachFixed-v0",
    "T1": "myoHandReachRandom-v0",
    "T2": "myoHandPoseRandom-v0",
    "T3": "myoSarcHandReachRandom-v0",
}
TRAIN_SEEDS = (11, 29, 47)
PROBE_SEED = 701
VALIDATION_SEEDS = tuple(range(20_000, 20_100))
FINAL_SEEDS = tuple(range(40_000, 40_100))
K = 8
UPSTREAM = "d914c4aa1eea568dd45bcc3d760a83dc9362c093"
