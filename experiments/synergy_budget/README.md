# Synergy representation and matched-budget PPO

Run from the repository root in the v0.1r GPU environment. The command first verifies the identity path, then evaluates the NMF capacity sweep, locates the onset of trajectory divergence, trains and evaluates the Reach matrix, and repeats the learned comparison on Pose using the same immutable Reach banks. It writes the scientific report and plots after the matrix completes.

```powershell
.\.venv-gpu\Scripts\python.exe -X utf8 -u -m experiments.synergy_budget.study
```

The study reads the original official checkpoints and data from the paths described in the v0.1r preflight. All new training checkpoints go to `checkpoints/synergy_budget`, new banks to `artifacts/synergy_budget` and all measured tables and plots to `results/synergy_budget`. The policy fixed-bank initialization uses NMF seed 11 for all training seeds; initialization robustness of the representation uses seeds 11, 29 and 47. [Protocol](protocol.json) · [v0.1r inputs](../../results/v0_1r/preflight.md)

To generate figures and tables from currently available data, use `python -m experiments.synergy_budget.report --interim`. For the complete report, use `python -m experiments.synergy_budget.report`. The latter checks all model hashes, paired goals, episode counts, budgets and evaluation artifacts before writing `RESULTS_REPORT.md`. Run one training matrix at a time. Training may resume from a saved optimizer checkpoint with a fresh seeded simulator; segment start and cost are recorded in the per-cell training history. A fresh repetition should use a separate checkout and empty output directories for this study.
