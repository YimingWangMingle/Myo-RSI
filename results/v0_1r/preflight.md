# v0.1r preflight and reproduction

The eight local CPU/GPU compatibility probes passed for fixed Reach, random Reach, random Pose and sarcopenic Reach. Nine legacy tests passed, and a read-only invocation of the historical auditor verified the 21 v0.1 checkpoint hashes, the old bank, evaluation splits and frozen base tensors. [Environment details](preflight.json) · [Environment table](tables/env_manifest.csv) · [Historical audit](historical_audit.json)

The tag `v3.1.0` in the upstream Hugging Face model repository resolved to `d6ed512bd329c018ecc373a351d00916cd552d34`. Both evaluated checkpoints passed the Git manifest blob check and the LFS SHA-256 check. The installed official policy loader reconstructed each deterministic actor and its empirical observation normalizer. Reach used 115 input features and Pose used 108. Both produced 39 muscle commands. Native final-step success agreed with the distance threshold for every measured teacher episode. [Teacher provenance](teacher_manifest.json) · [Teacher episodes](teacher_eval_episodes.csv)

The hardware was a GeForce RTX 4060 with 8,188 MiB and driver 621.01. The runtime record includes the library versions and PyTorch defaults. The experiments use deterministic actor means and seeded resets. The installed simulator uses single-precision GPU physics. [Runtime record](runtime.json) · [Research decisions and seed ranges](run_lock.json)

## Reproducing the measured stages

Use the repository's Windows/CPython 3.13 GPU environment and run commands from the repository root. The analysis used scikit-learn 1.9.1, SciPy 1.18.1, threadpoolctl 3.8.0 and matplotlib 3.11.2. For inspection of the published data use `git lfs pull` and run the audit against the included NPZs. The audit reads all frames and refits a candidate.

```powershell
python -X utf8 -m unittest discover -s tests
python -X utf8 -c "from pathlib import Path; from scripts.audit_v01 import audit; print(audit(Path.cwd()))"
python -X utf8 -m myo_rsi.probe --backend both --results-dir results/v0_1r
python -X utf8 scripts/fetch_v01r_teachers.py myoHandReachRandom-v0 myoHandPoseRandom-v0
python -X utf8 -m myo_rsi.rerun_v01r --stage teacher
python -X utf8 -m myo_rsi.expert_data
python -X utf8 -m myo_rsi.bank_study
python -X utf8 -m myo_rsi.projection_sensitivity
python -X utf8 -m myo_rsi.matched_response
python -X utf8 -m scripts.audit_v01r
python -X utf8 -m scripts.report_v01r
```

The historical audit also requires the original v0.1 local checkpoint archive. The v0.1r audit requires the two v0.1r dataset NPZs, saved candidate banks and measured episode tables. For a fresh run use a separate checkout with a new output directory at `results/v0_1r` containing only a copy of the published `run_lock.json`. Use fresh `artifacts/v0_1r` and `datasets/v0_1r` directories. Completed per-task teacher and per-candidate bank records are resumable, so reusing the published results directory would reuse measurements.

The Pose weight transfer was interrupted. An incomplete local file failed the length check before inference. We completed the transfer from the original immutable URL, verified the LFS digest, and resumed Pose. The already completed Reach cohort remained intact.

## How to read the records

- `teacher_eval_episodes.csv` contains the 200 native teacher validation episodes. `bank_episodes.csv` contains the unmodified teacher and the 12 primary projections on the bank-development cohort. `solver_sensitivity_episodes.csv` holds the 12 independent-solver comparisons.
- Each `rollouts_*.npz` groups frames by seed and timestep. `seed` identifies the episode. `obs` is the observation at action selection. `ctrl`, `act`, joint state and errors are post-step. `weighted_reward_terms` follows the order in the dataset manifest. Constant source and body identifiers are stored once per partition in the linked manifest.
- Each candidate directory contains `bank.npz` with the 39 × K matrix, muscle order and 95th-percentile fit coefficient scales. The adjacent JSON includes reconstruction, coverage by outcome, per-muscle errors, histograms and hashes.
- The primary projection uses the physically clipped `Wc`, nonnegative coefficients and zero L1 penalty. It starts with 48 accelerated iterations on the linear NNLS objective and permits 240 guarded projected-gradient steps. The independent sensitivity uses double-precision NNLS followed by L-BFGS-B, a 250-iteration ceiling and a projected fixed-point tolerance of 2 × 10⁻⁵. Solver timings appear separately in the functional tables.
- Blank success and episode cells in the downstream tables denote a stage that did not run. All available figures have both PNG and SVG files. Figures requiring a selected bank, trained PPO policies or a final cohort are absent after the P2 stop.
