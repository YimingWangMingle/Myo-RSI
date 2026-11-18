# Myo-RSI v0.1

This experiment studies a fixed, eight-component muscle-excitation bank in MyoSuite. The primary comparison trains direct and synergy PPO separately on random fingertip reach and random joint pose. Both synergy policies use the same bank fitted from direct Reach training rollouts. A third comparison tests frozen, fully fine-tuned and recruitment-residual policies on the sarcopenic Reach hand. V0.1 covers System 1.

## Local environment

The Windows setup uses CPython 3.13 and an NVIDIA GPU. These PowerShell commands create the project environment. The lock targets CUDA 13.0 wheels.

```powershell
python -m venv .venv-gpu
.\.venv-gpu\Scripts\python -m pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.14.1+cu130 torchvision==0.29.1+cu130
.\.venv-gpu\Scripts\python -m pip install -r requirements-win-cu130.txt
.\.venv-gpu\Scripts\python -X utf8 -m unittest discover -s tests -v
.\.venv-gpu\Scripts\python -X utf8 -m myo_rsi.probe --backend both --results-dir results
```

The probe checks both backends independently. It records the compiled model and mechanical hand signatures, ordered actuator names, observations, task-native success metric, horizon, peak forces, physical excitation and activation, and two seeded traces. An incomplete full check returns a nonzero exit code. `--backend cpu` can be used for CPU diagnostics, with a separate output directory.

Reach success is the final-step L2 error over the five stacked fingertip positions below 0.0625 m. Pose uses the installed task's joint-space error threshold. The hand uses an excitation of `sigmoid(5 * (action - 0.5))` after clipping each native action to [-1, 1]. Inverting a desired excitation projects it onto the range attained by those native endpoints and records the projection error.

The experiment uses MyoSuite [v3.1.0](https://github.com/MyoHub/myosuite/tree/d914c4aa1eea568dd45bcc3d760a83dc9362c093). The release includes the source for the [training CLI](https://github.com/MyoHub/myosuite/blob/d914c4aa1eea568dd45bcc3d760a83dc9362c093/scripts/train_mjlab.py) and [evaluation CLI](https://github.com/MyoHub/myosuite/blob/d914c4aa1eea568dd45bcc3d760a83dc9362c093/scripts/eval_mjlab_policy.py).

## Training and evaluation

After both checks pass, run the matrix serially on the local GPU. It skips finished runs, evaluates the saved checkpoints, and stops if the direct or synergy Reach policy fails to improve its three-seed mean validation success over initialization. A bank fit requires all three direct Reach runs. A failed, partial training directory is preserved for inspection.

```powershell
$env:MUJOCO_GL = 'glfw'
.\.venv-gpu\Scripts\python -X utf8 -u -m myo_rsi.run --stage all
.\.venv-gpu\Scripts\python -X utf8 scripts/make_figures.py --results-dir results
```

Individual stages accept `--stage A`, `--stage B` or `--stage C`. Individual cells use `python -m myo_rsi.train T1 A 11` and `python -m myo_rsi.evaluate T1 A 11`. A/B train for 256 × 24 × 256 transitions per seed. C1/C2 receive 256 × 24 × 64 additional transitions. All use the upstream actor/critic and PPO hyperparameters. Checkpoints are saved at initialization and every quarter budget. The validation curve uses 100 episodes at each checkpoint. The final table uses a separate list of 100 seeds. Each lane is individually reset to its assigned seed before running the deterministic mean-action policy.

The bank builder samples measured excitations in eight lanes every eight steps during the final quarter of each direct Reach training run. Each sample carries a run, unique lane/episode ID, timestep, raw action, applied control, activation, post-step observation, joint position and goal. Terminal autorestarts are excluded. An episode-level split separates NMF fitting and reconstruction validation. Each column of W has unit maximum. The decoder uses 0.15 × softplus(raw latent) as the nonnegative recruitment, projects Wc into the physical excitation interval, and sends its inverse through the native action term. PPO uses the Gaussian likelihood of the original 8D sample. C2 adds a zero-initialized-output residual to the raw mean, freezes the original actor, its observation statistics and action variance, and trains the critic and residual.

Checkpoints and training datasets are stored locally. The plotting command can also regenerate published figures from the CSVs and bank artifact in a standalone checkout. The resource ledger counts the bank source trajectories during A-Reach training. The unit tests generate synthetic arrays in temporary directories. The results report only consumes run logs.

## Outputs

The compatibility evidence lives in `results/preflight.json` and `results/tables/env_manifest.csv`. `myo_rsi/protocol.py` fixes the task IDs and seed sets. The bank builder records its training runs, training-episode holdout, actuator order, normalization, reconstruction and projection errors. Its tests use temporary synthetic controls.

```powershell
.\.venv-gpu\Scripts\python -m myo_rsi.bank --help
```
