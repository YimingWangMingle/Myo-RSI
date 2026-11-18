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

## Outputs

The compatibility evidence lives in `results/preflight.json` and `results/tables/env_manifest.csv`. `myo_rsi/protocol.py` fixes the task IDs and seed sets. The bank builder records its training runs, training-episode holdout, actuator order, normalization, reconstruction and projection errors. Its tests use temporary synthetic controls.

```powershell
.\.venv-gpu\Scripts\python -m myo_rsi.bank --help
```
