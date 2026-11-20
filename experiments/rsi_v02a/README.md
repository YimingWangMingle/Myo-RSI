# V0.2a diagnosis and recruitment repair

Three K=16 Reach policies from the [matched-budget study](../../results/synergy_budget/RESULTS_REPORT.md) are the parents of every primary arm. The source is commit `6456b680f379c197c80cd80845b330414a99e104`. The new study follows the development guidelines' directory spelling `rsi_v02a`.

| Question | Registered comparison |
|---|---|
| Can body weakness cause a recoverable loss? | Same 100 targets on normal Reach, native sarcopenic Reach and conditional S2. Full-muscle witnesses use the same body. |
| Can the frozen 16D bank correct individual failures? | Reset, replay and verify the physical state. Compare two equal-sized pools of 10-step corrections, then continue to native episode termination. |
| Does diagnosis improve adaptation at matched cost? | Frozen B0, full PPO B1, residual PPO B2 and conditional search–BC–residual PPO B3. Hold out 200 shared targets for both weak-body performance and normal-body retention. |
| Can the bank be edited safely? | Reload, gradient, action and old-skill tests on a 16D reshaped bank and a 17D expanded bank. |

## Running the study

Use a fresh worktree of the feature branch and the pinned Windows GPU environment described in the [upstream setup](../../results/v0_1r/preflight.md). From the repository root run `git lfs pull`. Restore the source checkpoints and banks before training. The [source manifest](../../results/rsi_v02a/source_manifest.json) names all source policies and hashes.

```powershell
python -X utf8 -m unittest discover -s tests
python -X utf8 -m experiments.rsi_v02a.screen
# Run the registered contingency when native T3 has insufficient degradation.
python -X utf8 -m experiments.rsi_v02a.weakness
python -X utf8 -m experiments.rsi_v02a.diagnosis
python -X utf8 -m experiments.rsi_v02a.adapt train
python -X utf8 -m experiments.rsi_v02a.structure
python -X utf8 -m experiments.rsi_v02a.adapt validation
python -X utf8 -m experiments.rsi_v02a.adapt final
python -X utf8 -m experiments.rsi_v02a.publish
```

The single entry point `python -m experiments.rsi_v02a.run` dispatches the same gated sequence. The release includes all completed and failed attempts. Repeating a stage in its published folder reuses completed attempts. To run a fresh repetition, start from an isolated checkout of the same code with a new empty `results/rsi_v02a` and `checkpoints/rsi_v02a` output location. The input directories in the previous studies must stay accessible. A new attempt receives the next available ID within the output root. A failed PPO process restarts as a new attempt with a fresh simulator and optimizer. The archived attempt remains in the cost ledger.

The weak-body ID T3 in the run names identifies the task family. Each screen and measurement explicitly records the environment or compiled body. The active adaptation site in this release is S2. It uses the native weak configuration with a further force multiplier of 0.75 applied before compiling. All three seeds have the same 39 ordered actuators and a distinct body from normal Reach.

## Evidence and storage

The [result report](../../results/rsi_v02a/RESULTS_REPORT.md) is derived from per-target records. The [registry](../../results/rsi_v02a/runs.csv) includes failures. The [replay audit](../../results/rsi_v02a/replay_audit.csv) distinguishes candidate-state matches from matches that also pass the registered control. Search candidates from the same goal are correlated. The phase that assembles the locked final set checks source hashes, target identifiers and budgets. B3 requires at least two distinct verified cases so BC can be tested on a held-out goal. Unavailable arms have a separate [gate record](../../results/rsi_v02a/blocked_arms.csv).

The historical repository already supplies pinned programs, checkpoints, bank checksums, per-target results, and a Git LFS store. This study adds an indexed run registry, immutable attempts, coupled physical-interaction accounting and cross-body retention. Git LFS stores selected adaptation model/optimizer checkpoints and the three complete diagnostic trajectory archives. Small derived files, figures, protocols and metadata are ordinary Git. Each checksum in the inventory refers to restored bytes. The historical studies are unchanged.

## Counterfactual search validation

Three frozen K16 checkpoints were evaluated on independent S2 development goals. The [counterfactual report](../../results/rsi_v02a/counterfactual_validation/RESULTS_REPORT.md) counts unique physical targets across two registered designs. The [first protocol](counterfactual_validation/protocol.json) checks full MuJoCo state restoration, same-lane action-prefix replay and native no-op continuations. The [calibration replication](counterfactual_validation/protocol_paired.json) gives the 39D controller the measured K16 excitation amplitude and number of saturated muscles at each intervention tick. The original V0.2a diagnostic remains reproducible from its published commit.

From the repository root, using the pinned GPU Python environment with Git LFS inputs restored:

```powershell
foreach ($policySeed in 11,29,47) {
    python -m experiments.rsi_v02a.counterfactual_validation.evaluate --seed $policySeed
    python -m experiments.rsi_v02a.counterfactual_validation.evaluate --seed $policySeed --paired
}
python -m experiments.rsi_v02a.counterfactual_validation.report
python -m experiments.rsi_v02a.counterfactual_validation.report --paired
python -m experiments.rsi_v02a.counterfactual_validation.publish
```

Each simulator execution creates a fresh run ID. When a checkout already has multiple completed runs per design and seed, pass three explicit `--run RUN_ID` arguments to each report command to select a release. The [run registry](../../results/rsi_v02a/counterfactual_validation/runs.csv) identifies the six included runs and excluded engineering attempts. The [BC dataset index](../../results/rsi_v02a/counterfactual_validation/bc_dataset.csv) defines the shared physical-goal split across both designs.
