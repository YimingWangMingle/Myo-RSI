"""Audit an executed local matrix, including frozen tensors inside the checkpoints."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from myo_rsi.bank import load_bank
from myo_rsi.protocol import FINAL_SEEDS, TASKS, TRAIN_SEEDS, VALIDATION_SEEDS


def require(condition, message):
    if not condition:
        raise ValueError(message)


def tensor_equal(left, right):
    return left.keys() == right.keys() and all(
        torch.equal(left[k], right[k]) for k in left
    )


def actor(path):
    return torch.load(path, map_location="cpu", weights_only=True)["actor_state_dict"]


def audit(root):
    tables = root / "results/tables"
    inventory = pd.read_csv(tables / "checkpoint_manifest.csv")
    rows = pd.read_csv(tables / "eval_episodes.csv")
    manifest = json.loads((root / "results/preflight.json").read_text(encoding="utf8"))
    require(manifest["normal_gate"] and manifest["adaptation_gate"], "P0 incomplete")
    require(
        len(inventory) == 21 and inventory.run_id.nunique() == 21,
        "Expected 21 method/task/seed cells",
    )
    require(inventory.sha256.nunique() == 21, "Final checkpoint IDs are not distinct")
    expect = {
        f"{t}-{m}-{s}" for t in ("T1", "T2") for m in ("A", "B") for s in TRAIN_SEEDS
    }
    expect.update(f"T3-{m}-{s}" for m in ("C0", "C1", "C2") for s in TRAIN_SEEDS)
    require(set(inventory.run_id) == expect, "Missing method/task/seed")
    for r in inventory.itertuples():
        with (root / r.final_checkpoint).open("rb") as f:
            require(
                hashlib.file_digest(f, "sha256").hexdigest() == r.sha256,
                "Changed checkpoint",
            )
        budget = (
            1_572_864
            if r.task_id != "T3"
            else (0 if r.method == "C0 Frozen" else 393_216)
        )
        require(r.train_steps == budget, f"Unexpected budget for {r.run_id}")
    for (checkpoint, task, split), group in rows.groupby(
        ["checkpoint_id", "task_id", "split"]
    ):
        seeds = FINAL_SEEDS if split == "final" else VALIDATION_SEEDS
        require(
            sorted(group.eval_seed) == list(seeds),
            f"Seed mismatch: {checkpoint}/{task}",
        )
        require(group.success.isin([0, 1]).all(), "Nonbinary success")
    require(len(rows) == 16_800, "Missing checkpoint-level episodes")
    require(
        rows.groupby(["task_id", "split", "eval_seed"]).goal_hash.nunique().max() == 1,
        "Goal drift",
    )
    for task, group in rows.groupby("task_id"):
        validation = set(group[group.split == "validation"].goal_hash)
        final = set(group[group.split == "final"].goal_hash)
        require(
            validation.isdisjoint(final),
            f"Validation and final goals overlap on {task}",
        )
    gpu = next(
        r for r in manifest["rows"] if r["task_id"] == "T1" and r["backend"] == "mjlab"
    )
    _, bank = load_bank(root / "artifacts/synergy_bank_k8.npz", gpu["actuator_order"])
    require(
        set(inventory.bank_hash.dropna()) == {bank["bank_hash"]}, "Frozen bank mismatch"
    )
    sample_hashes = set()
    for source in bank["source_runs"]:
        fpath = root / "datasets/direct_rollouts/reach_random_train" / source["file"]
        with fpath.open("rb") as f:
            require(
                hashlib.file_digest(f, "sha256").hexdigest() == source["sha256"],
                "Changed NMF source",
            )
        with np.load(fpath, allow_pickle=False) as d:
            m = json.loads(d["metadata"].item())
            require(
                m["task_id"] == "T1"
                and m["env_id"] == TASKS["T1"]
                and m["split"] == "train",
                "Invalid bank origin",
            )
            for goal in d["goal"]:
                sample_hashes.add(hashlib.sha256(goal.tobytes()).hexdigest())
    tested = set(rows[rows.task_id == "T1"].goal_hash)
    require(sample_hashes.isdisjoint(tested), "Bank sample contains an evaluation goal")
    freeze = []
    by_run = inventory.set_index("run_id")
    dims = {
        r["task_id"]: r["obs_dim"] for r in manifest["rows"] if r["backend"] == "mjlab"
    }
    for task in ("T1", "T2"):
        for code, channels in (("A", 39), ("B", 8)):
            for seed in TRAIN_SEEDS:
                tag = f"{task}-{code}-{seed}"
                sd = actor(root / by_run.loc[tag, "final_checkpoint"])
                weights = [
                    tensor
                    for key, tensor in sd.items()
                    if key.startswith("mlp.") and key.endswith(".weight")
                ]
                require(
                    weights[0].shape[1] == dims[task], f"Wrong task actor input: {tag}"
                )
                require(
                    weights[-1].shape[0] == channels, f"Wrong action dimension: {tag}"
                )
    for seed in TRAIN_SEEDS:
        parent = f"T1-B-{seed}"
        b = actor(root / by_run.loc[parent, "final_checkpoint"])
        for code in ("C0", "C1", "C2"):
            key = f"T3-{code}-{seed}"
            path = root / by_run.loc[key, "final_checkpoint"]
            require(
                by_run.loc[key, "source_checkpoint_id"] == parent + "/policy_256.pt",
                "Wrong parent",
            )
            initial = actor(path.parent / "policy_0.pt")
            if code != "C2":
                require(tensor_equal(initial, b), f"{key} did not start from B-Reach")
            else:
                trained = actor(path)
                for k, tensor in b.items():
                    local_key = ("mlp.base." + k[4:]) if k.startswith("mlp.") else k
                    require(
                        torch.equal(initial[local_key], tensor),
                        f"Initial C2 base differs: {k}",
                    )
                    require(
                        torch.equal(trained[local_key], tensor),
                        f"Frozen C2 base changed: {k}",
                    )
                require(
                    torch.count_nonzero(initial["mlp.delta.2.weight"]).item() == 0,
                    "Nonzero residual initialization",
                )
                require(
                    torch.count_nonzero(initial["mlp.delta.2.bias"]).item() == 0,
                    "Nonzero residual bias",
                )
                require(
                    any(
                        not torch.equal(initial[k], trained[k])
                        for k in initial
                        if k.startswith("mlp.delta.")
                    ),
                    "Residual did not update",
                )
            freeze.append(key)
    diagnostic_steps = 0
    diagnostic_paths = sorted(
        (root / "datasets/diagnostic_rollouts").glob("*/manifest.json")
    )
    require(
        len(diagnostic_paths) == len(inventory),
        "Missing instrumented final policy episodes",
    )
    for path in diagnostic_paths:
        manifest_diag = json.loads(path.read_text(encoding="utf8"))
        with np.load(path.parent / "trajectory.npz", allow_pickle=False) as episode:
            n = manifest_diag["environment_steps"]
            require(
                episode["excitation"].shape == (n, 39)
                and episode["activation"].shape == (n, 39),
                "Incomplete muscle signals",
            )
            require(
                all(np.isfinite(episode[k]).all() for k in episode.files),
                "Nonfinite diagnostic trace",
            )
            native = episode["native_action"]
            expected_u = 1 / (1 + np.exp(-5 * (native.clip(-1, 1) - 0.5)))
            np.testing.assert_allclose(
                episode["excitation"], expected_u, atol=2e-6, rtol=0
            )
            require(
                bool(episode["terminated"][-1]) or bool(episode["truncated"][-1]),
                "Missing terminal state",
            )
            diagnostic_steps += n
    return dict(
        diagnostic_final_policy_episodes=len(diagnostic_paths),
        diagnostic_transitions=diagnostic_steps,
        completed_cells=len(inventory),
        training_seeds=list(TRAIN_SEEDS),
        evaluation_episode_records=len(rows),
        validation_final_goal_overlap=0,
        observed_bank_sample_evaluation_goal_overlap=0,
        verified_bank_hash=bank["bank_hash"],
        source_matched_adaptation_cells=freeze,
        verified_checkpoint_sha256=len(inventory),
        c2_base_tensors_including_normalizer_unchanged=True,
    )


if __name__ == "__main__":
    base = Path.cwd()
    result = audit(base)
    (base / "results/verification.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf8"
    )
    print(json.dumps(result, indent=2))
