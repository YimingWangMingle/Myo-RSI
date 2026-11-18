"""Store complete official Reach trajectories, split by episode, and audit physical commands."""

import csv
import time
from pathlib import Path

import numpy as np

from .rerun_v01r import OUT, cpu, load_teacher, read, rollout, save, sha, table

DATA = Path("datasets/v0_1r")


def collect():
    start = time.perf_counter()
    lock = read(OUT / "run_lock.json")
    with (OUT / "teacher_eval_summary.csv").open() as stream:
        summary = {r["task"]: r for r in csv.DictReader(stream)}
    if summary["T1"]["gate"] != "PASS" or "T2" not in summary:
        raise ValueError("Official teacher preflight incomplete")
    if (OUT / "dataset_manifest.json").exists():
        recorded = read(OUT / "dataset_manifest.json")
        teacher = next(
            r for r in read(OUT / "teacher_manifest.json") if r["task"] == "T1"
        )
        if (
            recorded["gate"] != "PASS"
            or teacher["model_sha256"] != recorded["teacher_sha256"]
        ):
            raise ValueError("Dataset source or gate mismatch")
        for split in recorded["splits"]:
            if (
                sha((DATA / ("rollouts_" + split["split"] + ".npz")).read_bytes())
                != split["sha256"]
            ):
                raise ValueError("Existing dataset checksum mismatch")
        return
    DATA.mkdir(parents=True, exist_ok=True)
    seeds = np.arange(lock["dataset"][0], lock["dataset"][1] + 1)
    if len(seeds) % 100:
        raise ValueError("Expected complete fixed-width batches")
    env, policy, source = load_teacher("T1", 100)
    recorded = next(r for r in read(OUT / "teacher_manifest.json") if r["task"] == "T1")
    if source["model_sha256"] != recorded["model_sha256"]:
        raise ValueError("Teacher differs from P0")
    splits = {"train": {}, "val": {}}
    episodes = []
    sim_steps = 0
    goals_seen = set()
    with (OUT / "teacher_eval_episodes.csv").open() as f:
        eval_goals = {r["goal_sha256"] for r in csv.DictReader(f) if r["task"] == "T1"}
    try:
        for i in range(0, len(seeds), 100):
            batch = seeds[i : i + 100]

            def capture(t, active, before, raw, env, reward_terms, errors, goals):
                frame = dict(
                    seed=batch,
                    timestep=np.full(len(batch), t, dtype=np.int16),
                    goal=goals,
                    obs=cpu(before),
                    raw_action=cpu(raw),
                    native_action=cpu(raw.clamp(-1, 1)),
                    ctrl=cpu(env.sim.data.ctrl),
                    act=cpu(env.sim.data.act),
                    qpos=cpu(env.sim.data.qpos),
                    qvel=cpu(env.sim.data.qvel),
                    weighted_reward_terms=cpu(reward_terms),
                    task_error=cpu(errors),
                )
                for name, data in splits.items():
                    take = active & (
                        (batch % 5 == 0) if name == "val" else (batch % 5 != 0)
                    )
                    for key, values in frame.items():
                        data.setdefault(key, []).append(values[take])

            rows, cost = rollout(env, policy, "T1", batch, recorder=capture)
            sim_steps += cost
            for row in rows:
                goal = row["goal_sha256"]
                if goal in goals_seen or goal in eval_goals:
                    raise ValueError(
                        "Duplicate target instance or development goal leakage"
                    )
                goals_seen.add(goal)
                row["split"] = "val" if row["seed"] % 5 == 0 else "train"
                row["outcome"] = (
                    "success"
                    if row["success"]
                    else (
                        "near_success"
                        if row["terminal_error"] < lock["near_success_reach_m"]
                        else "other_failure"
                    )
                )
            episodes.extend(rows)
        terms = list(env.reward_manager.active_terms)
    finally:
        env.close()
    table(OUT / "dataset_episodes.csv", episodes)
    indexed = {r["seed"]: r for r in episodes}
    manifest = []
    for split, parts in splits.items():
        arrays = {k: np.concatenate(v) for k, v in parts.items()}
        idx = np.lexsort((arrays["timestep"], arrays["seed"]))
        arrays = {k: v[idx] for k, v in arrays.items()}
        ids, counts = np.unique(arrays["seed"], return_counts=True)
        for seed, count in zip(ids, counts):
            mask = arrays["seed"] == seed
            if count != indexed[seed]["length"] or not np.array_equal(
                arrays["timestep"][mask], np.arange(1, count + 1)
            ):
                raise ValueError("Incomplete episode")
        arrays["outcome"] = np.array(
            [indexed[s]["outcome"] for s in arrays["seed"]], dtype="U13"
        )
        expected = 1 / (1 + np.exp(-5 * (arrays["native_action"] - 0.5)))
        max_error = float(np.max(np.abs(expected - arrays["ctrl"])))
        if max_error > 2e-6 or not all(
            np.isfinite(v).all() for v in arrays.values() if v.dtype.kind != "U"
        ):
            raise ValueError("Physical frame integrity failure")
        if arrays["act"].min() < -1e-5 or arrays["act"].max() > 1.00001:
            raise ValueError("Activation outside physical range")
        file = DATA / f"rollouts_{split}.npz"
        temporary = file.with_suffix(".part.npz")
        np.savez_compressed(temporary, **arrays)
        temporary.replace(file)
        with np.load(file, allow_pickle=False) as verify:
            if not np.array_equal(verify["ctrl"], arrays["ctrl"]):
                raise ValueError("Saved excitation array mismatch")
        outcome = {
            c: sum(indexed[s]["outcome"] == c for s in ids)
            for c in ("success", "near_success", "other_failure")
        }
        manifest.append(
            dict(
                split=split,
                episodes=len(ids),
                goals=len(ids),
                frames=len(expected),
                **outcome,
                ctrl_min=float(arrays["ctrl"].min()),
                ctrl_max=float(arrays["ctrl"].max()),
                mapping_max_abs=max_error,
                bytes=file.stat().st_size,
                sha256=sha(file.read_bytes()),
            )
        )
    gas = sum(r["frames"] for r in manifest)
    passed = 50000 <= gas <= 100000 and all(
        r["success"] and r["near_success"] + r["other_failure"] for r in manifest
    )
    table(OUT / "dataset_manifest.csv", manifest)
    save(
        OUT / "dataset_manifest.json",
        dict(
            gate="PASS" if passed else "FAIL",
            splits=manifest,
            env_id=source["env_id"],
            teacher_sha256=source["model_sha256"],
            body_contract_hash=source["body_contract_hash"],
            actuator_order_hash=source["actuator_order_hash"],
            reward_terms=terms,
            goal_overlap=0,
            timestep_convention="obs and raw/native action at decision; ctrl, act, qpos, qvel and task error after step; goal hashes identify complete target instances",
            simulated_steps=sim_steps,
            wall_s=time.perf_counter() - start,
        ),
    )
    if not passed:
        raise ValueError("Dataset quantity or outcome coverage gate failed")


if __name__ == "__main__":
    collect()
