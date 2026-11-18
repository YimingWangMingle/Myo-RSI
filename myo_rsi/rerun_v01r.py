"""Reproduce the locked v0.1r teacher gate on the native GPU tasks."""

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from myosuite import make_env
from myosuite.utils.checkpoint_manifest import contract_hash, env_contract
from myosuite.utils.rslrl_policy import load_rslrl_policy

from .probe import describe_model, versions
from .protocol import TASKS, VALIDATION_SEEDS
from .tables import wilson

OUT = Path("results/v0_1r")
OFFICIAL = Path("checkpoints/v0_1r/official")
MODELS = {"T1": "model_1912.pt", "T2": "model_24999.pt", "T0": "model_97.pt"}


def read(file):
    return json.loads(file.read_text(encoding="utf8"))


def save(file, data):
    tmp = file.with_suffix(file.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf8")
    tmp.replace(file)


def table(file, rows):
    if not rows:
        raise ValueError("An observation table needs records")
    with file.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify_file(folder, name, tree):
    data = (folder / name).read_bytes()
    item = next(
        row for row in tree if row["path"] == f"checkpoints/{folder.name}/{name}"
    )
    digest = sha(data)
    if len(data) != item["size"]:
        raise ValueError(f"Wrong length for {name}")
    if "lfs" in item:
        if digest != item["lfs"]["oid"]:
            raise ValueError(f"LFS checksum mismatch for {name}")
    elif hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() != item["oid"]:
        raise ValueError(f"Git blob checksum mismatch for {name}")
    return digest


def load_teacher(task, lanes):
    lock = read(OUT / "run_lock.json")
    probe = read(OUT / "preflight.json")
    if not probe["normal_gate"] or versions() != probe["host"]["packages"]:
        raise RuntimeError("Environment preflight or dependency lock failed")
    env_id = TASKS[task]
    folder = OFFICIAL / env_id
    tree = read(folder / "tree.json")
    name = MODELS[task]
    manifest_sha = verify_file(folder, "manifest.json", tree)
    weights_sha = verify_file(folder, name, tree)
    manifest = read(folder / "manifest.json")
    contract = env_contract(env_id)
    if (
        manifest["env_id"] != env_id
        or name not in manifest["files"]
        or manifest["contract"] != contract
        or manifest["contract_hash"] != contract_hash(contract)
    ):
        raise ValueError(
            f"Upstream/local environment contract mismatch: {manifest}, {contract}"
        )
    env = make_env(
        env_id, backend="mjlab", num_envs=lanes, device=lock["device"], seed=701
    )
    try:
        env.cfg.auto_reset = False
        model = describe_model(env.sim.mj_model)
        reference = next(
            r for r in probe["rows"] if r["task_id"] == task and r["backend"] == "mjlab"
        )
        terms = list(env.cfg.observations["actor"].terms)
        policy = (
            load_rslrl_policy(folder / name, env.action_manager.total_action_dim)
            .to(env.device)
            .eval()
        )
        if (
            model["actuator_order"] != reference["actuator_order"]
            or model["hand_contract_hash"] != reference["hand_contract_hash"]
            or policy.obs_dim != contract["obs_shape"][0]
            or policy.action_dim != contract["action_shape"][0]
            or terms != contract["obs_keys"]
            or abs(env.step_dt - contract["ctrl_dt"]) > 1e-9
        ):
            raise ValueError(
                "GPU environment, policy dimensions or observation order mismatch"
            )
        return (
            env,
            policy,
            dict(
                task=task,
                env_id=env_id,
                model=name,
                hf_tag=lock["hf_tag"],
                hf_revision=lock["hf_sha"],
                model_sha256=weights_sha,
                manifest_sha256=manifest_sha,
                upstream_manifest=manifest,
                current_contract=contract,
                actuator_order_hash=model["actuator_order_hash"],
                body_contract_hash=model["hand_contract_hash"],
                normalizer=type(policy.normalizer).__name__,
                actor=str(policy.mlp),
                deterministic_mean=True,
            ),
        )
    except BaseException:
        env.close()
        raise


def cpu(tensor):
    return tensor.detach().cpu().numpy().copy()


@torch.inference_mode()
def rollout(env, policy, task, seeds, transform=None, recorder=None):
    """One first episode per seeded lane. Log the reward manager's pre-reset task distance."""
    if len(seeds) != env.num_envs:
        raise ValueError("Lane count differs from cohort")
    term = "pose" if task == "T2" else "reach"
    for lane, seed in enumerate(seeds):
        obs, _ = env.reset(
            seed=int(seed), env_ids=torch.tensor([lane], device=env.device)
        )
    goals = cpu(env.command_manager.get_command(term))
    start_obs = cpu(obs["actor"])
    error_col = env.reward_manager.active_terms.index(term)
    reward_weight = env.reward_manager.cfg[term].weight
    metric_col = env.metrics_manager.active_terms.index("success")
    success_bound = 0.7 if task == "T2" else 0.0625
    rows = {}
    active = np.ones(len(seeds), dtype=bool)
    returns = np.zeros(len(seeds))
    for t in range(1, env.max_episode_length + 1):
        before = obs["actor"]
        command = policy(before)
        action = command if transform is None else transform(command)
        obs, reward, terminated, truncated, _ = env.step(action)
        # Pinned mjlab stores per-step weighted terms before reset and before dt scaling.
        rwds = env.reward_manager._step_reward
        errors = -rwds[:, error_col] / reward_weight
        successes = env.metrics_manager._step_values[:, metric_col]
        valid = cpu(terminated | truncated)
        returns += cpu(reward) * active
        if recorder is not None:
            recorder(t, active, before, command, env, rwds, errors, goals)
        for lane in np.flatnonzero(valid & active):
            error = float(errors[lane])
            success = int(successes[lane].item())
            if bool(success) != (error < success_bound):
                raise ValueError(
                    f"Native success and recorded error disagree: {success}, {error}"
                )
            rows[lane] = dict(
                task=task,
                seed=int(seeds[lane]),
                success=success,
                terminal_error=error,
                error_unit="rad" if task == "T2" else "m",
                episode_return=float(returns[lane]),
                length=t,
                termination="terminated" if bool(terminated[lane]) else "time_limit",
                goal_sha256=sha(goals[lane].tobytes()),
                reset_obs_sha256=sha(start_obs[lane].tobytes()),
            )
        active[valid] = False
        if not active.any():
            return [rows[i] for i in range(len(seeds))], t * len(seeds)
        if valid.any():
            obs, _ = env.reset(env_ids=(terminated | truncated).nonzero().flatten())
    raise RuntimeError("Unfinished episodes at horizon")


def teacher():
    rows, summaries, manifests = [], [], []
    if (OUT / "teacher_eval_summary.csv").exists():
        with (OUT / "teacher_eval_summary.csv").open(newline="") as f:
            summaries = list(csv.DictReader(f))
        with (OUT / "teacher_eval_episodes.csv").open(newline="") as f:
            rows = list(csv.DictReader(f))
        manifests = read(OUT / "teacher_manifest.json")
    for task in ("T1", "T2"):
        if any(r["task"] == task for r in summaries):
            existing = [r for r in rows if r["task"] == task]
            summary = next(r for r in summaries if r["task"] == task)
            if [int(r["seed"]) for r in existing] != list(VALIDATION_SEEDS) or sum(
                int(r["success"]) for r in existing
            ) != int(summary["successes"]):
                raise ValueError("Incomplete persisted teacher cohort")
            continue
        start = time.perf_counter()
        env, policy, info = load_teacher(task, len(VALIDATION_SEEDS))
        try:
            measured, steps = rollout(env, policy, task, VALIDATION_SEEDS)
        finally:
            env.close()
        rows.extend(measured)
        count = sum(r["success"] for r in measured)
        n = len(measured)
        lo, hi = wilson(count, n)
        summaries.append(
            dict(
                task=task,
                successes=count,
                episodes=n,
                fraction=count / n,
                wilson95_low=lo,
                wilson95_high=hi,
                mean_terminal_error=float(
                    np.mean([r["terminal_error"] for r in measured])
                ),
                error_unit=measured[0]["error_unit"],
                valid_steps=sum(r["length"] for r in measured),
                simulated_steps=steps,
                wall_s=time.perf_counter() - start,
                gate=(
                    "PASS"
                    if count >= read(OUT / "run_lock.json")["teacher_required"]
                    else "FAIL"
                )
                if task == "T1"
                else "CONTEXT",
            )
        )
        manifests.append(info)
        save(OUT / "teacher_manifest.json", manifests)
        table(OUT / "teacher_eval_episodes.csv", rows)
        table(OUT / "teacher_eval_summary.csv", summaries)
        print(task, count, "/", n, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["teacher"], required=True)
    parser.parse_args()
    teacher()
