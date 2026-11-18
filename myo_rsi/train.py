"""Train a single V0.1 run, enforcing the simulated-body gate before PPO."""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.registry import load_rl_cfg
from myosuite import make_env

from .bank import load_bank
from .controller import HandVecEnv, install_local
from .probe import describe_model, versions
from .protocol import TASKS, TRAIN_SEEDS

METHODS = {
    "A": "Direct PPO",
    "B": "Synergy PPO",
    "C0": "C0 Frozen",
    "C1": "C1 Full FT",
    "C2": "C2 Local FT",
}
FOLDERS = {
    "A": "direct_ppo",
    "B": "synergy_ppo_k8",
    "C0": "adapt_c0_frozen",
    "C1": "adapt_c1_full",
    "C2": "adapt_c2_local",
}
TASK_DIR = {"T1": "reach_random", "T2": "pose_random", "T3": "reach_sarc"}
N_ENVS = 256
ITERATIONS = 256
ADAPT_ITERATIONS = 64
INTERVALS = 4
BANK = Path("artifacts/synergy_bank_k8.npz")


def folder(task, method, seed):
    return Path("checkpoints") / FOLDERS[method] / TASK_DIR[task] / str(seed)


def json_save(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf8")
    tmp.replace(path)


def start_env(task, method, n_envs, seed, collect=False, collect_from=0):
    pf = json.loads(Path("results/preflight.json").read_text(encoding="utf8"))
    if not pf["normal_gate"] or (task == "T3" and not pf["adaptation_gate"]):
        raise RuntimeError("Required compatibility gate has not passed")
    if versions() != pf["host"]["packages"]:
        raise RuntimeError("Installed packages differ from P0; rerun the probe")
    env = make_env(
        TASKS[task], backend="mjlab", num_envs=n_envs, seed=seed, device="cuda:0"
    )
    try:
        ref = next(
            r for r in pf["rows"] if r["task_id"] == task and r["backend"] == "mjlab"
        )
        model = describe_model(env.sim.mj_model)
        if model["hand_contract_hash"] != ref["hand_contract_hash"]:
            raise ValueError("Hand contract differs from P0")
        weights, bank = (
            (None, None) if method == "A" else load_bank(BANK, model["actuator_order"])
        )
        return HandVecEnv(env, weights, collect, collect_from), ref, bank
    except BaseException:
        env.close()
        raise


def agent(env, task, seed, logdir=None):
    cfg = asdict(load_rl_cfg(TASKS[task]))
    cfg.update(seed=seed, logger="tensorboard", save_interval=10**9)
    runner = MjlabOnPolicyRunner(
        env, cfg, str(logdir) if logdir else None, str(env.device)
    )
    return runner


def train(task, method, seed):
    if (task == "T3") != method.startswith("C") or seed not in TRAIN_SEEDS:
        raise ValueError("Invalid method/task/seed")
    out = folder(task, method, seed)
    if out.exists():
        raise FileExistsError(f"Run directory already exists: {out}")
    budget = ADAPT_ITERATIONS if task == "T3" else ITERATIONS
    collect = (method, task) == ("A", "T1")
    env, ref, bank = start_env(task, method, N_ENVS, seed, collect)
    try:
        cfg = load_rl_cfg(TASKS[task])
        horizon = cfg.num_steps_per_env
        env.collect_from = 3 * budget * horizon // 4
        out.mkdir(parents=True, exist_ok=False)
        run_id = f"{task}-{method}-{seed}"
        meta = dict(
            run_id=run_id,
            task_id=task,
            method=METHODS[method],
            code=method,
            seed=seed,
            env_id=TASKS[task],
            num_envs=N_ENVS,
            horizon=horizon,
            budget_iterations=0 if method == "C0" else budget,
            policy_action_dim=env.num_actions,
            bank_hash=bank["bank_hash"] if bank else None,
            model_hash=ref["model_hash"],
            actuator_names=ref["actuator_order"],
            package_versions=versions(),
            source_checkpoint_id=None,
            status="RUNNING",
        )
        json_save(out / "run.json", meta)
        # Save the unmodified upstream config before RSL resolves class-name strings.
        json_save(out / "ppo.json", asdict(cfg))
        runner = agent(env, task, seed, out / "log")
        if method.startswith("C"):
            parent = folder("T1", "B", seed)
            initial = json.loads((parent / "run.json").read_text(encoding="utf8"))
            if (
                initial["status"] != "COMPLETE"
                or initial["bank_hash"] != meta["bank_hash"]
            ):
                raise ValueError("Adaptation source is incomplete or bank differs")
            path = parent / f"policy_{ITERATIONS}.pt"
            runner.load(str(path), load_cfg={"actor": True, "critic": True})
            meta["source_checkpoint_id"] = f"{initial['run_id']}/{path.name}"
            if method == "C2":
                install_local(runner)
            # Residual initialization consumes RNG. Restore a common initial simulator,
            # target distribution and PPO random sequence across the three branches.
            env.unwrapped.reset(seed=seed)
        actor = runner.alg.get_policy()
        meta["trainable_actor_params"] = (
            0
            if method == "C0"
            else sum(p.numel() for p in actor.parameters() if p.requires_grad)
        )
        meta["actor_params"] = sum(p.numel() for p in actor.parameters())
        torch.cuda.synchronize()
        tic = time.perf_counter()
        points = []

        def save(it):
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - tic
            data = runner.alg.save()
            data.update(
                iter=it, infos={"run_id": run_id, "bank_hash": meta["bank_hash"]}
            )
            tmp = out / f"policy_{it}.tmp"
            torch.save(data, tmp)
            tmp.replace(out / f"policy_{it}.pt")
            points.append(
                dict(
                    iteration=it,
                    environment_steps=it * horizon * N_ENVS,
                    elapsed_wall_s=elapsed,
                    gpu_hours=elapsed / 3600,
                    train_return=(
                        float(np.mean(runner.logger.rewbuffer))
                        if runner.logger.rewbuffer
                        else None
                    ),
                )
            )
            json_save(out / "points.json", points)

        save(0)
        if method != "C0":
            original = runner.logger.log

            def observe(**kw):
                original(**kw)
                completed = kw["it"] + 1
                if completed % (budget // INTERVALS) == 0:
                    save(completed)

            runner.logger.log = observe
            runner.learn(budget, init_at_random_ep_len=True)
        if collect:
            target = Path("datasets/direct_rollouts/reach_random_train")
            target.mkdir(parents=True, exist_ok=True)
            if not env.records:
                raise ValueError("No physical excitation records")
            records = {
                k: np.concatenate([r[k] for r in env.records]) for k in env.records[0]
            }
            np.savez_compressed(
                target / f"{run_id}.npz",
                **records,
                metadata=json.dumps(dict(meta, split="train", signal="data.ctrl")),
            )
        if bank:
            _, after = load_bank(BANK, ref["actuator_order"])
            if after["bank_hash"] != meta["bank_hash"]:
                raise ValueError("Bank changed during training")
        clipped, error, entries = env.saturation.tolist()
        meta.update(
            status="COMPLETE",
            wall_s=time.perf_counter() - tic,
            projection_fraction=clipped / entries if entries else None,
            projection_mae=error / entries if entries else None,
        )
        json_save(out / "run.json", meta)
    finally:
        env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=list(TASK_DIR))
    p.add_argument("method", choices=list(METHODS))
    p.add_argument("seed", type=int, choices=TRAIN_SEEDS)
    args = p.parse_args()
    train(args.task, args.method, args.seed)
