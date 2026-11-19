"""Train or evaluate one from-scratch PPO policy in the matched-budget matrix."""

import argparse
import time
from dataclasses import asdict

import numpy as np
import torch
from mjlab.tasks.registry import load_rl_cfg
from myosuite import make_env
from tensordict import TensorDict

from myo_rsi.bank_study import LOW, HIGH
from myo_rsi.controller import HandVecEnv
from myo_rsi.probe import describe_model, versions
from myo_rsi.protocol import TASKS
from myo_rsi.rerun_v01r import rollout
from myo_rsi.train import agent

from .common import CKPT, LOCK, OUT, OLD, atomic_table, bank, read, save, seeds, sha

POINTS = LOCK["iterations"] + LOCK["extension"]
BATCH = LOCK["n_envs"] * LOCK["horizon"]


class RecruitmentEnv(HandVecEnv):
    def __init__(self, raw, k):
        super().__init__(raw)
        self.k = k
        self.bank_hash = None
        if k != 39:
            w, q, n, m = bank(k, LOCK["policy_bank_seed"])
            if n != describe_model(raw.sim.mj_model)["actuator_order"]:
                raise ValueError("Actuator order mismatch")
            self.W = torch.as_tensor(w.copy(), device=self.device)
            self.scale = torch.as_tensor(q.copy(), device=self.device)
            self.num_actions = k
            self.bank_hash = m["bank_hash"]

    def native(self, action, track=False):
        if self.k == 39:
            return action.clamp(-1, 1)
        c = self.scale * action.relu()
        u = (c @ self.W.T).clamp(LOW, HIGH)
        return (0.5 + torch.logit(u) / 5).clamp(-1, 1)


def paths(task, k, seed):
    case = f"{task}-K{k}-s{seed}"
    ck = CKPT / case
    cell = OUT / "cells" / case
    ck.mkdir(parents=True, exist_ok=True)
    cell.mkdir(parents=True, exist_ok=True)
    return ck, cell


def setup(task, k, seed, n):
    pf = read(OLD / "preflight.json")
    if versions() != pf["host"]["packages"]:
        raise ValueError("Package lock mismatch")
    raw = make_env(TASKS[task], backend="mjlab", num_envs=n, device="cuda:0", seed=seed)
    try:
        ref = next(
            r for r in pf["rows"] if r["task_id"] == task and r["backend"] == "mjlab"
        )
        if (
            describe_model(raw.sim.mj_model)["hand_contract_hash"]
            != ref["hand_contract_hash"]
        ):
            raise ValueError("Musculoskeletal model mismatch")
        env = RecruitmentEnv(raw, k)
        torch.manual_seed(seed)
        runner = agent(env, task, seed)
        return env, runner
    except BaseException:
        raw.close()
        raise


def train(task, k, seed, target):
    ck, cell = paths(task, k, seed)
    config = asdict(load_rl_cfg(TASKS[task]))
    if config["num_steps_per_env"] != LOCK["horizon"]:
        raise ValueError("Rollout horizon mismatch")
    if not (OUT / "identity" / "complete.json").is_file():
        raise RuntimeError("Verify the action adaptor on the identity control first")
    history = read(cell / "points.json") if (cell / "points.json").exists() else []
    if history and history[-1]["iteration"] >= target:
        return
    env, runner = setup(task, k, seed, LOCK["n_envs"])
    try:
        start = history[-1]["iteration"] if history else 0
        prior = history[-1]["train_wall_s"] if history else 0
        if history:
            file = ck / f"policy_{start}.pt"
            if sha(file.read_bytes()) != history[-1]["sha256"]:
                raise ValueError("Resume checkpoint mismatch")
            info = runner.load(str(file))
            if (
                info["bank_hash"] != env.bank_hash
                or info["seed"] != seed
                or info["task"] != task
            ):
                raise ValueError("Checkpoint provenance mismatch")
            runner.current_learning_iteration = start
        # Every continuing arm starts the extension with a reproducible fresh simulator.
        env.unwrapped.reset(seed=seed + start * 1000)
        config.update(logger="tensorboard", seed=seed, save_interval=10**9)
        metadata = dict(
            task=task,
            env=TASKS[task],
            K=k,
            seed=seed,
            bank_hash=env.bank_hash,
            ppo=config,
            body=describe_model(env.unwrapped.sim.mj_model)["hand_contract_hash"],
            trainable_actor=sum(
                p.numel()
                for p in runner.alg.get_policy().parameters()
                if p.requires_grad
            ),
            n_envs=env.num_envs,
            source="from scratch",
            decoder=LOCK["decoder"],
            versions=versions(),
        )
        save(cell / "config.json", metadata)
        torch.cuda.synchronize()
        tic = time.perf_counter()

        def checkpoint(it):
            torch.cuda.synchronize()
            path = ck / f"policy_{it}.pt"
            tmp = path.with_suffix(".tmp")
            state = runner.alg.save()
            state.update(
                iter=it, infos=dict(bank_hash=env.bank_hash, task=task, seed=seed)
            )
            torch.save(state, tmp)
            tmp.replace(path)
            history.append(
                dict(
                    iteration=it,
                    transitions=it * BATCH,
                    train_wall_s=prior + time.perf_counter() - tic,
                    sha256=sha(path.read_bytes()),
                    segment_start_iteration=start,
                )
            )
            save(cell / "points.json", history)
            print(task, k, seed, it, it * BATCH, flush=True)

        if not history:
            checkpoint(0)

        def monitor(**kw):
            completed = kw["it"] + 1
            if completed in POINTS:
                checkpoint(completed)
            if completed % 10 == 0:
                save(
                    cell / "progress.json",
                    dict(
                        iteration=completed,
                        transitions=completed * BATCH,
                        train_wall_s=prior + time.perf_counter() - tic,
                        learning_rate=float(kw["learning_rate"]),
                    ),
                )

        # No console or tensorboard overhead per iteration. The upstream algorithm is unchanged.
        runner.logger.log = monitor
        runner.learn(target - start, init_at_random_ep_len=True)
        if history[-1]["iteration"] != target:
            raise RuntimeError("Missing terminal checkpoint")
        if k != 39:
            w, q, _, m = bank(k, LOCK["policy_bank_seed"])
            np.testing.assert_array_equal(env.W.cpu().numpy(), w)
            np.testing.assert_array_equal(env.scale.cpu().numpy(), q)
    finally:
        env.close()


def evaluate(task, k, seed, target, group):
    ck, cell = paths(task, k, seed)
    cohort = seeds(group)
    points = [
        r
        for r in read(cell / "points.json")
        if r["iteration"] <= target and (group != "test" or r["iteration"] == target)
    ]
    if group == "test" and not (OUT / f"{task}_budget.json").exists():
        raise ValueError("Freeze the common budget before final evaluation")
    pending = [
        r for r in points if not (cell / f"{group}_{r['iteration']}.json").exists()
    ]
    if not pending:
        return
    env, runner = setup(task, k, seed, len(cohort))
    try:
        env.unwrapped.cfg.auto_reset = False
        for p in pending:
            it = p["iteration"]
            file = ck / f"policy_{it}.pt"
            if sha(file.read_bytes()) != p["sha256"]:
                raise ValueError("Evaluation checkpoint mismatch")
            runner.load(str(file), load_cfg={"actor": True})
            policy = runner.get_inference_policy()

            def act(obs):
                vector = TensorDict({"actor": obs}, batch_size=[len(obs)])
                return env.native(policy(vector))

            tic = time.perf_counter()
            rows, steps = rollout(env.unwrapped, act, task, cohort)
            atomic_table(cell / f"{group}_{it}_episodes.csv", rows)
            save(
                cell / f"{group}_{it}.json",
                dict(
                    task=task,
                    K=k,
                    seed=seed,
                    group=group,
                    iteration=it,
                    training_steps=it * BATCH,
                    checkpoint_sha256=p["sha256"],
                    bank_hash=env.bank_hash,
                    episodes=len(rows),
                    successes=sum(r["success"] for r in rows),
                    mean_terminal_error=float(
                        np.mean([r["terminal_error"] for r in rows])
                    ),
                    mean_return=float(np.mean([r["episode_return"] for r in rows])),
                    eval_steps=steps,
                    eval_wall_s=time.perf_counter() - tic,
                ),
            )
    finally:
        env.close()


def replay_unmatched_baselines(task):
    """Keep recovered, off-schedule simulator resets outside the matched cohort."""
    for history_file in sorted((OUT / "cells").glob(f"{task}-*/points.json")):
        points = read(history_file)
        off_schedule = {p["segment_start_iteration"] for p in points} - {0, 1000}
        if not off_schedule:
            continue
        case = history_file.parent.name
        cfg = read(history_file.parent / "config.json")
        k, seed = cfg["K"], cfg["seed"]
        if (OUT / f"{task}_budget.json").exists():
            raise RuntimeError("A published budget requires explicit recovery review")
        checkpoints, cell = paths(task, k, seed)
        token = "attempt_" + str(time.time_ns())
        for directory, base in ((checkpoints, CKPT), (cell, OUT)):
            destination = base / "restarted" / case / token
            if not directory.resolve().is_relative_to(
                base.resolve()
            ) or not destination.resolve().is_relative_to(base.resolve()):
                raise ValueError("Unexpected archive directory")
            destination.parent.mkdir(parents=True, exist_ok=True)
            directory.rename(destination)
        save(
            OUT / "restarted" / case / token / "recovery.json",
            dict(
                off_schedule_resets=sorted(off_schedule),
                kept_out_of_primary=True,
                reason="Host session interruption before the common extension boundary",
                cost_note="Saved checkpoint counters give a lower bound; work after the last checkpoint before interruption is unknown.",
            ),
        )
        train(task, k, seed, 1000)
        evaluate(task, k, seed, 1000, "validation")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "validation", "test"])
    p.add_argument("task", choices=["T1", "T2"])
    p.add_argument("k", type=int)
    p.add_argument("seed", type=int, choices=LOCK["training_seeds"])
    p.add_argument("target", type=int, choices=[1000, 1500])
    a = p.parse_args()
    if a.mode == "train":
        replay_unmatched_baselines(a.task)
        train(a.task, a.k, a.seed, a.target)
    else:
        evaluate(a.task, a.k, a.seed, a.target, a.mode)
