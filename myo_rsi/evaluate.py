"""Evaluate deterministic mean actions. Each vector lane gets its own fixed seed."""

import argparse
import hashlib
import json
import time

import numpy as np
import torch
from tensordict import TensorDict

from .controller import install_local
from .protocol import FINAL_SEEDS, VALIDATION_SEEDS
from .train import METHODS, agent, folder, json_save, start_env


@torch.inference_mode()
def episodes(env, actor, seeds):
    raw = env.unwrapped
    goals = raw.command_manager.active_terms
    if len(goals) != 1 or len(seeds) != env.num_envs:
        raise ValueError("Unexpected evaluation layout")
    for lane, seed in enumerate(seeds):
        ids = torch.tensor([lane], device=raw.device)
        obs, _ = raw.reset(seed=seed, env_ids=ids)
    initial_goal = raw.command_manager.get_command(goals[0]).detach().cpu().numpy()
    initial_obs = obs["actor"].detach().cpu().numpy().copy()
    total = np.zeros(len(seeds))
    records = {}
    transitions = 0
    for tick in range(1, raw.max_episode_length + 1):
        vec = TensorDict(obs, batch_size=[env.num_envs])
        action = env.native(actor(vec))
        obs, reward, term, trunc, _ = raw.step(action)
        transitions += env.num_envs
        total += reward.cpu().numpy()
        for lane in (term | trunc).nonzero().flatten().tolist():
            if lane in records:
                continue
            val = dict(raw.metrics_manager.get_active_iterable_terms(lane))["success"][
                0
            ]
            records[lane] = dict(
                eval_seed=seeds[lane],
                episode_id=lane,
                success=int(bool(val)),
                episode_return=float(total[lane]),
                episode_length=tick,
                terminal_reason=("terminated" if bool(term[lane]) else "time_limit"),
                goal_hash=hashlib.sha256(initial_goal[lane].tobytes()).hexdigest(),
                reset_obs_hash=hashlib.sha256(initial_obs[lane].tobytes()).hexdigest(),
            )
        if len(records) == len(seeds):
            return [records[i] for i in range(len(seeds))], transitions
    raise RuntimeError(f"Only {len(records)}/{len(seeds)} evaluation episodes finished")


def evaluate(task, method, seed):
    out = folder(task, method, seed)
    meta = json.loads((out / "run.json").read_text(encoding="utf8"))
    if meta["status"] != "COMPLETE":
        raise ValueError("Incomplete training run")
    points = json.loads((out / "points.json").read_text(encoding="utf8"))
    for eval_task in (task, "T1") if task == "T3" else (task,):
        file = out / f"eval_{eval_task}.json"
        if file.exists():
            continue
        tic = time.perf_counter()
        env, _, bank = start_env(eval_task, method, len(FINAL_SEEDS), seed)
        try:
            if (bank["bank_hash"] if bank else None) != meta["bank_hash"]:
                raise ValueError("Evaluation bank differs from checkpoint")
            runner = agent(env, task, seed)
            if method == "C2":
                install_local(runner)
            all_points = []
            for point in points:
                it = point["iteration"]
                name = f"policy_{it}.pt"
                runner.load(str(out / name), load_cfg={"actor": True, "critic": True})
                actor = runner.get_inference_policy(device=env.device)
                # Validation list generates curves. Independent final list is used
                # only for the locked final checkpoint and adaptation pre/post pairs.
                splits = [("validation", VALIDATION_SEEDS)]
                if it == meta["budget_iterations"] or (task == "T3" and it == 0):
                    splits.append(("final", FINAL_SEEDS))
                for split, seeds in splits:
                    start = time.perf_counter()
                    batch, cost = episodes(env, actor, seeds)
                    all_points.append(
                        dict(
                            checkpoint_id=meta["run_id"] + "/" + name,
                            iteration=it,
                            environment_steps=point["environment_steps"],
                            eval_task=eval_task,
                            split=split,
                            episodes=batch,
                            sim_transitions=cost,
                            wall_s=time.perf_counter() - start,
                        )
                    )
            json_save(
                file, dict(points=all_points, stage_wall_s=time.perf_counter() - tic)
            )
        finally:
            env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["T1", "T2", "T3"])
    parser.add_argument("method", choices=METHODS)
    parser.add_argument("seed", type=int)
    args = parser.parse_args()
    evaluate(args.task, args.method, args.seed)
