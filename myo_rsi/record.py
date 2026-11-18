"""Archive an instrumented deployment episode per completed final checkpoint."""

import json
import time
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from .controller import install_local
from .train import agent, json_save, start_env


@torch.inference_mode()
def record(meta_file):
    meta = json.loads(meta_file.read_text(encoding="utf8"))
    if meta["status"] != "COMPLETE":
        return
    dest = Path("datasets/diagnostic_rollouts") / meta["run_id"]
    if (dest / "manifest.json").exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    tic = time.perf_counter()
    task, method, seed = meta["task_id"], meta["code"], meta["seed"]
    env, _, _ = start_env(task, method, 1, seed)
    try:
        runner = agent(env, task, seed)
        if method == "C2":
            install_local(runner)
        path = meta_file.parent / f"policy_{meta['budget_iterations']}.pt"
        runner.load(str(path), load_cfg={"actor": True, "critic": True})
        actor = runner.get_inference_policy()
        raw = env.unwrapped
        raw.cfg.auto_reset = False
        obs, _ = raw.reset(seed=8501)
        goalname = raw.command_manager.active_terms[0]
        samples = []
        for step in range(raw.max_episode_length):
            inputs = TensorDict(obs, batch_size=[1])
            latent = actor(inputs)
            applied = env.native(latent)

            def cpu(tensor):
                return tensor.detach().cpu().numpy()[0].copy()

            sample = dict(
                timestep=step,
                raw_action=cpu(latent),
                native_action=cpu(applied),
                pre_observation=cpu(obs["actor"]),
                goal=cpu(raw.command_manager.get_command(goalname)),
            )
            if env.weights is not None:
                sample["recruitment"] = cpu(0.15 * torch.nn.functional.softplus(latent))
            obs, r, term, trunc, _ = raw.step(applied)
            terms = dict(raw.reward_manager.get_active_iterable_terms(0))
            solved = dict(raw.metrics_manager.get_active_iterable_terms(0))["success"][
                0
            ]
            sample.update(
                excitation=cpu(raw.sim.data.ctrl),
                activation=cpu(raw.sim.data.act),
                qpos=cpu(raw.sim.data.qpos),
                qvel=cpu(raw.sim.data.qvel),
                reward=float(r),
                native_success=int(bool(solved)),
                terminated=bool(term),
                truncated=bool(trunc),
            )
            for name, value in terms.items():
                sample["reward_" + name] = value[0]
            samples.append(sample)
            if bool(term) or bool(trunc):
                break
        if not (samples[-1]["terminated"] or samples[-1]["truncated"]):
            raise ValueError("Unterminated diagnostic episode")
        np.savez_compressed(
            dest / "trajectory.npz",
            **{k: np.stack([row[k] for row in samples]) for k in samples[0]},
        )
        json_save(
            dest / "manifest.json",
            dict(
                run_id=meta["run_id"],
                task_id=task,
                method=meta["method"],
                training_seed=seed,
                rollout_seed=8501,
                split="diagnostic",
                environment_steps=len(samples),
                checkpoint=path.name,
                bank_hash=meta["bank_hash"],
                wall_s=time.perf_counter() - tic,
                signal_units="ctrl and act in native actuator order; joint angles radians",
                actuator_names=meta["actuator_names"],
                model_hash=meta["model_hash"],
            ),
        )
    finally:
        env.close()


if __name__ == "__main__":
    for source in sorted(Path("checkpoints").glob("*/*/*/run.json")):
        record(source)
