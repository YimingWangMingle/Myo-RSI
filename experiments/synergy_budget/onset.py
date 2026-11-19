"""Locate trajectory divergence in named joints and test identity on recorded actions."""

import time
from collections import Counter

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from myo_rsi.bank_study import Projection
from myo_rsi.expert_data import DATA
from myo_rsi.rerun_v01r import cpu, load_teacher, rollout
from .capacity import first
from .common import OUT, atomic_table, bank, save, seeds


def run():
    start = time.perf_counter()
    identity = Projection(np.eye(39, dtype=np.float32))
    sq = 0.0
    norm = 0.0
    worst = 0.0
    with np.load(DATA / "rollouts_val.npz") as d:
        u = d["ctrl"]
        a = d["raw_action"]
        for j in range(0, len(a), 250):
            target = u[j : j + 250]
            native = identity(torch.from_numpy(a[j : j + 250]))
            reconstructed = cpu(torch.sigmoid(5 * (native - 0.5)))
            diff = reconstructed.astype(float) - target
            sq += np.sum(diff**2)
            norm += np.sum(target.astype(float) ** 2)
            worst = max(worst, abs(diff).max())
    if worst > 2e-6:
        raise ValueError("Identity failed archived physical commands")
    save(
        OUT / "identity_holdout.json",
        dict(
            heldout_frames=len(u),
            applied_relative_frobenius=float(np.sqrt(sq / norm)),
            maximum_entry_error=float(worst),
            wall_s=time.perf_counter() - start,
        ),
    )
    cohort = seeds("projection_dev")
    env, policy, source = load_teacher("T1", len(cohort))
    muscles = bank(8, 11)[2]
    joints = [
        env.sim.mj_model.joint(i).name.split("/")[-1]
        for i in range(env.sim.mj_model.njnt)
    ]
    rows = []
    summary = []
    try:
        for k in [0, 8, 16, 24, 32]:
            tr = {key: [] for key in ["q", "ctrl", "alive"]}

            def recorder(t, active, before, command, raw, weighted, err, goals):
                tr["q"].append(cpu(raw.sim.data.qpos))
                tr["ctrl"].append(cpu(raw.sim.data.ctrl))
                tr["alive"].append(active.copy())

            tic = time.perf_counter()
            project = Projection(bank(k, 11)[0]) if k else None
            episodes, steps = rollout(
                env, policy, "T1", cohort, transform=project, recorder=recorder
            )
            tr = {n: np.stack(v) for n, v in tr.items()}
            goals = [(r["goal_sha256"], r["reset_obs_sha256"]) for r in episodes]
            if not k:
                base = tr
                ref = goals
            elif goals != ref:
                raise ValueError("Unmatched development target")
            nt = min(len(tr["q"]), len(base["q"]))
            for j, seed in enumerate(cohort):
                mask = base["alive"][:nt, j] & tr["alive"][:nt, j]
                delta = tr["q"][:nt, j][mask] - base["q"][:nt, j][mask]
                exc = tr["ctrl"][:nt, j][mask] - base["ctrl"][:nt, j][mask]
                tq = first(np.sqrt(np.mean(delta**2, axis=1)) >= 0.02)
                tu = first(np.sqrt(np.mean(exc**2, axis=1)) >= 0.05)
                rows.append(
                    dict(
                        K=k,
                        nmf_seed=11 if k else "",
                        target_seed=seed,
                        goal=goals[j][0],
                        joint_onset=tq,
                        joint_name=joints[np.argmax(abs(delta[tq - 1]))] if tq else "",
                        joint_max_abs_rad=float(np.max(abs(delta[tq - 1])))
                        if tq
                        else None,
                        excitation_onset=tu,
                        muscle_name=muscles[np.argmax(abs(exc[tu - 1]))] if tu else "",
                        muscle_max_abs=float(np.max(abs(exc[tu - 1]))) if tu else None,
                        censored_or_terminal_length=len(delta),
                    )
                )
            obs = [r for r in rows if r["K"] == k and r["joint_onset"] is not None]
            counts = Counter(r["joint_name"] for r in obs)
            summary.append(
                dict(
                    K=k,
                    episodes=len(cohort),
                    diverged=len(obs),
                    median_first_joint_step=float(
                        np.median([r["joint_onset"] for r in obs])
                    )
                    if obs
                    else None,
                    most_frequent_joint=counts.most_common(1)[0][0] if obs else "",
                    episodes_with_that_joint=counts.most_common(1)[0][1] if obs else 0,
                    simulated_steps=steps,
                    wall_s=time.perf_counter() - tic,
                )
            )
    finally:
        env.close()
    atomic_table(OUT / "divergence_targets.csv", rows)
    atomic_table(OUT / "divergence_onset.csv", summary)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        run()
