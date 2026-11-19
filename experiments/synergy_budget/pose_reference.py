"""Challenge the official Pose policy with the shared Reach banks on frozen final goals."""

import time
import numpy as np
from threadpoolctl import threadpool_limits

from myo_rsi.bank_study import Projection
from myo_rsi.rerun_v01r import load_teacher, rollout
from .common import LOCK, OUT, atomic_table, bank, save, seeds


def evaluate():
    if not (OUT / "T2_complete.json").exists():
        raise RuntimeError(
            "Complete the Pose matrix and freeze every test choice first"
        )
    folder = OUT / "pose_projection_test"
    folder.mkdir(exist_ok=True)
    if (folder / "complete.json").exists():
        return
    cohort = seeds("test")
    env, actor, source = load_teacher("T2", len(cohort))
    summary = []
    episodes = []
    try:
        ref = None
        for k in [0, 39] + LOCK["k"]:
            if k == 0:
                mapping = None
                label = "Teacher"
                digest = ""
            elif k == 39:
                mapping = Projection(np.eye(k, dtype=np.float32))
                label = "Identity"
                digest = "identity39"
            else:
                w, _, _, m = bank(k, LOCK["policy_bank_seed"])
                mapping = Projection(w)
                label = "NMF"
                digest = m["bank_hash"]
            start = time.perf_counter()
            rows, steps = rollout(env, actor, "T2", cohort, transform=mapping)
            pair = [(r["goal_sha256"], r["reset_obs_sha256"]) for r in rows]
            if ref is None:
                ref = pair
            elif ref != pair:
                raise ValueError("Pose reference targets differ")
            key = dict(
                controller=label,
                K=k,
                nmf_seed=LOCK["policy_bank_seed"] if label == "NMF" else 0,
                bank_hash=digest,
            )
            summary.append(
                dict(
                    **key,
                    successes=sum(r["success"] for r in rows),
                    episodes=len(rows),
                    mean_terminal_error_rad=float(
                        np.mean([r["terminal_error"] for r in rows])
                    ),
                    simulated_steps=steps,
                    wall_s=time.perf_counter() - start,
                )
            )
            episodes.extend(dict(**key, **r) for r in rows)
    finally:
        env.close()
    atomic_table(folder / "summary.csv", summary)
    atomic_table(folder / "episodes.csv", episodes)
    save(
        folder / "complete.json",
        dict(teacher=source["model_sha256"], task="T2", group="test"),
    )


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        evaluate()
