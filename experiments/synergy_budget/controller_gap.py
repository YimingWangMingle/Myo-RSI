"""Compare the learned and teacher-projected controllers on the exact same final targets."""

import matplotlib.pyplot as plt
import numpy as np

from .common import LOCK, OUT, atomic_table, readcsv


def compare(finals, figs):
    output = []
    for task, folder in [("T1", "capacity_test"), ("T2", "pose_projection_test")]:
        path = OUT / folder / "episodes.csv"
        if not path.exists():
            continue
        projections = readcsv(path)
        task_rows = [r for r in finals if r["task"] == task]
        ks = sorted({r["K"] for r in task_rows}, key=lambda k: 0 if k == 39 else k)
        if not ks:
            continue
        fig, ax = plt.subplots(figsize=(6.3, 3.6), layout="constrained")
        for i, k in enumerate(ks):
            refs = [
                r
                for r in projections
                if (
                    r["controller"] == "Teacher"
                    if k == 39
                    else r["controller"] == "NMF"
                    and int(r["K"]) == k
                    and int(r["nmf_seed"]) == LOCK["policy_bank_seed"]
                )
            ]
            reference = {int(r["seed"]): r for r in refs}
            if len(reference) != 100:
                raise ValueError("Reference cohort incomplete")
            nref = sum(int(r["success"]) for r in refs)
            ax.scatter(
                i - 0.13,
                nref,
                marker="D",
                color="#b35821",
                label="Teacher in action space" if i == 0 else None,
            )
            for j, seed in enumerate(LOCK["training_seeds"]):
                sub = [r for r in task_rows if r["K"] == k and r["train_seed"] == seed]
                if not sub:
                    continue
                if len(sub) != 100 or any(
                    (r["goal_sha256"], r["reset_obs_sha256"])
                    != (
                        reference[int(r["seed"])]["goal_sha256"],
                        reference[int(r["seed"])]["reset_obs_sha256"],
                    )
                    for r in sub
                ):
                    raise ValueError("Trained and teacher final target mismatch")
                diff = np.array(
                    [
                        int(r["success"]) - int(reference[int(r["seed"])]["success"])
                        for r in sub
                    ]
                )
                output.append(
                    dict(
                        task=task,
                        K=k,
                        training_seed=seed,
                        teacher_successes=nref,
                        trained_successes=nref + int(diff.sum()),
                        training_minus_projection_pp=int(diff.sum()),
                        targets_gained=int((diff == 1).sum()),
                        targets_lost=int((diff == -1).sum()),
                        episodes=len(sub),
                    )
                )
                ax.scatter(
                    i + 0.06 + 0.055 * j,
                    nref + diff.sum(),
                    color="#165d81",
                    s=22,
                    label="Trained PPO, each seed" if (i, j) == (0, 0) else None,
                )
        ax.set(
            xticks=range(len(ks)),
            xticklabels=["Direct 39D" if k == 39 else f"K={k}" for k in ks],
            ylabel="Successful targets / 100",
            ylim=(-3, 103),
            title=("Reach" if task == "T1" else "Pose") + " on paired final targets",
        )
        ax.legend(frameon=False, fontsize=8)
        for ext in ("png", "svg"):
            fig.savefig(
                figs / f"{task}_trained_vs_projected.{ext}",
                dpi=220,
                bbox_inches="tight",
            )
        plt.close(fig)
    if output:
        atomic_table(OUT / "trained_vs_projected.csv", output)
    return output
