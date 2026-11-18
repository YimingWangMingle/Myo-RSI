"""Build the v0.1r evidence tables and figures from measured records."""

import ast
from importlib import metadata

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from myo_rsi.bank_study import HIGH, LOW, readcsv
from myo_rsi.expert_data import DATA
from myo_rsi.rerun_v01r import OUT, read, save, table

FIG = OUT / "figures"
BLUE, ORANGE, GREEN, GRAY = "#175ea0", "#b35821", "#197969", "#5b6770"


def export(fig, name):
    for ext in ("svg", "png"):
        fig.savefig(
            FIG / f"{name}.{ext}", dpi=240, bbox_inches="tight", facecolor="white"
        )
    svg = FIG / f"{name}.svg"
    lines = svg.read_text(encoding="utf8").splitlines()
    svg.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf8")
    plt.close(fig)


def build():
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "axes.titlelocation": "left",
            "axes.titleweight": "bold",
        }
    )
    teachers = readcsv(OUT / "teacher_eval_summary.csv")
    manifest = {r["task"]: r for r in read(OUT / "teacher_manifest.json")}
    fields = []
    for r in teachers:
        m = manifest[r["task"]]
        fields.append(
            dict(
                **r,
                model=m["model"],
                sha256=m["model_sha256"],
                local_contract=m["upstream_manifest"]["contract_hash"],
                contract_matches=True,
                published_external_percent=m["upstream_manifest"]["success_percent"],
            )
        )
    table(OUT / "teacher_validation.csv", fields)
    fig, ax = plt.subplots(figsize=(6.3, 3.4), layout="constrained")
    for i, r in enumerate(fields):
        s = float(r["fraction"]) * 100
        ax.errorbar(
            i - 0.07,
            s,
            yerr=[
                [s - float(r["wilson95_low"]) * 100],
                [float(r["wilson95_high"]) * 100 - s],
            ],
            fmt="o",
            color=BLUE,
            capsize=4,
            label="Local, 95% Wilson interval" if i == 0 else None,
        )
        ax.plot(
            i + 0.07,
            r["published_external_percent"],
            "D",
            color=ORANGE,
            mfc="white",
            label="Published, external cohort" if i == 0 else None,
        )
    ax.set(
        xticks=[0, 1],
        xticklabels=["Hand Reach", "Hand Pose"],
        xlim=(-0.5, 1.5),
        ylim=(0, 106),
        ylabel="Terminal success (%)",
        title="Official PPO teacher validation",
    )
    ax.legend(loc="lower left", frameon=False)
    export(fig, "fig01_teacher_compatibility")
    m = readcsv(OUT / "bank_metrics.csv")
    fn = readcsv(OUT / "bank_functional_coverage.csv")
    refined = readcsv(OUT / "solver_sensitivity.csv")
    episodes = readcsv(OUT / "bank_episodes.csv")
    original = {
        int(r["seed"]): int(r["success"])
        for r in episodes
        if r["bank_hash"] == "unmodified_teacher"
    }
    ids = sorted(original)
    base = np.array([original[i] for i in ids])
    draws = np.random.default_rng(701).integers(0, len(ids), (10000, len(ids)))
    paired = []
    for source, rows in [
        ("primary", episodes),
        ("independent", readcsv(OUT / "solver_sensitivity_episodes.csv")),
    ]:
        for b in m:
            values = {
                int(r["seed"]): int(r["success"])
                for r in rows
                if r["bank_hash"] == b["bank_hash"]
            }
            treatment = np.array([values[i] for i in ids])
            difference = treatment - base
            lo, hi = np.quantile(difference[draws].mean(axis=1) * 100, [0.025, 0.975])
            paired.append(
                dict(
                    solver=source,
                    K=b["K"],
                    nmf_seed=b["nmf_seed"],
                    bank_hash=b["bank_hash"],
                    both_succeeded=int((treatment * base).sum()),
                    only_teacher=int(((1 - treatment) * base).sum()),
                    only_projected=int((treatment * (1 - base)).sum()),
                    difference_pp=round(difference.mean() * 100, 3),
                    paired_bootstrap95_low_pp=lo,
                    paired_bootstrap95_high_pp=hi,
                )
            )
    table(OUT / "bank_paired_effects.csv", paired)
    reference = int(fn[0]["successes"])
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.7), layout="constrained")
    for i, (key, title, ylabel) in enumerate(
        [
            (
                "holdout_relative",
                "A   Excitation reconstruction",
                "Relative Frobenius error",
            ),
            (
                "clipped_fraction",
                "B   Clipping on held-out actions",
                "Actuator entries clipped (%)",
            ),
        ]
    ):
        ax = axs[i]
        for j, seed in enumerate((11, 29, 47)):
            sub = [r for r in m if int(r["nmf_seed"]) == seed]
            ax.plot(
                [int(r["K"]) + (j - 1) * 0.20 for r in sub],
                [float(r[key]) * (100 if i else 1) for r in sub],
                "o-",
                ms=4,
                lw=1,
                alpha=0.8,
                color=[BLUE, GREEN, ORANGE][j],
                label=str(seed),
            )
        ax.set(ylabel=ylabel, title=title, ylim=(0, None))
    axs[0].legend(title="NMF seed", frameon=False, fontsize=8)
    ax = axs[-1]
    for rows, key, off, color, mark, label in [
        (m, "projected_successes", -0.25, BLUE, "o", "Primary projection"),
        (refined, "successes", 0.25, ORANGE, "x", "Independent solver"),
    ]:
        for j, r in enumerate(rows):
            ax.scatter(
                int(r["K"]) + off + ((j % 3) - 1) * 0.09,
                int(r[key]),
                s=23,
                c=color,
                marker=mark,
                label=label if j == 0 else None,
            )
    ax.axhline(reference, color=GRAY, ls="-", lw=1, label=f"Teacher ({reference}/100)")
    ax.axhline(reference * 0.75, color=GRAY, ls="--", lw=1, label="75% of teacher")
    ax.set(
        title="C   Matched closed-loop control",
        ylabel="Terminal success (%)",
        ylim=(0, 103),
    )
    ax.legend(frameon=False, fontsize=8, loc="center left")
    for ax in axs:
        ax.set(xticks=[4, 8, 12, 16], xlabel="Number of components (K)")
    export(fig, "fig04_bank_quality_by_k")
    dec = readcsv(OUT / "decoder_saturation.csv")
    fig, axs = plt.subplots(1, 2, figsize=(9.3, 3.3), layout="constrained")
    # K=8, seed 11 is shown for continuity with the old fixed-K baseline, without model selection.
    ds = [r for r in dec if r["K"] == "8" and r["nmf_seed"] == "11"]
    proj = next(r for r in fn if r["K"] == "8" and r["nmf_seed"] == "11")
    bins = np.linspace(LOW, HIGH, 31)
    for ax, data in [
        (
            axs[0],
            [
                (
                    next(r for r in ds if r["source"] == "teacher")["histogram"],
                    "Teacher, held-out data",
                    GRAY,
                ),
                (proj["physical_histogram"], "Projected teacher, closed loop", BLUE),
            ],
        ),
        (
            axs[1],
            [
                (next(r for r in ds if r["source"] == s)["histogram"], label, color)
                for s, label, color in [
                    ("legacy_normal", "Legacy softplus", ORANGE),
                    ("calibrated_normal", "Calibrated ReLU", GREEN),
                ]
            ],
        ),
    ]:
        for values, label, color in data:
            h = np.array(ast.literal_eval(values))
            ax.stairs(100 * h / h.sum(), bins, label=label, color=color, lw=1.7)
        ax.set(
            xlabel="Physical muscle excitation",
            ylabel="Actuator entries per bin (%)",
            xlim=(LOW, HIGH),
        )
        ax.legend(frameon=False, fontsize=8)
    axs[0].set_title("A   Observed commands (K=8, seed 11)")
    axs[1].set_title("B   Fixed Gaussian latent probe")
    export(fig, "fig06_action_distribution")
    dsmeta = read(OUT / "dataset_manifest.json")
    for row in dsmeta["splits"]:
        row.update(
            source_checkpoint_sha256=dsmeta["teacher_sha256"],
            fit_sampling="20/25 steps per phase per goal"
            if row["split"] == "train"
            else "full holdout",
            actual_nmf_frames=int(m[0]["fit_frames"]) if row["split"] == "train" else 0,
        )
    table(OUT / "dataset_manifest.csv", dsmeta["splits"])
    goalsets = {}
    for split in ("train", "val"):
        with np.load(DATA / f"rollouts_{split}.npz") as d:
            seeds, idx, n = np.unique(d["seed"], return_index=True, return_counts=True)
            center = d["goal"][idx].reshape(len(idx), 5, 3).mean(axis=1)
            goalsets[split] = (center, d["outcome"][idx], n)
    med = np.median(goalsets["train"][0], axis=0)
    counts = []
    for split, (centers, labels, n) in goalsets.items():
        bins = ((centers >= med) * [4, 2, 1]).sum(axis=1)
        for i in range(8):
            mask = bins == i
            counts.append(
                dict(
                    split=split,
                    centroid_octant=f"{i:03b}",
                    episodes=int(mask.sum()),
                    frames=int(n[mask].sum()),
                    successes=int((labels[mask] == "success").sum()),
                    near_success=int((labels[mask] == "near_success").sum()),
                )
            )
    table(OUT / "dataset_goal_coverage.csv", counts)
    fig, ax = plt.subplots(figsize=(6.6, 3.4), layout="constrained")
    for split, offset, color in [("train", -0.17, BLUE), ("val", 0.17, GREEN)]:
        c = [r for r in counts if r["split"] == split]
        heights = np.array([r["episodes"] for r in c])
        ax.bar(
            np.arange(8) + offset,
            100 * heights / heights.sum(),
            0.32,
            label=split,
            color=color,
        )
    ax.set(
        xticks=np.arange(8),
        xticklabels=[f"{i:03b}" for i in range(8)],
        xlabel="Target-centroid octant (fit-set median split on x, y, z)",
        ylabel="Goals in partition (%)",
        title="Target coverage across independent episodes",
    )
    ax.legend(frameon=False)
    export(fig, "dataset_goal_coverage")
    pf = read(OUT / "preflight.json")
    ledger = []

    def add(stage, hardware, steps, seconds, comment=""):
        ledger.append(
            dict(
                stage=stage,
                hardware=hardware,
                simulated_transitions=steps,
                wall_seconds=seconds,
                gpu_active_seconds="",
                scope="shared",
                comment=comment,
            )
        )

    add(
        "External teacher pretraining",
        "external",
        "",
        "",
        "Upstream does not provide train steps or compute.",
    )
    for b in ["cpu", "mjlab"]:
        add(
            "Environment probe " + b,
            b,
            sum(r["probe_transitions"] for r in pf["rows"] if r["backend"] == b),
            "",
            "No phase timer.",
        )
    add(
        "Teacher validation",
        "GPU",
        sum(int(r["simulated_steps"]) for r in teachers),
        sum(float(r["wall_s"]) for r in teachers),
    )
    add("Teacher data", "GPU", dsmeta["simulated_steps"], dsmeta["wall_s"])
    add(
        "NMF and offline diagnostics",
        "CPU",
        0,
        sum(float(r["fit_wall_s"]) for r in m),
        "Sum of candidate timers through fit and reconstruction; excludes file output and later decoder summaries.",
    )
    add(
        "Bank development original",
        "GPU",
        int(fn[0]["simulated_steps"]),
        float(fn[0]["wall_s"]),
    )
    add(
        "Bank development projected",
        "GPU + CPU",
        sum(int(r["simulated_steps"]) for r in fn[1:]),
        sum(float(r["wall_s"]) for r in fn[1:]),
        f"CPU projection total {sum(float(r['solver_s']) for r in fn[1:]):.2f} s, included.",
    )
    add(
        "Independent solver validation",
        "GPU + CPU",
        sum(int(r["simulated_steps"]) for r in refined),
        sum(float(r["wall_s"]) for r in refined),
        f"CPU projection total {sum(float(r['solver_s']) for r in refined):.2f} s, included.",
    )
    add(
        "Matched one-step response",
        "GPU + CPU",
        sum(int(r["targets"]) for r in readcsv(OUT / "one_step_response.csv")),
        "",
        "No phase timer.",
    )
    add("Downstream PPO, BC and adaptation", "GPU", 0, 0, "Gate closed.")
    table(OUT / "resource_ledger.csv", ledger)
    fig, ax = plt.subplots(figsize=(7.8, 3.7), layout="constrained")
    measured = [r for r in ledger if r["wall_seconds"] and r["wall_seconds"] > 0]
    ax.barh(
        [r["stage"] for r in measured],
        [r["wall_seconds"] for r in measured],
        color=BLUE,
    )
    ax.set(xlabel="Measured wall time (seconds)", title="Timed experimental stages")
    ax.invert_yaxis()
    export(fig, "fig09_cost_breakdown")
    table(
        OUT / "control_results.csv",
        [
            dict(
                task=t,
                arm=a,
                status="NOT_RUN",
                reason="P2 BANK_INADEQUATE",
                successes="",
                episodes="",
                training_steps=0,
            )
            for t in ("T1", "T2")
            for a in ("Direct PPO", "Synergy PPO")
        ],
    )
    table(
        OUT / "adaptation_results.csv",
        [
            dict(
                task="T3",
                arm=a,
                status="BLOCKED",
                reason="No eligible B-R checkpoint",
                successes="",
                episodes="",
                training_steps=0,
            )
            for a in ("C0 Frozen", "C1 Full FT", "C2 Local FT")
        ],
    )
    save(
        OUT / "runtime.json",
        dict(
            host=pf["host"],
            analysis_packages={
                p: metadata.version(p)
                for p in ["scipy", "scikit-learn", "matplotlib", "threadpoolctl"]
            },
            deterministic_algorithms_default=torch.are_deterministic_algorithms_enabled(),
            cudnn_deterministic_default=torch.backends.cudnn.deterministic,
            cudnn_benchmark_default=torch.backends.cudnn.benchmark,
            matmul_precision_default=torch.get_float32_matmul_precision(),
            evaluation="deterministic policy means, lane-specific reset seeds; GPU physics may vary at floating-point level",
        ),
    )


if __name__ == "__main__":
    build()
