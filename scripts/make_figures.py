"""Build the V0.1 paper tables and figures from the observed run artifacts."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from myo_rsi.tables import SCHEMAS, collect, read_json, write_csv

LABELS = ["Direct PPO", "Synergy PPO", "C0 Frozen", "C1 Full FT", "C2 Local FT"]
COLORS = dict(zip(LABELS, ["#174a7c", "#cb631f", "#606060", "#2b7963", "#9b467d"]))
TITLES = {
    "T1": "Random fingertip reach",
    "T2": "Random joint pose",
    "T3": "Sarcopenic reach",
}


def panel():
    return plt.subplots(figsize=(6.2, 3.7), layout="constrained")


def save(fig, output, name):
    for kind in ("png", "pdf"):
        fig.savefig(output / f"{name}.{kind}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def percent_limits(values):
    """Zoom low success rates while keeping real zero visible."""
    top = min(102.0, max(5.0, 5.0 * np.ceil(np.max(values) * 1.12 / 5.0)))
    return -max(0.6, top * 0.025), top


def curves(frame, task, output, name):
    if frame.empty:
        return False
    fig, ax = panel()
    for method, group in frame.groupby("method", sort=False):
        pivot = group.pivot(index="steps", columns="seed", values="rate").sort_index()
        if method == "C0 Frozen":
            # Frozen checkpoints undergo no adaptation transitions.
            xmax = frame.steps.max()
            for val in pivot.iloc[0]:
                ax.plot([0, xmax], [val, val], color=COLORS[method], alpha=0.23, lw=0.8)
            ax.axhline(
                pivot.iloc[0].mean(),
                color=COLORS[method],
                ls="--",
                label=f"{method} (n={len(pivot.columns)})",
            )
        else:
            ax.plot(pivot.index, pivot.values, color=COLORS[method], alpha=0.25, lw=1)
            ax.plot(
                pivot.index,
                pivot.mean(axis=1),
                color=COLORS[method],
                lw=2,
                linestyle="--" if method in ("Synergy PPO", "C2 Local FT") else "-",
                marker="o" if method in ("Direct PPO", "C1 Full FT") else "x",
                markersize=4,
                label=f"{method} (n={len(pivot.columns)})",
            )
    ax.set(
        xlabel="Additional adaptation transitions"
        if task == "T3"
        else "Cumulative training transitions",
        ylabel="Validation success (%)",
        ylim=percent_limits(frame.rate),
        title=TITLES[task],
    )
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.grid(axis="y", alpha=0.19)
    ax.legend(frameon=False, fontsize=9)
    fig.text(
        0.12,
        -0.04,
        "Thin lines show training seeds. Each checkpoint uses 100 validation episodes.",
        fontsize=8,
    )
    save(fig, output, name)
    return True


def build(results):
    root = results.resolve().parent
    if any((root / "checkpoints").glob("*/*/*/run.json")):
        data = collect(root, results)
    else:
        # A checkout with published CSVs can regenerate the report without PPO weights.
        data = {}
        for name in SCHEMAS:
            path = (
                results / (name + ".csv")
                if name.startswith("exp_")
                else results / "tables" / (name + ".csv")
            )
            frame = pd.read_csv(path)
            data[name] = frame.to_dict("records")
    output = results / "figures"
    output.mkdir(parents=True, exist_ok=True)
    progress = pd.DataFrame(data["train_progress"])
    end = pd.DataFrame(data["exp_ab_comparison"])
    adapt = pd.DataFrame(data["exp_c_adaptation"])
    acurve = pd.DataFrame(data["adaptation_progress"])
    resource = pd.DataFrame(data["resource_cost"])
    preflight = read_json(results / "preflight.json")
    dims = {
        r["task_id"]: r.get("native_action_dim")
        for r in preflight["rows"]
        if r["backend"] == "mjlab"
    }
    summary = []
    figures = []
    for tid, name in (
        ("T1", "fig01_reach_learning_curve"),
        ("T2", "fig02_pose_learning_curve"),
    ):
        for method in LABELS[:2]:
            group = (
                end[(end.task_id == tid) & (end.method == method)]
                if not end.empty
                else end
            )
            n = len(group)
            summary.append(
                dict(
                    task_id=tid,
                    method=method,
                    policy_action_dim=dims.get(tid) if method == "Direct PPO" else 8,
                    bank_source="none"
                    if method == "Direct PPO"
                    else "T1 training only",
                    per_seed_wilson=json.dumps(
                        {
                            str(r.seed): [r.wilson_low, r.wilson_high]
                            for r in group.itertuples()
                        }
                    )
                    if n
                    else None,
                    training_seeds=n,
                    episodes_per_seed=100 if n else None,
                    mean_success=group.success_rate.mean() if n else None,
                    seed_sd=group.success_rate.std(ddof=1) if n > 1 else None,
                    final_steps_per_seed=int(group.environment_steps.iloc[0])
                    if n
                    else None,
                    mean_gpu_hours=group.gpu_hours.mean() if n else None,
                    status="RUN" if n == 3 else ("PARTIAL" if n else "NOT_RUN"),
                )
            )
        if not progress.empty:
            valid = (
                progress[(progress.task_id == tid) & progress.method.isin(LABELS[:2])]
                .dropna(subset=["eval_success_n"])
                .copy()
            )
            if not valid.empty:
                frame = valid.rename(
                    columns={"environment_steps": "steps", "training_seed": "seed"}
                )
                frame["rate"] = 100 * frame.eval_success_n / frame.eval_episodes
                curves(frame, tid, output, name)
                figures.append(name)
    write_csv(
        results / "summary.csv",
        "task_id method policy_action_dim bank_source training_seeds episodes_per_seed mean_success seed_sd per_seed_wilson final_steps_per_seed mean_gpu_hours status",
        summary,
    )
    if not end.empty:
        fig, ax = panel()
        ticks = []
        for i, row in enumerate(summary):
            group = end[(end.task_id == row["task_id"]) & (end.method == row["method"])]
            ticks.append(row["task_id"] + "\n" + row["method"].replace(" PPO", ""))
            if group.empty:
                continue
            rates = 100 * group.sort_values("seed").success_rate.to_numpy()
            ax.scatter(
                i + np.linspace(-0.10, 0.10, len(rates)),
                rates,
                color=COLORS[row["method"]],
                s=23,
            )
            ax.errorbar(
                i,
                rates.mean(),
                yerr=rates.std(ddof=1) if len(rates) > 1 else None,
                color="black",
                fmt="_",
                capsize=4,
            )
        ax.set(
            xticks=range(len(ticks)),
            xticklabels=ticks,
            ylabel="Held-out success (%)",
            ylim=percent_limits(100 * end.success_rate),
            title="Final checkpoint performance",
        )
        fig.text(
            0.12,
            -0.04,
            "Dots are training seeds, bars are mean ± sample SD. 100 episodes per seed.",
            fontsize=8,
        )
        name = "fig03_final_success"
        save(fig, output, name)
        figures.append(name)
    if data["bank_metrics"]:
        from myo_rsi.bank import load_bank

        bankpath = root / "artifacts/synergy_bank_k8.npz"
        info = read_json(root / "artifacts/nmf.json")["bank"]
        w, _ = load_bank(bankpath, info["actuator_names"])
        fig, ax = plt.subplots(figsize=(4.6, 9.1), layout="constrained")
        img = ax.imshow(w, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set(
            yticks=range(len(w)),
            yticklabels=info["actuator_names"],
            xticks=range(w.shape[1]),
            xticklabels=range(1, w.shape[1] + 1),
            xlabel="Synergy",
            ylabel="Actuator",
            title="MyoHand · MyoSuite 3.1.0",
        )
        ax.tick_params(axis="y", labelsize=7)
        fig.colorbar(img, ax=ax, label="Column-normalized weight", shrink=0.5)
        fig.text(
            0.1,
            -0.015,
            "T1 training only · SHA256 " + info["bank_hash"][:16],
            fontsize=8,
        )
        name = "fig04_bank_heatmap"
        save(fig, output, name)
        figures.append(name)
    if not acurve.empty:
        g = acurve[(acurve.eval_task == "T3") & (acurve.split == "validation")].copy()
        if not g.empty:
            g = g.rename(
                columns={
                    "adapt_method": "method",
                    "adapt_steps": "steps",
                    "training_seed": "seed",
                }
            )
            g["rate"] = 100 * g.eval_success_n / g.eval_episodes
            name = "fig05_adaptation_curve"
            curves(g, "T3", output, name)
            figures.append(name)
    if not adapt.empty:
        fig, ax = panel()
        for i, label in enumerate(LABELS[2:]):
            groups = adapt[adapt.method == label].sort_values("seed")
            for j, (_, r) in enumerate(groups.iterrows()):
                x = np.array([i - 0.16, i + 0.16]) + (j - 1) * 0.035
                ax.plot(
                    x,
                    100 * np.array([r.t1_before, r.t1_after]),
                    "o-",
                    c=COLORS[label],
                    alpha=0.7,
                    ms=4,
                )
        ax.set(
            xticks=range(3),
            xticklabels=LABELS[2:],
            ylabel="Normal T1 held-out success (%)",
            ylim=percent_limits(100 * adapt[["t1_before", "t1_after"]].to_numpy()),
            title="Reach retention before and after adaptation",
        )
        fig.text(
            0.14,
            -0.04,
            "Each line joins one matched source seed before (left) and after (right).",
            fontsize=8,
        )
        name = "fig06_retention"
        save(fig, output, name)
        figures.append(name)
        fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.7), layout="constrained")
        for i, label in enumerate(LABELS[2:]):
            charges = resource[
                (resource.method == label)
                & resource.stage.isin(["adapt_train", "frozen_setup", "evaluate"])
            ]
            for _, g in charges.groupby("seed"):
                axes[0].scatter(
                    i - 0.10, g.environment_steps.sum(), color=COLORS[label], marker="o"
                )
                axes[0].scatter(
                    i + 0.10, g.eval_steps.sum(), color=COLORS[label], marker="x"
                )
                axes[1].scatter(i, g.gpu_hours.sum(), color=COLORS[label])
        for ax in axes:
            ax.set_xticks(range(3), ["C0", "C1", "C2"])
        axes[0].set(
            ylabel="Simulation transitions", title="Training (●) and evaluation (×)"
        )
        axes[1].set(ylabel="Allocated GPU hours", title="Training and evaluation")
        name = "fig07_adaptation_cost"
        save(fig, output, name)
        figures.append(name)
    report(results, summary, end, adapt, figures, resource)
    return figures


def report(results, summary, end, adapt, figures, resource):
    pf = read_json(results / "preflight.json")
    gpu_rows = [r for r in pf["rows"] if r["backend"] == "mjlab"]

    def images(names):
        return (
            "\n\n".join(f"![{n}](figures/{n}.png)" for n in names if n in figures)
            or "NOT_RUN"
        )

    lines = [
        "# Myo-RSI v0.1 results",
        "## 1. Setup",
        f"{pf['host']['gpu']}. Python {pf['host']['python']}. Package versions are in [the runtime manifest](runtime-packages.txt).",
        f"Normal-body gate {pf['normal_gate']}, adaptation gate {pf['adaptation_gate']}. "
        + ", ".join(
            f"{r['task_id']} {r.get('obs_dim', '?')}D observation / {r.get('native_action_dim', '?')}D action"
            for r in gpu_rows
        )
        + ".",
        "Three training seeds (11, 29, 47), 256 environments × 24 steps × 256 iterations per A/B run. "
        "C1/C2 receive 64 iterations. Each evaluation uses 100 seeded episodes with deterministic mean actions. "
        "Validation seeds generate curves. The disjoint final seeds score the fixed-budget endpoints. "
        "Both synergy policies require the identical frozen bank derived from direct T1 training data.",
        "## 2. Table 1. Normal-body final success",
        "| Task | Method | Action dim | Bank source | Seeds | Mean ± SD success | Training steps/seed | Mean training GPU h | Status |\n"
        "|---|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for r in summary:
        pct = "—" if r["mean_success"] is None else f"{100 * r['mean_success']:.1f}%"
        if r["seed_sd"] is not None:
            pct += f" ± {100 * r['seed_sd']:.1f} pp"
        hours = "—" if r["mean_gpu_hours"] is None else f"{r['mean_gpu_hours']:.3f}"
        lines[-1] += (
            f"\n| {r['task_id']} | {r['method']} | {r['policy_action_dim']} | {r['bank_source']} | {r['training_seeds']} | {pct} | {r['final_steps_per_seed'] or '—'} | {hours} | {r['status']} |"
        )
    if not end.empty:
        runs = [
            "| Task | Method | Seed | Success | 95% Wilson episode CI |",
            "|---|---|---:|---:|---:|",
        ]
        for r in end.sort_values(["task_id", "method", "seed"]).itertuples():
            runs.append(
                f"| {r.task_id} | {r.method} | {r.seed} | {r.success_n}/{r.episodes} | {100 * r.wilson_low:.1f}–{100 * r.wilson_high:.1f}% |"
            )
        lines.append("\n".join(runs))
    lines += [
        "## 3. Reach and pose learning curves",
        images(["fig01_reach_learning_curve", "fig02_pose_learning_curve"]),
        "## 4. Final success and shared bank",
        images(["fig03_final_success", "fig04_bank_heatmap"]),
        "## 5. Table 2. Sarcopenic Reach adaptation",
    ]
    if adapt.empty:
        lines.append("NOT_RUN")
    else:
        rows = [
            "| Source seed | Method | T3 before → after | T1 retention | Additional transitions | Training GPU h |",
            "|---:|---|---:|---:|---:|---:|",
        ]
        for r in adapt.itertuples():
            rows.append(
                f"| {r.seed} | {r.method} | {r.t3_before:.0%} → {r.t3_after:.0%} | {r.t1_after:.0%} | {r.adaptation_steps} | {r.gpu_hours:.3f} |"
            )
        for label, g in adapt.groupby("method"):
            rows.append(
                f"| mean ± SD | {label} | {g.t3_before.mean():.1%} → {g.t3_after.mean():.1%} ± {g.t3_after.std():.1%} | {g.t1_after.mean():.1%} ± {g.t1_after.std():.1%} | — | {g.gpu_hours.mean():.3f} |"
            )
        lines.append("\n".join(rows))
    lines.extend(
        [
            images(["fig05_adaptation_curve"]),
            "## 6. Retention and adaptation cost",
            images(["fig06_retention", "fig07_adaptation_cost"]),
        ]
    )
    if not resource.empty:
        upstream = resource[
            (resource.task_id == "T1")
            & (resource.method == "Direct PPO")
            & (resource.stage == "train")
        ]
        lines.append(
            f"The direct Reach teacher accounts for {upstream.environment_steps.sum():,} training transitions across completed seeds. "
            "Excitations were sampled during those rollouts. The ledger charges each source run once. "
            "GPU hours are single-device wall allocations. CPU NMF time is recorded separately."
        )
    lines += [
        "## 7. Interpretation and limitations",
        "Seed SD measures variation across independently trained policies. Wilson intervals describe episode uncertainty within one policy. "
        "The small number of training seeds limits precision. See [implementation and numerical limitations](limitations.md).",
    ]
    observations = []
    for task in ("T1", "T2"):
        rows = {
            r["method"]: r
            for r in summary
            if r["task_id"] == task and r["status"] == "RUN"
        }
        if len(rows) != 2:
            continue
        direct = rows["Direct PPO"]["mean_success"]
        synergy = rows["Synergy PPO"]["mean_success"]
        observations.append(
            f"{TITLES[task]} reached {direct:.1%} with direct control and {synergy:.1%} "
            f"with fixed-bank control at matched downstream budgets "
            f"(synergy minus direct {(synergy - direct) * 100:+.1f} percentage points)."
        )
    bank = results / "tables/bank_metrics.csv"
    if bank.exists():
        bm = pd.read_csv(bank)
        if len(bm) == 1:
            v = bm.iloc[0]
            observations.append(
                f"The bank used {int(v.num_actions_fit):,} T1 fitting samples. "
                f"The training-episode holdout relative Frobenius error was {v.nmf_val_recon:.3f}, "
                f"with {v.excitation_clip_fraction:.1%} of reconstructed channel values outside the reachable interval."
            )
    if not adapt.empty and all(
        len(adapt[adapt.method == label]) == 3 for label in LABELS[2:]
    ):
        full = adapt[adapt.method == "C1 Full FT"]
        local = adapt[adapt.method == "C2 Local FT"]
        observations.append(
            f"Full and local fine-tuning reached {full.t3_after.mean():.1%} and "
            f"{local.t3_after.mean():.1%} mean T3 success. Their original-body retention was "
            f"{full.t1_after.mean():.1%} and {local.t1_after.mean():.1%} respectively. Each received "
            f"{int(local.adaptation_steps.iloc[0]):,} additional training transitions per seed."
        )
        ck = pd.read_csv(results / "tables/checkpoint_manifest.csv")
        full_count = ck.loc[ck.method == "C1 Full FT", "trainable_actor_params"].iloc[0]
        local_count = ck.loc[ck.method == "C2 Local FT", "trainable_actor_params"].iloc[
            0
        ]
        observations.append(
            f"The local update trained {local_count:,} actor parameters versus {full_count:,} "
            "for full fine-tuning. Both conditions also updated a critic."
        )
    if observations:
        lines.append("\n".join("- " + item for item in observations))
    (results / "RESULTS_REPORT.md").write_text(
        "\n\n".join(lines) + "\n", encoding="utf8"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    args = p.parse_args()
    print(build(args.results_dir))
