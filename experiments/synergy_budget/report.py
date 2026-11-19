"""Generate scientific figures, comparison tables and an auditable report from observed runs."""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from myo_rsi.tables import wilson
from .controller_gap import compare
from .common import CKPT, LOCK, OLD, OUT, atomic_table, bank, read, readcsv, seeds, sha

COL = {39: "#304b67", 8: "#be6238", 16: "#167666", 24: "#75508e", 32: "#367da8"}
LABEL = {
    39: "Direct 39D",
    8: "Synergy K=8",
    16: "Synergy K=16",
    24: "Synergy K=24",
    32: "Synergy K=32",
}
TASK = {"T1": "Reach Random", "T2": "Pose Random"}


def output_fig(fig, figs, name):
    for ext in ("png", "svg"):
        fig.savefig(figs / f"{name}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plain(rows, cols):
    headings = [heading for key, heading in cols]
    body = [
        "| " + " | ".join(headings) + " |",
        "|" + "|".join("---" for _ in cols) + "|",
    ]
    for r in rows:
        body.append("| " + " | ".join(str(r[key]) for key, _ in cols) + " |")
    return "\n".join(body)


def representation(figs):
    tab = []
    for cohort in ("projection_dev", "test"):
        folder = OUT / f"capacity_{cohort}"
        if not (folder / "complete.json").exists():
            continue
        rows = readcsv(folder / "summary.csv")
        episodes = readcsv(folder / "episodes.csv")
        teacher = {int(r["seed"]): r for r in episodes if r["controller"] == "Teacher"}
        for r in rows:
            sub = [
                e
                for e in episodes
                if (e["controller"], e["K"], e["nmf_seed"])
                == (r["controller"], r["K"], r["nmf_seed"])
            ]
            if len(sub) != 100 or any(
                (e["goal_sha256"], e["reset_obs_sha256"])
                != (
                    teacher[int(e["seed"])]["goal_sha256"],
                    teacher[int(e["seed"])]["reset_obs_sha256"],
                )
                for e in sub
            ):
                raise ValueError("Projection targets differ")
            original = sum(int(e["success"]) for e in teacher.values())
            both = sum(
                int(e["success"]) * int(teacher[int(e["seed"])]["success"]) for e in sub
            )
            rec = dict(
                cohort=cohort,
                **r,
                teacher_successes=original,
                also_teacher_success=both,
                regained=int(r["successes"]) - both,
                fraction_teacher_goals_preserved=both / original,
                relative_success=int(r["successes"]) / original,
            )
            tab.append(rec)
    if not tab:
        return []
    atomic_table(OUT / "functional_coverage.csv", tab)
    dev = [r for r in tab if r["cohort"] == "projection_dev"]
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6), layout="constrained")
    for j, s in enumerate(LOCK["nmf_seeds"]):
        sub = [r for r in dev if r["controller"] == "NMF" and int(r["nmf_seed"]) == s]
        ax[0].plot(
            [int(r["K"]) for r in sub],
            [float(r["heldout_relative"]) for r in sub],
            "o-",
            lw=1.2,
            label=f"NMF seed {s}",
        )
        ax[1].plot(
            [int(r["K"]) + (j - 1) * 0.2 for r in sub],
            [int(r["successes"]) for r in sub],
            "o-",
            lw=1.2,
        )
    teacher = next(int(r["successes"]) for r in dev if r["controller"] == "Teacher")
    ident = next(int(r["successes"]) for r in dev if r["controller"] == "Identity")
    ax[1].axhline(teacher, color="#333333", ls="--", label="Original teacher")
    ax[1].scatter([39], [ident], marker="D", color="#333333", label="Identity")
    for a, title, y in zip(
        ax,
        ["Held-out excitation fidelity", "Closed-loop functional coverage"],
        ["Relative Frobenius error", "Successful targets / 100"],
    ):
        a.set(xlabel="Action dimension", ylabel=y, title=title)
        a.legend(fontsize=8, frameon=False)
    output_fig(fig, figs, "functional_coverage")
    fig, axes = plt.subplots(
        1, 4, figsize=(12.2, 7.8), sharey=True, layout="constrained"
    )
    for a, k in zip(axes, LOCK["k"]):
        w, _, names, _ = bank(k, LOCK["policy_bank_seed"])
        im = a.imshow(
            w, vmin=0, vmax=1, cmap="viridis", aspect="auto", interpolation="nearest"
        )
        a.set(title=f"K={k}", xlabel="Component", yticks=np.arange(len(names)))
        a.set_yticklabels(names, fontsize=7)
    fig.colorbar(im, ax=axes, label="Normalized recruitment weight", shrink=0.65)
    output_fig(fig, figs, "bank_heatmap")
    df = readcsv(OUT / "capacity_projection_dev" / "trajectories.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), layout="constrained")
    for name, k, s in [("Teacher repeat", 0, 0), ("Identity", 39, 0)] + [
        (LABEL[k], k, LOCK["policy_bank_seed"]) for k in LOCK["k"]
    ]:
        sub = [
            r
            for r in df
            if int(r["K"]) == k
            and int(r["nmf_seed"]) == s
            and (k or r["controller"] == "TeacherRepeat")
        ]
        for a, col in zip(axes, ("excitation_rmse", "joint_rmse_rad")):
            a.plot(
                [int(r["timestep"]) * 0.02 for r in sub],
                [float(r[col]) for r in sub],
                label=name,
                lw=1.25,
            )
    for a, title in zip(
        axes, ["Excitation RMS difference", "Joint RMS difference (rad)"]
    ):
        a.set(title=title, xlabel="Time from reset (s)")
    axes[-1].legend(fontsize=7, frameon=False, ncols=2)
    output_fig(fig, figs, "paired_trajectories")
    return dev


def collect(strict):
    records = []
    finals = []
    costs = []
    patterns = {}
    cfgs = {}
    for task in LOCK["tasks"]:
        decision = OUT / f"{task}_budget.json"
        budget = (
            read(decision)
            if decision.exists()
            else dict(arms=LOCK["primary_actions"], iterations=1000)
        )
        if strict and not (OUT / f"{task}_complete.json").exists():
            raise RuntimeError(f"Missing mandatory {task} matrix")
        for k in budget["arms"]:
            for seed in LOCK["training_seeds"]:
                case = f"{task}-K{k}-s{seed}"
                cell = OUT / "cells" / case
                hist = cell / "points.json"
                if not hist.exists():
                    continue
                pts = read(hist)
                cfg = read(cell / "config.json")
                policy_cfg = dict(cfg["ppo"])
                policy_cfg.pop("seed")
                if task in cfgs and cfgs[task] != policy_cfg:
                    raise ValueError("PPO configurations differ across arms")
                cfgs[task] = policy_cfg
                if (
                    k != 39
                    and cfg["bank_hash"]
                    != bank(k, LOCK["policy_bank_seed"])[3]["bank_hash"]
                ):
                    raise ValueError("Unexpected bank")
                costs.append(
                    dict(
                        stage="PPO training",
                        task=task,
                        K=k,
                        seed=seed,
                        shared=False,
                        transitions=pts[-1]["transitions"],
                        wall_s=pts[-1]["train_wall_s"],
                        source="local",
                        note="optimizer and rollout wall time through checkpoint",
                    )
                )
                expected = [
                    p
                    for p in LOCK["iterations"] + LOCK["extension"]
                    if p <= budget["iterations"]
                ]
                if strict and [p["iteration"] for p in pts] != expected:
                    raise ValueError("Unmatched checkpoint budgets")
                for p in pts:
                    it = p["iteration"]
                    if (
                        strict
                        and sha((CKPT / case / f"policy_{it}.pt").read_bytes())
                        != p["sha256"]
                    ):
                        raise ValueError("Checkpoint integrity failure")
                    for group in (
                        ["validation", "test"]
                        if it == budget["iterations"]
                        else ["validation"]
                    ):
                        f = cell / f"{group}_{it}.json"
                        if not f.exists():
                            if strict:
                                raise ValueError(f"Missing evaluation {f}")
                            continue
                        rec = read(f)
                        ep = readcsv(cell / f"{group}_{it}_episodes.csv")
                        if (
                            rec["checkpoint_sha256"] != p["sha256"]
                            or len(ep) != 100
                            or [int(e["seed"]) for e in ep] != seeds(group)
                        ):
                            raise ValueError("Invalid evaluation provenance")
                        tags = [(e["goal_sha256"], e["reset_obs_sha256"]) for e in ep]
                        if (task, group) in patterns and patterns[task, group] != tags:
                            raise ValueError("Goals or initial observations unpaired")
                        patterns[task, group] = tags
                        if sum(int(e["success"]) for e in ep) != rec["successes"]:
                            raise ValueError("Endpoint aggregate mismatch")
                        lo, hi = wilson(rec["successes"], 100)
                        records.append(
                            dict(
                                **rec,
                                train_wall_s=p["train_wall_s"],
                                wilson95_low=lo,
                                wilson95_high=hi,
                            )
                        )
                        costs.append(
                            dict(
                                stage=f"PPO {group}",
                                task=task,
                                K=k,
                                seed=seed,
                                shared=False,
                                transitions=rec["eval_steps"],
                                wall_s=rec["eval_wall_s"],
                                source="local",
                                note=f"checkpoint {it}",
                            )
                        )
                        if group == "test":
                            finals.extend(
                                dict(
                                    task=task,
                                    K=k,
                                    train_seed=seed,
                                    training_steps=p["transitions"],
                                    **{a: b for a, b in e.items() if a != "task"},
                                )
                                for e in ep
                            )
    for task in LOCK["tasks"]:
        if (task, "test") in patterns and set(
            v[0] for v in patterns[task, "test"]
        ) & set(v[0] for v in patterns[task, "validation"]):
            raise ValueError("Validation/test goal leakage")
    return records, finals, costs


def plots_and_stats(rows, finals, figs):
    if not rows:
        return [], [], []
    atomic_table(OUT / "learning_curves.csv", rows)
    if finals:
        atomic_table(OUT / "test_episodes.csv", finals)
    metrics = []
    comparisons = []
    efficiency = []
    for task in LOCK["tasks"]:
        part = [r for r in rows if r["task"] == task]
        if not part:
            continue
        ks = sorted({r["K"] for r in part}, key=lambda k: 0 if k == 39 else k)
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
        final_fig, fa = plt.subplots(figsize=(6, 3.4), layout="constrained")
        for pos, k in enumerate(ks):
            sub = [r for r in part if r["K"] == k and r["group"] == "validation"]
            points = sorted({r["training_steps"] for r in sub})
            for ax, col in zip(axes, ["successes", "mean_terminal_error"]):
                means = []
                sd = []
                for x in points:
                    vals = [r[col] for r in sub if r["training_steps"] == x]
                    means.append(np.mean(vals))
                    sd.append(np.std(vals, ddof=1) if len(vals) > 1 else 0)
                xx = np.array(points) / 1e6
                ax.plot(xx, means, ".-", label=LABEL[k], color=COL[k])
                ax.fill_between(
                    xx,
                    np.array(means) - sd,
                    np.array(means) + sd,
                    color=COL[k],
                    alpha=0.13,
                )
            for s in LOCK["training_seeds"]:
                sequence = sorted(
                    [r for r in sub if r["seed"] == s],
                    key=lambda r: r["training_steps"],
                )
                if len(sequence) < 2:
                    continue
                for horizon in sorted({12000000, sequence[-1]["training_steps"]}):
                    cut = [r for r in sequence if r["training_steps"] <= horizon]
                    if cut[-1]["training_steps"] != horizon or not horizon:
                        continue
                    x = np.array([r["training_steps"] for r in cut])
                    y = np.array([r["successes"] for r in cut])
                    efficiency.append(
                        dict(
                            task=task,
                            K=k,
                            seed=s,
                            horizon=horizon,
                            normalized_auc_pct=float(np.trapezoid(y, x) / horizon),
                            first_25pct=next(
                                (
                                    r["training_steps"]
                                    for r in cut
                                    if r["successes"] >= 25
                                ),
                                None,
                            ),
                            first_50pct=next(
                                (
                                    r["training_steps"]
                                    for r in cut
                                    if r["successes"] >= 50
                                ),
                                None,
                            ),
                        )
                    )
            final = [r for r in part if r["K"] == k and r["group"] == "test"]
            if not final:
                continue
            ys = [r["successes"] for r in final]
            metrics.append(
                dict(
                    task=task,
                    K=k,
                    seeds=len(final),
                    per_seed="/".join(
                        str(r["successes"])
                        for r in sorted(final, key=lambda x: x["seed"])
                    ),
                    final_mean_pct=np.mean(ys),
                    final_sd_pp=float(np.std(ys, ddof=1)) if len(ys) > 1 else None,
                    mean_error=np.mean([r["mean_terminal_error"] for r in final]),
                    training_steps_per_seed=final[0]["training_steps"],
                    mean_train_wall_s=np.mean([r["train_wall_s"] for r in final]),
                )
            )
            fa.scatter(
                np.full(len(ys), pos) + np.linspace(-0.06, 0.06, len(ys)),
                ys,
                color=COL[k],
                s=20,
            )
            fa.plot([pos - 0.16, pos + 0.16], [np.mean(ys)] * 2, color=COL[k], lw=2)
            if k != 39 and len(final) == 3:
                lookup = {
                    (r["K"], r["train_seed"], int(r["seed"])): int(r["success"])
                    for r in finals
                    if r["task"] == task
                }
                if not all(
                    (39, s, t) in lookup
                    for s in LOCK["training_seeds"]
                    for t in seeds("test")
                ):
                    continue
                delta = np.array(
                    [
                        [lookup[k, s, t] - lookup[39, s, t] for t in seeds("test")]
                        for s in LOCK["training_seeds"]
                    ]
                )
                rng = np.random.default_rng(73011)
                ss = rng.integers(3, size=(10000, 3, 1))
                tt = rng.integers(100, size=(10000, 1, 100))
                dif = 100 * delta[ss, tt].mean(axis=(1, 2))
                lo, hi = np.quantile(dif, [0.025, 0.975])
                lower = float(np.quantile(dif, 0.05))
                comparisons.append(
                    dict(
                        task=task,
                        K=k,
                        delta_pp=100 * delta.mean(),
                        discordant_runs_targets=int(np.count_nonzero(delta)),
                        crossed_ci95_low=lo if np.any(delta) else None,
                        crossed_ci95_high=hi if np.any(delta) else None,
                        lower_one_sided_95=lower if np.any(delta) else None,
                        within_5pp_exploratory=bool(
                            lower >= -LOCK["near_final_margin_pp"]
                        )
                        if np.any(delta)
                        else None,
                        three_seed_deltas="/".join(map(str, delta.sum(axis=1))),
                    )
                )
        for ax, col in zip(
            axes,
            [
                "Terminal success (%)",
                "Mean terminal error (m)"
                if task == "T1"
                else "Mean terminal error (rad)",
            ],
        ):
            ax.set(
                xlabel="Training transitions (millions)", ylabel=col, title=TASK[task]
            )
            ax.legend(fontsize=8, frameon=False)
        output_fig(fig, figs, f"{task}_learning_curves")
        if any(r["task"] == task for r in metrics):
            fa.set(
                xticks=range(len(ks)),
                xticklabels=[LABEL[k] for k in ks],
                ylabel="Independent test success (%)",
                ylim=(-3, 103),
                title=TASK[task],
            )
            output_fig(final_fig, figs, f"{task}_final_success")
        else:
            plt.close(final_fig)
    if metrics:
        atomic_table(OUT / "final_performance.csv", metrics)
    if comparisons:
        atomic_table(OUT / "paired_final_differences.csv", comparisons)
    if efficiency:
        atomic_table(OUT / "sample_efficiency.csv", efficiency)
    return metrics, comparisons, efficiency


def ledger(costs):
    def add(stage, task, k, s, n, t, src, note):
        costs.append(
            dict(
                stage=stage,
                task=task,
                K=k,
                seed=s,
                shared=True,
                transitions=n,
                wall_s=t,
                source=src,
                note=note,
            )
        )

    for history in sorted((OUT / "restarted").glob("*/*/points.json")):
        config = read(history.parent / "config.json")
        last = read(history)[-1]
        add(
            "Excluded interrupted training",
            config["task"],
            config["K"],
            config["seed"],
            last["transitions"],
            last["train_wall_s"],
            "local",
            "Lower-bound cost; excluded from core PPO budget.",
        )
        for evaluation in history.parent.glob("validation_*.json"):
            record = read(evaluation)
            add(
                "Excluded checkpoint evaluation",
                config["task"],
                config["K"],
                config["seed"],
                record["eval_steps"],
                record["eval_wall_s"],
                "local",
                "Archived and excluded from outcomes.",
            )
    ds = read(OLD / "dataset_manifest.json")
    add(
        "Teacher demonstration collection",
        "T1",
        "",
        "",
        ds["simulated_steps"],
        ds["wall_s"],
        "v0.1r reused",
        "Shared across NMF banks; zero recollection in this experiment",
    )
    add(
        "External teacher pretraining",
        "T1",
        "",
        "",
        "",
        "",
        "upstream unknown",
        "External compute not reported",
    )
    for k in LOCK["k"]:
        for s in LOCK["nmf_seeds"]:
            m = bank(k, s)[3]
            add(
                "NMF fit and offline metrics",
                "T1",
                k,
                s,
                0,
                m["fit_wall_s"],
                "v0.1r reused" if k in (8, 16) else "local",
                "CPU timer coverage can differ across studies",
            )
    for folder in (
        "identity",
        "capacity_projection_dev",
        "capacity_test",
        "pose_projection_test",
    ):
        file = OUT / folder / "summary.csv"
        if file.exists():
            for r in readcsv(file):
                add(
                    folder + " " + r["controller"],
                    "T2" if folder == "pose_projection_test" else "T1",
                    r["K"],
                    r["nmf_seed"],
                    r["simulated_steps"],
                    r["wall_s"],
                    "local",
                    "Closed-loop costs include projection solver",
                )
    for r in (
        readcsv(OUT / "divergence_onset.csv")
        if (OUT / "divergence_onset.csv").exists()
        else []
    ):
        add(
            "Divergence localization",
            "T1",
            r["K"],
            11,
            r["simulated_steps"],
            r["wall_s"],
            "local",
            "Additional paired trajectories",
        )
    if (OUT / "identity_holdout.json").exists():
        add(
            "Offline identity test",
            "T1",
            39,
            "",
            0,
            read(OUT / "identity_holdout.json")["wall_s"],
            "local",
            "20,000 held-out real demonstration frames",
        )
    add("Behavioral cloning", "", "", "", 0, 0, "not run", "From-scratch primary study")
    atomic_table(OUT / "resource_ledger.csv", costs)


def late_trends(rows):
    trends = []
    for task in LOCK["tasks"]:
        for k in (39, 8, 16, 24, 32):
            group = [
                r
                for r in rows
                if r["task"] == task and r["K"] == k and r["group"] == "validation"
            ]
            points = sorted(
                {r["training_steps"] for r in group if r["training_steps"] >= 6000000}
            )
            full = [[r for r in group if r["training_steps"] == p] for p in points]
            full = [g for g in full if len(g) == len(LOCK["training_seeds"])]
            if len(full) < 3:
                continue
            groups = full[-3:]
            score = np.array([np.mean([r["successes"] for r in g]) for g in groups])
            error = np.array(
                [np.mean([r["mean_terminal_error"] for r in g]) for g in groups]
            )
            delta = np.diff(score)
            reduction = 1 - error[1:] / error[:-1]
            if np.all(abs(delta) <= 3) and np.all(abs(reduction) <= 0.05):
                pattern = "locally flat at measured checkpoints"
            elif np.all(delta > 3) or np.all(reduction > 0.05):
                pattern = "continued improvement"
            else:
                pattern = "mixed or variable"
            trends.append(
                dict(
                    task=task,
                    K=k,
                    last_three_steps="/".join(
                        str(g[0]["training_steps"]) for g in groups
                    ),
                    previous_gain_pp=float(delta[0]),
                    last_gain_pp=float(delta[1]),
                    previous_fractional_error_reduction=float(reduction[0]),
                    last_fractional_error_reduction=float(reduction[1]),
                    descriptive_pattern=pattern,
                )
            )
    if trends:
        atomic_table(OUT / "late_learning_trends.csv", trends)
    return trends


def report(strict=True):
    figs = OUT / "figures"
    figs.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "axes.titlelocation": "left",
        }
    )
    dev = representation(figs)
    rows, finals, costs = collect(strict)
    metrics, comparisons, eff = plots_and_stats(rows, finals, figs)
    gap = compare(finals, figs)
    if dev:
        ledger(costs)
    cap = OUT / "capacity_test" / "complete.json"
    if strict and (
        not (OUT / "pose_projection_test" / "complete.json").exists()
        or not cap.exists()
        or not (OUT / "divergence_onset.csv").exists()
    ):
        raise RuntimeError("Diagnostics incomplete")
    title = "Complete" if strict else "Interim"
    md = [
        f"# {title} results · Synergy representation and matched-budget PPO",
        "## Experiment",
        "The unit of training comparison is a policy seed. The deterministic evaluation unit is a complete target episode. Direct PPO controls 39 native muscle actions. Synergy PPO samples a K-dimensional Gaussian action and decodes train-calibrated nonnegative recruitment through an immutable Reach NMF bank. Each actor and critic starts from scratch. The Pose actors are initialized separately and reuse the same frozen Reach weights.",
        "[Configuration and decision rules](../../experiments/synergy_budget/protocol.json) · [Registered environment settings and source manifests](config) · [Original teacher provenance](../v0_1r/teacher_manifest.json)",
        "All arms within a task use 500 environments, 24 transitions per rollout, the task-native reward and observation, the same RSL-RL PPO hyperparameters, training seeds 11/29/47 and equal checkpoint spacing. Development targets for projection were 25000–25099, curve-validation targets 61000–61099 and fresh final targets 71000–71099. Any common extension reinitializes the simulator for all arms and restores model, normalizer, optimizer and learning rate. Full per-cell configurations, checkpoint hashes and per-episode records are in [cells](cells).",
        "## Projection and representation",
    ]
    if dev:
        md += [
            "On the paired development targets the original teacher succeeded on "
            + str(dev[0]["successes"])
            + "/100. The identity control traversed the same numerical projection, conversion and simulator path. The teacher-repeat condition measures baseline trajectory variation. The following triplets follow NMF seeds 11, 29 and 47.",
            plain(
                [
                    dict(
                        K=k,
                        relative="/".join(
                            f"{float(r['heldout_relative']):.3f}"
                            for r in dev
                            if r["K"] == str(k) and r["controller"] == "NMF"
                        ),
                        successes="/".join(
                            r["successes"]
                            for r in dev
                            if r["K"] == str(k) and r["controller"] == "NMF"
                        ),
                    )
                    for k in LOCK["k"]
                ],
                [
                    ("K", "K"),
                    ("relative", "Held-out relative error"),
                    ("successes", "Closed-loop successes / 100"),
                ],
            ),
            "![Reconstruction and projection success](figures/functional_coverage.png)",
            "[Paired functional coverage](functional_coverage.csv) · [Bank heatmaps](figures/bank_heatmap.png) · [Joint and excitation trajectories](figures/paired_trajectories.png)",
        ]
    if (OUT / "divergence_onset.csv").exists():
        onset = readcsv(OUT / "divergence_onset.csv")
        md += [
            "For bank seed 11, the following onset is the first of three consecutive 20 ms steps at which joint RMS deviation reached 0.02 rad. The named joint had the largest deviation in the greatest number of diverged episodes at their own onset.",
            plain(
                [r for r in onset if r["K"] != "0"],
                [
                    ("K", "K"),
                    ("diverged", "Diverged / 100"),
                    ("median_first_joint_step", "Median first step"),
                    ("most_frequent_joint", "Most frequent joint"),
                ],
            ),
            "[Per-target first muscle and joint deviations](divergence_targets.csv) · [Offline identity verification](identity_holdout.json)",
        ]
    md += ["## Learned control on held-out targets"]
    for task in LOCK["tasks"]:
        group = [r for r in metrics if r["task"] == task]
        if not group:
            continue
        md += [
            f"### {TASK[task]}",
            plain(
                [
                    dict(
                        arm=LABEL[r["K"]],
                        seed=r["per_seed"],
                        mean=f"{r['final_mean_pct']:.1f} ± {r['final_sd_pp']:.1f}"
                        if r["final_sd_pp"] is not None
                        else str(r["final_mean_pct"]),
                        err=f"{r['mean_error']:.4f}",
                        budget=f"{r['training_steps_per_seed'] / 1e6:g}M",
                    )
                    for r in group
                ],
                [
                    ("arm", "Policy"),
                    ("seed", "Successes / 100 for seeds 11, 29, 47"),
                    ("mean", "Mean % ± seed SD (pp)"),
                    (
                        "err",
                        "Terminal error (m)"
                        if task == "T1"
                        else "Terminal error (rad)",
                    ),
                    ("budget", "Transitions per seed"),
                ],
            ),
            f"![{TASK[task]} validation learning curves](figures/{task}_learning_curves.png)",
            f"[Final individual-seed comparison](figures/{task}_final_success.png)",
        ]
        for r in comparisons:
            if r["task"] == task:
                if r["discordant_runs_targets"] == 0:
                    md.append(
                        f"{LABEL[r['K']]} and Direct PPO had identical binary outcomes on every paired target and training seed. The bootstrap is degenerate. Continuous error and per-policy Wilson intervals are reported separately."
                    )
                    continue
                md.append(
                    f"{LABEL[r['K']]} minus Direct PPO was {r['delta_pp']:+.1f} pp on the independent test (exploratory crossed seed/target bootstrap 95% interval {r['crossed_ci95_low']:+.1f} to {r['crossed_ci95_high']:+.1f} pp)."
                )
    if gap:
        md.append("## Training specifically for the action space")
        for task in LOCK["tasks"]:
            subset = [g for g in gap if g["task"] == task]
            if not subset:
                continue
            md.append(
                f"![{TASK[task]} trained and projected controllers](figures/{task}_trained_vs_projected.png)"
            )
            for k in sorted({g["K"] for g in subset}):
                g = [r for r in subset if r["K"] == k]
                md.append(
                    f"{TASK[task]}, {LABEL[k]} achieved {np.mean([r['trained_successes'] for r in g]):.1f}% mean trained success against {g[0]['teacher_successes']}% for the corresponding teacher controller on the same targets."
                )
        md.append(
            "[Target-level gains and losses per training seed](trained_vs_projected.csv) · [Pose teacher reference](pose_projection_test/summary.csv)"
        )
    md += ["## Sample efficiency and stability"]
    if eff:
        tab = []
        for task in LOCK["tasks"]:
            for k in [39, 8, 16, 24, 32]:
                sub = [
                    r
                    for r in eff
                    if r["task"] == task and r["K"] == k and r["horizon"] == 12000000
                ]
                final_auc = [
                    r["normalized_auc_pct"]
                    for r in eff
                    if r["task"] == task and r["K"] == k and r["horizon"] == 18000000
                ]
                if sub:
                    tab.append(
                        dict(
                            task=TASK[task],
                            arm=LABEL[k],
                            auc=f"{np.mean([r['normalized_auc_pct'] for r in sub]):.1f}",
                            auc18=f"{np.mean(final_auc):.1f}"
                            if len(final_auc) == len(LOCK["training_seeds"])
                            else "—",
                            reach25=f"{sum(r['first_25pct'] is not None for r in sub)}/{len(sub)}",
                            reach50=f"{sum(r['first_50pct'] is not None for r in sub)}/{len(sub)}",
                        )
                    )
        if tab:
            md.append(
                plain(
                    tab,
                    [
                        ("task", "Task"),
                        ("arm", "Policy"),
                        ("auc", "Mean success AUC through 12M (%)"),
                        ("auc18", "Mean success AUC through 18M (%)"),
                        ("reach25", "Seeds reaching 25% by 12M"),
                        ("reach50", "Seeds reaching 50% by 12M"),
                    ],
                )
            )
        md.append(
            "AUC is the trapezoidal integral of deterministic validation success divided by the interaction horizon. Threshold times are the first observed checkpoints reaching each rate. A blank threshold is right-censored. Curve bands represent one standard deviation across training seeds. [Per-seed efficiency](sample_efficiency.csv) · [Learning curves and endpoint Wilson intervals](learning_curves.csv)"
        )
    md += ["## Budget decision and costs"]
    for task in LOCK["tasks"]:
        if (OUT / f"{task}_budget.json").exists():
            b = read(OUT / f"{task}_budget.json")
            trend = ", ".join(
                f"K={d['K']} {d['gain_pp']:+.1f} pp" for d in b["decisions"]
            )
            md.append(
                f"{TASK[task]} used {b['iterations'] * 12000:,} interactions per arm per seed. Validation success changes from 6M to 12M were {trend}. The finite resource ceiling leaves convergence unestablished."
            )
    trends = late_trends(rows)
    if trends:
        md.append(
            "Descriptive changes over the final two observed intervals are listed below. Continued improvement or variable behavior leaves a longer matched run warranted to assess asymptotic performance. Flat local behavior alone is insufficient to establish convergence. [Late learning trends](late_learning_trends.csv)"
        )
        md.append(
            plain(
                trends,
                [
                    ("task", "Task"),
                    ("K", "Action dimension"),
                    ("last_three_steps", "Last three checkpoints"),
                    ("descriptive_pattern", "Observed pattern"),
                ],
            )
        )
    if (OUT / "restarted").exists():
        md.append(
            "A host session interruption caused an extra simulator reset in an initial Pose Direct run. That run and its validation were archived and excluded. Its replacement started from zero at the same seed. The ledger includes measured recovery overhead as a lower bound."
        )
    if costs:
        train = [r for r in costs if r["stage"] == "PPO training"]
        md.append(
            f"Recorded PPO training consumed {sum(int(r['transitions']) for r in train):,} transitions and {sum(float(r['wall_s']) for r in train) / 3600:.2f} aggregate measured training wall hours. Across all locally measured training, evaluation, projection and excluded recovery rows, the ledger records a lower bound of {sum(int(r["transitions"] or 0) for r in costs if r["source"] == "local"):,} new transitions. Demonstrations contributed 100,000 separate historical teacher transitions. The ledger separates new and reused NMF fits, evaluations, projection diagnostics and the zero BC budget. The first Reach run shared the GPU briefly with capacity diagnostics and early checkpoint evaluation. The first Pose run overlapped the independent Reach projection test. Host wall time includes this contention. Device-active time and external teacher pretraining are unknown. [Resource ledger](resource_ledger.csv)"
        )
    md += ["## Interpretation"]
    if strict:
        for task in LOCK["tasks"]:
            for k in (8, 16):
                pair = next(r for r in comparisons if r["task"] == task and r["K"] == k)
                entries = {
                    s: [
                        r["normalized_auc_pct"]
                        for r in eff
                        if r["task"] == task
                        and r["K"] == s
                        and r["horizon"] == 12000000
                    ]
                    for s in (39, k)
                }
                gain = np.mean(entries[k]) - np.mean(entries[39])
                if pair["discordant_runs_targets"] == 0:
                    md.append(
                        f"{TASK[task]}, K={k}: mean validation AUC difference through 12M was {gain:+.1f} points. Identical final binary outcomes preclude an informative paired bootstrap interval or a near-direct inference."
                    )
                    continue
                verdict = "meets" if pair["within_5pp_exploratory"] else "does not meet"
                md.append(
                    f"{TASK[task]}, K={k}: the mean validation AUC difference through 12M was {gain:+.1f} points. At the common final budget the one-sided lower 95% bound for the paired success difference was {pair['lower_one_sided_95']:+.1f} pp, which {verdict} the exploratory −5 pp near-direct criterion."
                )
    md += [
        "Projected-teacher control minimizes immediate excitation error on states reached by that controller. PPO learns feedback specifically in its own action space. Their gap can indicate controller adaptation or optimization effects. These experiments do not identify the optimum possible in a bank. A frozen Reach bank contains external teacher information; the scratch comparison measures downstream interaction efficiency conditional on that prior. Three seeds limit claims about superiority and equivalence. Target sharing induces dependence across policies; paired estimates use crossed resampling of seed and target. The study contains no behavior cloning or agent-driven adaptation."
    ]
    if not strict:
        md.append(
            "Only currently observed measurements are included. Remaining cells and final decisions are pending."
        )
    if strict:
        from .common import save

        expert_goals = {r["goal_sha256"] for r in readcsv(OLD / "dataset_episodes.csv")}
        final_goals = {r["goal_sha256"] for r in finals if r["task"] == "T1"}
        if expert_goals & final_goals:
            raise ValueError(
                "Expert training targets overlap the new final Reach cohort"
            )
        checked = 0
        for task in LOCK["tasks"]:
            budget = read(OUT / f"{task}_budget.json")
            for k in budget["arms"]:
                for seed in LOCK["training_seeds"]:
                    pts = read(OUT / "cells" / f"{task}-K{k}-s{seed}" / "points.json")
                    if any(
                        p["segment_start_iteration"]
                        != (0 if p["iteration"] <= 1000 else 1000)
                        for p in pts
                    ):
                        raise ValueError(
                            "Unmatched optimizer or simulator reset history"
                        )
                    if any(
                        p["transitions"]
                        != p["iteration"] * LOCK["n_envs"] * LOCK["horizon"]
                        for p in pts
                    ):
                        raise ValueError("Transition accounting mismatch")
                    checked += len(pts)
        save(
            OUT / "verification.json",
            dict(
                status="PASS",
                valid_primary_training_runs=len(metrics) * len(LOCK["training_seeds"]),
                verified_policy_checkpoints=checked,
                matched_test_records=len(finals),
                matched_validation_records=sum(
                    r["episodes"] for r in rows if r["group"] == "validation"
                ),
                train_transitions=sum(
                    int(c["transitions"]) for c in costs if c["stage"] == "PPO training"
                ),
                reused_expert_final_goal_overlap=0,
                per_task_common_targets=100,
                all_main_simulator_restart_schedules_match=True,
            ),
        )
        md.append(
            "[Release audit of matched cohorts, checkpoints and interaction budgets](verification.json)"
        )
    path = OUT / ("RESULTS_REPORT.md" if strict else "INTERIM_REPORT.md")
    temp = path.with_suffix(".tmp")
    temp.write_text("\n\n".join(md) + "\n", encoding="utf8")
    temp.replace(path)
    return rows, finals


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--interim", action="store_true")
    report(not p.parse_args().interim)
