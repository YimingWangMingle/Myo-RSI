"""Derive release tables and figures from complete original runs and all attempt ledgers."""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..registry import read, sha, source, write
from .evaluate import BANK, PLAN, ROOT

TRIAL = [
    "seed",
    "goal",
    "goal_hash",
    "lane",
    "source_success",
    "route",
    "candidate",
    "space",
    "repeat",
    "branch_valid",
    "native_success",
    "source_error_m",
    "terminal_error_m",
    "improvement_m",
    "excitation_rms",
    "saturation",
    "paired_calibrated",
    "repeat_stable",
    "success_witness",
]
WITNESS = [
    "seed",
    "goal",
    "goal_hash",
    "space",
    "candidate",
    "terminal_error_m",
    "trajectory",
    "trajectory_sha256",
]
SPACES = ("K16", "muscle39")
COLORS = ("#00796B", "#5457A5")
STATES = (
    "Replay failed",
    "No calibrated pair",
    "No witness",
    "K16 only",
    "39D only",
    "Both",
)
PALETTE = ("#dadde1", "#f4cf88", "#92a4b3", "#157d72", "#6955a5", "#164366")


def read_table(loc, name, fields=None):
    path = loc / (name + ".csv")
    return (
        pd.read_csv(path, keep_default_na=False)
        if path.exists()
        else pd.DataFrame(columns=fields)
    )


def save(frame, name):
    frame.to_csv(ROOT / (name + ".csv"), index=False)


def tables(run_ids=None):
    p = read(PLAN)
    ledger, audits, episodes, trials, witnesses, records = [], [], [], [], [], []
    for loc in sorted((ROOT / "runs").iterdir()):
        m = read(loc / "manifest.json")
        assert (
            m["run_id"] == loc.name
            and m["source"] == source(m["seed"])
            and m["bank_sha256"] == sha(BANK)
        )
        entry = read_table(loc, "resource_ledger")
        assert int(entry.transitions.sum()) == m["transitions"]
        include = not m["pilot"] and m["status"] in ("COMPLETED", "REPLAY_BLOCKED")
        if run_ids is not None:
            include = include and loc.name in run_ids
        entry = entry.assign(
            run_id=loc.name,
            analysis=int(include),
            pilot=int(m["pilot"]),
            status=m["status"],
        )
        ledger.append(entry)
        records.append(
            {
                "run_id": loc.name,
                "seed": m["seed"],
                "status": m["status"],
                "pilot": m["pilot"],
                "included": include,
                "transitions": m["transitions"],
                "commit": m["commit"],
                "protocol_sha256": m["protocol_sha256"],
                "checkpoint_sha256": m["source"]["sha256"],
                "bank_sha256": m["bank_sha256"],
                "failure": m.get("failure", ""),
            }
        )
        if not include:
            continue
        assert m["protocol_sha256"] == sha(PLAN)
        a = read_table(loc, "replay_audit")
        e = read_table(loc, "source_episodes").query("selected==1").copy()
        assert e.goal_hash.is_unique
        e["replay_valid"] = e.goal.astype(str).isin(m["verified_lanes"]).astype(int)
        e["route"] = e.goal.astype(str).map(m["verified_lanes"]).fillna("")
        for row in e.itertuples():
            group = a[a.goal.eq(row.goal)].groupby("route").valid.agg(["count", "min"])
            assert bool(row.replay_valid) == bool(
                ((group["count"] == p["noop_repeats"]) & (group["min"] == 1)).any()
            )
        for frame, target in [
            (a, audits),
            (e, episodes),
            (read_table(loc, "counterfactual_trials", TRIAL), trials),
            (read_table(loc, "verified_witnesses", WITNESS), witnesses),
        ]:
            if len(frame):
                target.append(frame.assign(run_id=loc.name))
    reg = pd.DataFrame(records)
    selected = reg[reg.included]
    if sorted(selected.seed) != sorted(p["seeds"]):
        raise ValueError(
            "Need exactly one completed immutable analysis run per source seed"
        )
    l, a, e = pd.concat(ledger), pd.concat(audits), pd.concat(episodes)
    t = pd.concat(trials) if trials else pd.DataFrame(columns=TRIAL + ["run_id"])
    w = (
        pd.concat(witnesses)
        if witnesses
        else pd.DataFrame(columns=WITNESS + ["run_id"])
    )
    assert e.goal_hash.is_unique
    for row in w.itertuples():
        if row.space == "K16":
            assert sha(row.trajectory) == row.trajectory_sha256
            with np.load(row.trajectory) as data:
                assert (
                    int(data["goal_id"]) == row.goal
                    and bool(data["native_success"])
                    and data["observation"].shape == (p["horizon"] - p["branch"], 115)
                )
    w["bc_split"] = ""
    for seed in p["seeds"]:
        target = w.seed.eq(seed) & w.space.eq("K16")
        keys = sorted(set(w.loc[target, "goal_hash"]))
        valid = set(keys[: math.ceil(len(keys) * 0.25)])
        w.loc[target, "bc_split"] = w.loc[target, "goal_hash"].map(
            lambda key, reserved=valid: "validation" if key in reserved else "train"
        )
    if len(t):
        assert (
            t.groupby(["seed", "goal", "space", "candidate"])
            .size()
            .eq(p["independent_repeats"])
            .all()
        )
        assert (
            t.groupby(["seed", "goal", "space"])["candidate"]
            .nunique()
            .eq(p["candidates"])
            .all()
        )
        assert (
            (
                t.loc[
                    t.success_witness.eq(1),
                    [
                        "paired_calibrated",
                        "repeat_stable",
                        "branch_valid",
                        "native_success",
                    ],
                ]
                == 1
            )
            .all()
            .all()
        )
        assert t.query("success_witness==1 and repeat==0").shape[0] == len(w)
        keys = ["run_id", "goal", "space", "candidate"]
        assert set(
            map(tuple, t.query("success_witness==1 and repeat==0")[keys].values)
        ) == set(map(tuple, w[keys].values))
    g = []
    for row in e[e.success == 0].itertuples():
        subset = t[t.goal.eq(row.goal) & t["repeat"].eq(0)]
        kval, fval = (len(w[w.goal.eq(row.goal) & w.space.eq(s)]) for s in SPACES)
        pairs = len(subset[subset.space.eq("K16") & subset.paired_calibrated.eq(1)])
        classification = (
            "Replay failed"
            if not row.replay_valid
            else "No calibrated pair"
            if not pairs
            else "Both"
            if kval and fval
            else "K16 only"
            if kval
            else "39D only"
            if fval
            else "No witness"
        )
        g.append(
            {
                "seed": row.seed,
                "goal": row.goal,
                "goal_hash": row.goal_hash,
                "replay_valid": row.replay_valid,
                "route": row.route,
                "source_error_m": row.terminal_error_m,
                "valid_pairs": pairs,
                "k16_witness_candidates": kval,
                "muscle39_witness_candidates": fval,
                "classification": classification,
                "run_id": row.run_id,
            }
        )
    g = pd.DataFrame(g)
    summary = []
    for seed in p["seeds"]:
        eligible = e[e.seed == seed]
        for space in SPACES:
            subset = t[t.seed.eq(seed) & t.space.eq(space) & t["repeat"].eq(0)]
            for control in (0, 1):
                src = eligible[eligible.success == control]
                s = subset[subset.source_success == control]
                matched = s[s.paired_calibrated.eq(1)]
                ws = (
                    w[w.seed.eq(seed) & w.space.eq(space)]
                    if not control
                    else w.iloc[:0]
                )
                summary.append(
                    {
                        "seed": seed,
                        "space": space,
                        "source_success": control,
                        "selected_goals": len(src),
                        "replay_valid_goals": int(src.replay_valid.sum()),
                        "tested_goals": s.goal.nunique(),
                        "candidates": len(s),
                        "calibrated_candidates": len(matched),
                        "valid_fraction": len(matched) / len(s) if len(s) else np.nan,
                        "successful_candidates": len(ws),
                        "verified_repaired_goals": ws.goal.nunique(),
                        "calibrated_native_success": int(matched.native_success.sum()),
                        "median_improvement_m": matched.improvement_m.median(),
                        "median_excitation_rms": matched.excitation_rms.median(),
                        "mean_saturation": matched.saturation.mean(),
                        "candidate_transitions_including_repeats_and_padding": int(
                            l[
                                (l.analysis == 1)
                                & (l.seed == seed)
                                & (l.stage == "candidate")
                                & l.details.str.contains(space)
                            ].transitions.sum()
                        ),
                    }
                )
    s = pd.DataFrame(summary)
    for seed in p["seeds"]:
        arm = s[s.seed.eq(seed) & s.source_success.eq(0)]
        assert arm.candidate_transitions_including_repeats_and_padding.nunique() == 1
    return reg, l, a, e, t, w, g, s


def plots(a, e, t, g, s, l):
    dest = ROOT / "figures"
    dest.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    def finish(fig, name):
        fig.tight_layout()
        fig.savefig(dest / (name + ".png"), dpi=165, bbox_inches="tight")
        plt.close(fig)

    p = read(PLAN)
    keys = [
        "qpos",
        "qvel",
        "act",
        "ctrl",
        "qacc_warmstart",
        "actuator_length",
        "actuator_velocity",
        "obs",
        "tips",
    ]
    fig, ax = plt.subplots(figsize=(7.7, 3.8))
    for route, c, m in zip(p["replay_routes"], COLORS, ["o", "D"]):
        group = a[a.route == route]
        for phase, style in [("branch_", "--"), ("suffix_", "-")]:
            ratio = [
                max(1e-8, group[phase + k].max() / p["state_atol"][k]) for k in keys
            ]
            ax.plot(
                range(len(keys)),
                ratio,
                style + m,
                color=c,
                label=f"{route}, {phase[:-1]}",
                ms=4,
            )
    ax.axhline(1, color="#555", lw=0.9)
    ax.set(
        yscale="log",
        ylabel="Maximum error / frozen tolerance",
        title="No-op state divergence across selected goals",
    )
    ax.set_xticks(
        range(len(keys)),
        [
            "qpos",
            "qvel",
            "activation",
            "excitation",
            "warm start",
            "length",
            "velocity",
            "observation",
            "tips",
        ],
        rotation=24,
        ha="right",
    )
    ax.legend(ncol=2, fontsize=8)
    finish(fig, "replay_state_error")
    seeds = p["seeds"]
    x = np.arange(len(seeds))
    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    for j, (label, c) in enumerate(
        [("Failed source", COLORS[0]), ("Successful control", COLORS[1])]
    ):
        rates = []
        for seed in seeds:
            subset = e[e.seed.eq(seed) & e.success.eq(j)]
            rates.append(float(subset.replay_valid.mean()))
        bars = ax.bar(x + (j - 0.5) * 0.35, rates, width=0.35, color=c, label=label)
        ax.bar_label(bars, fmt="%.2f", padding=2)
    ax.set(
        xticks=x,
        xticklabels=seeds,
        ylim=(0, 1.12),
        ylabel="Fraction passing both no-op repeats",
        xlabel="Source policy seed",
    )
    ax.legend(fontsize=8)
    finish(fig, "replay_validity_rate")
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    primary = s[s.source_success == 0]
    for j, space in enumerate(SPACES):
        ss = primary[primary.space == space]
        bars = ax.bar(
            x + (j - 0.5) * 0.35,
            ss.verified_repaired_goals,
            width=0.35,
            color=COLORS[j],
            label=space,
        )
        for bar, row in zip(bars, ss.itertuples()):
            ax.annotate(
                f"{row.verified_repaired_goals}/{row.tested_goals}"
                if row.tested_goals
                else "N/A",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
            )
    ax.set(
        xticks=x,
        xticklabels=seeds,
        ylim=(0, max(2, primary.verified_repaired_goals.max() + 1.5)),
        ylabel="Distinct verified repairs",
        xlabel="Source policy seed",
        title="Repairs among replay-qualified failed goals",
    )
    ax.legend()
    finish(fig, "search_success")
    fig, ax = plt.subplots(figsize=(10, 3.2))
    for j, seed in enumerate(seeds):
        ss = g[g.seed == seed].sort_values("goal")
        ax.scatter(
            range(1, len(ss) + 1),
            [j] * len(ss),
            c=[PALETTE[STATES.index(c)] for c in ss.classification],
            marker="s",
            s=97,
        )
    for name, c in zip(STATES, PALETTE):
        ax.scatter([], [], c=c, label=name, marker="s")
    ax.set(
        yticks=x,
        yticklabels=seeds,
        ylabel="Source policy seed",
        xlabel="Failed target in ascending goal ID",
        xticks=range(1, max(g.groupby("seed").size()) + 1),
    )
    ax.legend(ncol=3, bbox_to_anchor=(0.5, -0.32), loc="upper center", fontsize=8)
    finish(fig, "goal_repairs")
    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    for j, space in enumerate(SPACES):
        for i, seed in enumerate(seeds):
            sub = t[
                t.seed.eq(seed)
                & t.space.eq(space)
                & t.source_success.eq(0)
                & t.paired_calibrated.eq(1)
                & t["repeat"].eq(0)
            ]
            if len(sub):
                q = sub.groupby("goal").improvement_m.median() * 1000
                ax.scatter(
                    np.full(len(q), i + (j - 0.5) * 0.25)
                    + np.linspace(-0.04, 0.04, len(q)),
                    q,
                    s=22,
                    color=COLORS[j],
                    alpha=0.7,
                )
            elif j == 0:
                ax.text(i, 0, "N/A", ha="center")
        ax.scatter([], [], color=COLORS[j], label=space)
    ax.axhline(0, color="#555", lw=0.8)
    ax.set(
        xticks=x,
        xticklabels=seeds,
        xlabel="Source policy seed",
        ylabel="Median terminal improvement per goal (mm)",
    )
    ax.legend()
    finish(fig, "terminal_improvement")
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    bottom = np.zeros(3)
    for label, group, color in [
        ("Source", l[(l.analysis == 1) & (l.stage == "source_episode")], "#91b4b0"),
        ("No-op audit", l[(l.analysis == 1) & (l.stage == "no_op")], "#528b95"),
        (
            "K16 search",
            l[
                (l.analysis == 1)
                & (l.stage == "candidate")
                & l.details.str.contains("K16")
            ],
            COLORS[0],
        ),
        (
            "39D search",
            l[
                (l.analysis == 1)
                & (l.stage == "candidate")
                & l.details.str.contains("muscle39")
            ],
            COLORS[1],
        ),
        ("Pilots / failed attempt", l[l.analysis == 0], "#c5aa7d"),
    ]:
        counts = (
            group.groupby("seed").transitions.sum().reindex(seeds, fill_value=0).values
            / 1000
        )
        ax.bar(x, counts, bottom=bottom, color=color, label=label)
        bottom += counts
    ax.set(
        xticks=x,
        xticklabels=seeds,
        xlabel="Source policy seed",
        ylabel="Environment transitions (thousands)",
    )
    ax.legend(fontsize=8, bbox_to_anchor=(1, 1))
    finish(fig, "simulation_cost")


def publish(run_ids=None):
    reg, l, a, e, t, w, g, s = tables(run_ids)
    p = read(PLAN)
    agg = []
    for (seed, route), group in a.groupby(["seed", "route"]):
        for phase in ("branch", "suffix"):
            for variable, tolerance in p["state_atol"].items():
                col = phase + "_" + variable
                if col in group:
                    values = group[col]
                    agg.append(
                        {
                            "seed": seed,
                            "route": route,
                            "phase": phase,
                            "variable": variable,
                            "absolute_tolerance": tolerance,
                            "noop_cases": len(values),
                            "min_error": values.min(),
                            "median_error": values.median(),
                            "p95_error": values.quantile(0.95),
                            "maximum_error": values.max(),
                            "pass_rate": (values <= tolerance).mean(),
                        }
                    )
    for name, frame in [
        ("runs", reg),
        ("replay_audit", a),
        ("replay_error_summary", pd.DataFrame(agg)),
        ("selected_goals", e),
        ("search_summary", s),
        ("counterfactual_trials", t),
        ("verified_witnesses", w),
        ("functional_gap_summary", g),
        ("resource_ledger", l),
    ]:
        save(frame, name)
    plots(a, e, t, g, s, l)
    by_seed = {
        str(seed): int(w[w.space.eq("K16") & w.seed.eq(seed)].goal.nunique())
        for seed in p["seeds"]
    }
    qualified = all(n >= 2 for n in by_seed.values())
    verified = int(g.replay_valid.sum())
    tested = int((g.valid_pairs > 0).sum())
    klass = g.classification.value_counts()
    counts = {k: int(klass.get(k, 0)) for k in STATES}
    archive = Path(p["historical_report"])
    v = {
        "protocol_sha256": sha(PLAN),
        "historical_report_sha256": sha(archive),
        "source_checkpoints": {str(seed): source(seed) for seed in p["seeds"]},
        "bank_file_sha256": sha(BANK),
        "included_run_ids": reg.loc[reg.included, "run_id"].tolist(),
        "attempts": len(reg),
        "completed_analysis_runs": int(reg.included.sum()),
        "original_selected_failures": len(g),
        "replay_valid_failures": verified,
        "with_calibrated_pair": tested,
        "goal_counts": counts,
        "k16_distinct_bc_goals_by_seed": by_seed,
        "b3_gate": "ELIGIBLE" if qualified else "BLOCKED",
        "b3_minimum_per_seed": 2,
        "b3_reason": "At least two distinct verified K16 repairs are required in every policy seed.",
        "analysis_transitions": int(l[l.analysis == 1].transitions.sum()),
        "all_transitions": int(l.transitions.sum()),
        "runwise_replay_gate_recomputed": True,
        "checkpoint_hashes_checked": True,
        "budgets_paired": True,
        "trajectory_hashes_checked": True,
        "unique_physical_goal_hashes": True,
        "artifact_sha256": {
            f.relative_to(ROOT).as_posix(): sha(f)
            for f in sorted([*ROOT.glob("*.csv"), *(ROOT / "figures").glob("*.png")])
        },
    }
    write(ROOT / "verification.json", v)
    lines = [
        "# V0.2a-CF | S2 反事实搜索验证",
        "",
        (
            f"在三个冻结的 K16 PPO 策略上，{len(g)} 个不同的原始失败目标中，**{verified} 个通过两次严格回放**，{tested} 个有合格的配对候选。"
            f"K16 募集搜索在 **{counts['Both'] + counts['K16 only']} 个目标**上复现成功修正，39 维肌肉搜索在 **{counts['Both'] + counts['39D only']} 个目标**上复现成功修正。其中 {counts['39D only']} 个只有 39D 找到修正。"
        ),
        "",
        f"**B3 门控为 {v['b3_gate']}。** 三个策略种子的合格 K16 修复目标数依次为 {by_seed['11']}、{by_seed['29']}、{by_seed['47']}，每个种子的最低要求为 2。",
        "",
        "## 按独立失败目标比较",
        "",
        "| Policy seed | 失败目标 | 回放通过 | 有合格候选对 | K16 成功候选 / 修复目标 | 39D 成功候选 / 修复目标 | 仅 39D 修复 | 两组均未找到 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in p["seeds"]:
        ss = s[s.seed.eq(seed) & s.source_success.eq(0)].set_index("space")
        sub = g[g.seed == seed]
        lines.append(
            f"| {seed} | {len(sub)} | {sub.replay_valid.sum()} | {(sub.valid_pairs > 0).sum()} | {ss.loc['K16', 'successful_candidates']} / {ss.loc['K16', 'verified_repaired_goals']} | {ss.loc['muscle39', 'successful_candidates']} / {ss.loc['muscle39', 'verified_repaired_goals']} | {(sub.classification == '39D only').sum()} | {(sub.classification == 'No witness').sum()} |"
        )
    lines += [
        "",
        "修复要求同一候选在两次独立恢复后均通过原生终局成功判据，终点误差之差 ≤1 mm，且两组扰动通过物理尺度匹配。两组均未找到只适用于有合格候选的目标。只有 39D 成功提示潜在表示缺口，仍需 Same-K Reshape 或 Expand 验证。",
        "",
        "![逐目标修复](figures/goal_repairs.png)",
        "",
        "## 回放可信度",
        "",
        "固定 S2 肌力、原 Checkpoint 和 Bank。在新的开发目标上保留原始 world lane 与批宽。第 75 步先恢复 MuJoCo 全积分状态、肌肉状态、动作及命令缓冲、目标、计数器和随机状态，刷新派生观测。若同一目标无法通过两次无干预对照，再从相同初始化重放保存的 75 步原动作。后续 25 步使用冻结策略闭环。",
        "",
        "沿用历史绝对容差，并逐步检查全部后续状态。每次对照同时要求原生成功标志相同且终点误差差值 ≤1 mm。误差表中的 suffix 是整段轨迹逐变量的最大绝对误差。零宽状态字段记录在各次运行清单内。",
        "",
        "| 路径 | 对照次数 | 分支所有变量通过 | 原生终局及 1 mm 通过 | 全轨迹通过 | 终点差中位数 / 最大值（mm） |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for route, group in a.groupby("route", sort=False):
        branch = np.logical_and.reduce(
            [
                group["branch_" + k] <= at
                for k, at in p["state_atol"].items()
                if "branch_" + k in group
            ]
        )
        end = (group.source_success == group.noop_success) & (
            group.terminal_delta_m <= p["terminal_error_atol_m"]
        )
        lines.append(
            f"| {route} | {len(group)} | {int(branch.sum())} | {int(end.sum())} | {int(group.valid.sum())} | {1000 * group.terminal_delta_m.median():.4f} / {1000 * group.terminal_delta_m.max():.4f} |"
        )
    lines += [
        "",
        "第二条路径只用于第一条路径未通过的目标。分母包含成功对照和失败目标的重复测量。按目标需要同一条路径两次都通过。",
        "",
        "![回放误差](figures/replay_state_error.png)",
        "",
        "![回放通过率](figures/replay_validity_rate.png)",
        "",
        "## 搜索预算、强度及终点变化",
        "",
        "每个合格目标每组 16 个随机候选，每个候选从同一分支独立执行两次。前 10 步分两个 5 步动作块，其后由原策略闭环到第 100 步。39D 方向独立采样，在实际 excitation 空间匹配 K16 扰动的范数及饱和数；每次比较都读取仿真器实际 ctrl。两组平均 RMS 差需 ≤0.01，动作饱和比例差需 ≤0.05。达不到尺度匹配的候选仍保留其终局记录。",
        "",
        "| Seed | 动作空间 | 有效配对候选 / 全部失败目标候选 | Excitation RMS 中位数 | 平均饱和比例 | 终点误差改善中位数（mm） |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for r in s[s.source_success == 0].itertuples():

        def num(value, scale=1):
            return f"{scale * value:.4f}" if np.isfinite(value) else "未测"

        lines.append(
            f"| {r.seed} | {r.space} | {r.calibrated_candidates} / {r.candidates} | {num(r.median_excitation_rms)} | {num(r.mean_saturation)} | {num(r.median_improvement_m, 1000)} |"
        )
    lines += [
        "",
        "误差改善为原始失败终点误差减去干预终点误差，正数表示更接近目标。表内以候选为单位；下图先在同一目标内取中位数，再显示各独立目标。成功对照另列于 search_summary.csv。",
        "",
        "![搜索成功](figures/search_success.png)",
        "",
        "![终点误差改善](figures/terminal_improvement.png)",
        "",
        "## 仿真成本与证据范围",
        "",
        "| Seed | 原始筛查 | 无干预回放 | K16 搜索 | 39D 搜索 | 预检及失败尝试 | 全部交互 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in p["seeds"]:
        q = l[l.seed == seed]
        z = q[q.analysis == 1]
        charges = [
            int(z[z.stage == stage].transitions.sum())
            for stage in ("source_episode", "no_op")
        ]
        charges.extend(
            int(
                z[
                    (z.stage == "candidate") & z.details.str.contains(space)
                ].transitions.sum()
            )
            for space in SPACES
        )
        charges += [int(q[q.analysis == 0].transitions.sum()), int(q.transitions.sum())]
        lines.append(
            "| " + str(seed) + " | " + " | ".join(f"{v:,}" for v in charges) + " |"
        )
    lines += [
        "",
        f"正式实验共 {v['analysis_transitions']:,} 次交互；含所有工程预检和失败尝试共 **{v['all_transitions']:,}** 次。每个未参与搜索的填充 lane 也按实际步数计费。resource_ledger.csv 包含每阶段实测墙钟时间。未重新拟合 Bank 或训练源 PPO。",
        "",
        "![仿真成本](figures/simulation_cost.png)",
        "",
        "对照通过率衡量这套后端、该批宽和该分支时刻的可重复性。被排除的目标没有可解释的干预结果，搜索的修复率仅适用于通过回放的子集。候选共享同一物理目标，不能作为独立样本估计泛化。",
        "",
        "## 数据与复现",
        "",
        "[冻结协议](../../../experiments/rsi_v02a/counterfactual_validation/protocol.json) · [全部运行、代码 Commit 与模型哈希](runs.csv) · [运行清单与原始记录](runs/) · [机器核验](verification.json)",
        "",
        "[逐次回放](replay_audit.csv) · [逐变量误差及通过率](replay_error_summary.csv) · [逐候选终局](counterfactual_trials.csv) · [按种子搜索](search_summary.csv) · [按目标判别](functional_gap_summary.csv) · [已验证修正及 BC 分区](verified_witnesses.csv) · [资源账本](resource_ledger.csv)",
    ]
    if not qualified:
        lines += [
            "",
            "B3 停在数据门槛。未启动 BC 或 Local PPO，本轮没有新增 B3 对 B1/B2 的同预算成绩。",
        ]
    (ROOT / "RESULTS_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    print(
        json.dumps(
            {
                k: v[k]
                for k in [
                    "b3_gate",
                    "goal_counts",
                    "replay_valid_failures",
                    "k16_distinct_bc_goals_by_seed",
                    "all_transitions",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run", action="append", help="Explicit included run ID, once per source seed"
    )
    parser.add_argument("--paired", action="store_true")
    args = parser.parse_args()
    if args.paired:
        ROOT = ROOT / "paired_excitation"
        PLAN = PLAN.with_name("protocol_paired.json")
    publish(args.run)
