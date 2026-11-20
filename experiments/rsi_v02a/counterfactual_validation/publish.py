"""Publish both registered designs and count evidence once per physical target."""

import math
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..registry import read, sha, write
from .evaluate import PLAN, ROOT
from .report import PALETTE, STATES

DESIGNS = ("initial_search", "paired_excitation")
LABEL = {"initial_search": "首轮", "paired_excitation": "实测幅度复验"}


def collect(name):
    return pd.concat(
        [
            pd.read_csv(ROOT / d / (name + ".csv"), keep_default_na=False).assign(
                design=d
            )
            for d in DESIGNS
        ],
        ignore_index=True,
    )


def figure(gaps, comparison):
    fig, (a, b) = plt.subplots(
        1, 2, figsize=(10.4, 3.8), gridspec_kw={"width_ratios": [1, 2.5]}
    )
    for j, row in enumerate(comparison.itertuples()):
        bar = a.bar(
            j,
            row.pairs / row.proposed_pairs,
            color=["#71969b", "#137970"][j],
            width=0.55,
        )
        a.bar_label(
            bar, labels=[f"{row.pairs}/{row.proposed_pairs}"], padding=3, fontsize=9
        )
    a.set(
        xticks=[0, 1],
        xticklabels=["Initial", "Measured"],
        ylabel="Eligible candidate pairs / proposed",
        ylim=(0, 1.15),
        title="Muscle excitation matching",
    )
    for j, seed in enumerate([11, 29, 47]):
        group = gaps[gaps.seed.eq(seed)].sort_values("goal")
        b.scatter(
            np.arange(1, len(group) + 1),
            np.full(len(group), j),
            s=75,
            marker="s",
            c=[PALETTE[STATES.index(k)] for k in group.classification],
        )
    for name, c in zip(STATES, PALETTE):
        b.scatter([], [], marker="s", s=48, label=name, c=c)
    b.set(
        yticks=[0, 1, 2],
        yticklabels=[11, 29, 47],
        xticks=[1, 5, 10, 15, 20],
        ylim=(-0.5, 2.5),
        xlabel="Failed target in ascending ID order",
        ylabel="Policy seed",
        title="Verified repairs across both searches",
    )
    b.legend(
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.26),
        fontsize=8,
        frameon=False,
    )
    for ax in (a, b):
        ax.spines[["right", "top"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(
        ROOT / "figures" / "validation_overview.png", dpi=175, bbox_inches="tight"
    )
    plt.close(fig)


def publish():
    files = [
        "runs",
        "resource_ledger",
        "replay_audit",
        "replay_error_summary",
        "selected_goals",
        "search_summary",
        "counterfactual_trials",
        "verified_witnesses",
    ]
    frames = {name: collect(name) for name in files}
    runs, ledger, audits, selection, search, witnesses = (
        frames[n]
        for n in (
            "runs",
            "resource_ledger",
            "replay_audit",
            "selected_goals",
            "search_summary",
            "verified_witnesses",
        )
    )
    # Recompute gates from measured values independently of the trial writer.
    p = read(PLAN)
    observed = list(p["state_atol"].items())
    valid = (
        audits.rng_equal.eq(1)
        & audits.source_success.eq(audits.noop_success)
        & audits.terminal_delta_m.le(p["terminal_error_atol_m"])
    )
    for key, tol in observed:
        if "branch_" + key in audits:
            valid &= audits["branch_" + key].le(tol) & audits["suffix_" + key].le(tol)
    np.testing.assert_array_equal(audits.valid, valid.astype(int))
    trials = frames["counterfactual_trials"]
    assert trials.native_success.eq(trials.terminal_error_m.lt(0.0625)).all()
    for _, sample in trials.groupby(["run_id", "goal_hash", "candidate"]):
        assert len(sample) == 4
        pair = sample.set_index(["space", "repeat"])
        matched = sample.branch_valid.eq(1).all()
        for repeat in (0, 1):
            k, f = pair.loc[("K16", repeat)], pair.loc[("muscle39", repeat)]
            matched &= (
                abs(k.excitation_rms - f.excitation_rms) <= 0.01
                and abs(k.saturation - f.saturation) <= 0.05
            )
        assert sample.paired_calibrated.eq(int(matched)).all()
        for _, replicates in sample.groupby("space"):
            stable = (
                replicates.native_success.nunique() == 1
                and replicates.terminal_error_m.max()
                - replicates.terminal_error_m.min()
                <= p["terminal_error_atol_m"]
            )
            witness = (
                matched
                and stable
                and not replicates.source_success.any()
                and replicates.native_success.all()
            )
            assert (
                replicates.repeat_stable.eq(int(stable)).all()
                and replicates.success_witness.eq(int(witness)).all()
            )
    runs["run_directory"] = [
        (("paired_excitation/" if d == "paired_excitation" else "") + "runs/" + i)
        for d, i in zip(runs.design, runs.run_id, strict=True)
    ]
    witnesses["bc_split"] = ""
    for seed, group in witnesses[witnesses.space.eq("K16")].groupby("seed"):
        keys = sorted(group.goal_hash.unique())
        val = set(keys[: math.ceil(len(keys) * 0.25)])
        ix = group.index
        witnesses.loc[ix, "bc_split"] = np.where(
            witnesses.loc[ix, "goal_hash"].isin(val), "validation", "train"
        )
    bc = witnesses[witnesses.space.eq("K16")].copy()
    assert (bc.groupby("goal_hash").bc_split.nunique() <= 1).all()
    for r in bc.itertuples():
        assert sha(r.trajectory) == r.trajectory_sha256
    for name, df in frames.items():
        df.to_csv(ROOT / (name + ".csv"), index=False)
    bc.to_csv(ROOT / "bc_dataset.csv", index=False)
    gaps = collect("functional_gap_summary")
    rows = []
    for (seed, goal, h), sub in gaps.groupby(["seed", "goal", "goal_hash"]):
        w = witnesses[witnesses.seed.eq(seed) & witnesses.goal_hash.eq(h)]
        k = (w.space == "K16").sum()
        f = (w.space == "muscle39").sum()
        valid = bool(sub.replay_valid.max())
        pairs = int(sub.valid_pairs.sum())
        label = (
            "Both"
            if k and f
            else "K16 only"
            if k
            else "39D only"
            if f
            else "No witness"
            if pairs
            else "No calibrated pair"
            if valid
            else "Replay failed"
        )
        rows.append(
            {
                "seed": seed,
                "goal": goal,
                "goal_hash": h,
                "replay_valid": int(valid),
                "valid_pairs_across_runs": pairs,
                "k16_witnesses_across_runs": k,
                "muscle39_witnesses_across_runs": f,
                "classification": label,
                "designs_with_replay": ",".join(
                    sub.loc[sub.replay_valid.eq(1), "design"]
                ),
            }
        )
    g = pd.DataFrame(rows)
    g["interpretation"] = g.classification.map(
        {
            "Both": "Reuse",
            "K16 only": "Reuse",
            "39D only": "Potential representation gap; test same-K reshape or expansion",
            "No witness": "Unresolved within budget",
            "Replay failed": "Unassessed",
            "No calibrated pair": "Unassessed",
        }
    )
    g.to_csv(ROOT / "functional_gap_summary.csv", index=False)
    assert not selection.groupby(["seed", "goal"]).goal_hash.nunique().gt(1).any()
    overview = []
    for design in DESIGNS:
        ss = search[search.design.eq(design) & search.source_success.eq(0)]
        s = ss[ss.space == "K16"]
        z = ledger[ledger.design.eq(design) & ledger.analysis.eq(1)]
        v = read(ROOT / design / "verification.json")
        overview.append(
            {
                "design": design,
                "failures": v["original_selected_failures"],
                "replay": v["replay_valid_failures"],
                "calibrated_goals": v["with_calibrated_pair"],
                "pairs": int(s.calibrated_candidates.sum()),
                "proposed_pairs": int(s.candidates.sum()),
                "k16_repairs": int(s.verified_repaired_goals.sum()),
                "muscle39_repairs": int(
                    ss[ss.space == "muscle39"].verified_repaired_goals.sum()
                ),
                "formal_transitions": int(z.transitions.sum()),
                "all_transitions": v["all_transitions"],
            }
        )
    comp = pd.DataFrame(overview)
    comp.to_csv(ROOT / "protocol_comparison.csv", index=False)
    figs = ROOT / "figures"
    figs.mkdir(exist_ok=True)
    for f in (ROOT / "paired_excitation" / "figures").glob("*.png"):
        shutil.copyfile(f, figs / f.name)
    figure(g, comp)
    per_seed = {}
    for seed, group in g.groupby("seed"):
        per_seed[str(seed)] = {
            "failures": len(group),
            "passed_replay": int(group.replay_valid.sum()),
            "calibrated_goals": int(group.valid_pairs_across_runs.gt(0).sum()),
            "k16_repaired": int(group.k16_witnesses_across_runs.gt(0).sum()),
            "muscle39_repaired": int(group.muscle39_witnesses_across_runs.gt(0).sum()),
            "full_only": int(group.classification.eq("39D only").sum()),
        }
    classified = {name: int(g.classification.eq(name).sum()) for name in STATES}
    eligible = all(entry["k16_repaired"] >= 2 for entry in per_seed.values())
    costs = int(ledger.transitions.sum())
    arm_cost = ledger[ledger.analysis.eq(1) & ledger.stage.eq("candidate")]
    arm_costs = {
        s: int(arm_cost[arm_cost.details.str.contains(s)].transitions.sum())
        for s in ("K16", "muscle39")
    }
    assert arm_costs["K16"] == arm_costs["muscle39"]
    checks = {d: read(ROOT / d / "verification.json") for d in DESIGNS}
    assert len({v["historical_report_sha256"] for v in checks.values()}) == 1
    assert len({v["bank_file_sha256"] for v in checks.values()}) == 1
    assert (
        checks[DESIGNS[0]]["source_checkpoints"]
        == checks[DESIGNS[1]]["source_checkpoints"]
    )
    validation = {
        "protocols": {
            d: {
                "sha256": v["protocol_sha256"],
                "included": v["included_run_ids"],
                "verification": d + "/verification.json",
            }
            for d, v in checks.items()
        },
        "source_checkpoints": checks[DESIGNS[0]]["source_checkpoints"],
        "bank_file_sha256": checks[DESIGNS[0]]["bank_file_sha256"],
        "historical_report_sha256": checks[DESIGNS[0]]["historical_report_sha256"],
        "attempts": len(runs),
        "analysis_runs": int(runs.included.sum()),
        "independent_failures": len(g),
        "per_seed": per_seed,
        "goal_counts": classified,
        "b3_gate": "ELIGIBLE" if eligible else "BLOCKED",
        "minimum_verified_k16_targets_per_seed": 2,
        "bc_partition_unit": "physical goal SHA256",
        "formal_search_transitions": arm_costs,
        "formal_transitions": int(ledger[ledger.analysis.eq(1)].transitions.sum()),
        "all_transitions": costs,
        "paired_budget_verified": True,
        "bc_goal_leakage": False,
        "artifact_sha256": {
            f.relative_to(ROOT).as_posix(): sha(f)
            for f in sorted([*ROOT.glob("*.csv"), *figs.glob("*.png")])
        },
    }
    write(ROOT / "verification.json", validation)
    matched = int(g.valid_pairs_across_runs.gt(0).sum())
    qualified = int(g.replay_valid.sum())
    kcount = classified["Both"] + classified["K16 only"]
    fcount = classified["Both"] + classified["39D only"]
    text = [
        "# V0.2a-CF｜相同物理状态下的反事实修复",
        "",
        (
            f"**K16 重新募集在 {kcount} 个独立失败目标上找到并复现了修正。** 39D 肌肉搜索修复了 {fcount} 个目标，其中 {classified['39D only']} 个只有 39D 找到修正。"
            f"两轮使用同一批开发目标。按物理目标去重，{len(g)} 个失败中 {qualified} 个至少一次通过严格回放，{matched} 个获得合格的等强度候选对。"
        ),
        "",
        "## 回放与修复结果",
        "",
        "| 策略种子 | 独立失败目标 | 两次无干预回放通过¹ | 有合格候选对 | K16 修复 | 39D 修复 | 仅 39D 修复 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed, value in per_seed.items():
        text.append("| " + " | ".join(map(str, [seed, *value.values()])) + " |")
    text += [
        "",
        "¹ 同一实验、同一条恢复路径的两次回放均需通过；同一目标跨轮仅计一次。每个种子另保留 4 个原本成功的目标作对照。",
        "",
        "![校准覆盖与逐目标修复](figures/validation_overview.png)",
        "",
        f"在有可比候选的目标内，{classified['K16 only']} 个仅 K16 修复，{classified['Both']} 个两组都修复，{classified['39D only']} 个仅 39D 修复，{classified['No witness']} 个两组都未找到修正。有限搜索无法排除其他修正路径，也无法由这些开发目标估计跨目标泛化。",
        "",
        "## 为什么做两轮等预算搜索",
        "",
        "首轮在 39D 轨迹自身的闭环观测上估算 K16 会产生多大的扰动。两组轨迹分开后，实际扰动可能超过配对误差限。第二份协议在看到该问题后冻结，完整重跑三个种子。39D 使用成对 K16 实际 rollout 同一步的 excitation 差向量范数和饱和肌肉数，方向仍在 39 维独立采样。两轮按原容差审核实际 ctrl，所有候选均闭环运行至原生终点。",
        "",
        "| 阶段 | 回放通过的失败目标 | 至少一对有效候选的目标 | 校准候选对 / 提出 | K16 修复目标 | 39D 修复目标 | 正式交互数 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in comp.itertuples():
        text.append(
            f"| {LABEL[r.design]} | {r.replay} | {r.calibrated_goals} | {r.pairs}/{r.proposed_pairs} ({r.pairs / r.proposed_pairs:.1%}) | {r.k16_repairs} | {r.muscle39_repairs} | {r.formal_transitions:,} |"
        )
    text += [
        "",
        "每轮每个合格目标每组 16 个候选，每个候选独立执行两遍。K16 与 39D 共用原始状态、75 步分支点、10 步干预、15 步后续闭环和每个种子的仿真预算。比较使用 39 块肌肉的实际 excitation。",
        "",
        "## 物理重放审计",
        "",
        "保存 MuJoCo 全积分状态，包括位置、速度、肌肉 activation、ctrl、warm-start 和时间，同时恢复任务目标、计数器、动作缓冲及随机源。原 world lane 与批宽固定。刷新观测缓存后重新计算策略输入；所有无干预闭环均逐步检查原始容差。快照失败的目标再用同一初始化与记录的原动作前缀复验。",
        "",
        "| 阶段 | 路径 | 对照次数 | 分支状态全部合格 | 终局判据及 1 mm 合格 | 整段轨迹合格 | 终点最大偏差（mm） |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for (design, route), q in audits.groupby(["design", "route"], sort=False):
        tol = read(PLAN)["state_atol"]
        starts = np.logical_and.reduce(
            [q["branch_" + k] <= v for k, v in tol.items() if "branch_" + k in q]
        )
        end = (q.source_success == q.noop_success) & q.terminal_delta_m.le(0.001)
        text.append(
            f"| {LABEL[design]} | {route} | {len(q)} | {starts.sum()} | {end.sum()} | {q.valid.sum()} | {1000 * q.terminal_delta_m.max():.4f} |"
        )
    text += [
        "",
        "终局一致仍可能伴随中途速度、肌肉 excitation 或积分 warm-start 超限。无干预对照逐变量通过后才启动该目标的搜索。按协议的误差分布和成功对照记录见分轮报告。",
        "",
        "## 修正数据与成本",
        "",
        f"**B3 状态为 {validation['b3_gate']}。** 三个种子的独立 K16 修复目标数为 "
        + "、".join(str(s["k16_repaired"]) for s in per_seed.values())
        + "；门槛为每个种子至少 2 个。已验证 K16 轨迹含观测、原始和修正募集量、身体状态、目标、终局及来源，已按物理目标哈希统一划分训练和验证。",
        "",
        f"正式实验消耗 {validation['formal_transitions']:,} 次仿真交互，搜索两组各 {arm_costs['K16']:,} 次。含两次工程预检和中途失败尝试共 **{costs:,} 次**。每个填充环境、前缀重放、原始筛查及重复验证均计费。每个阶段的实测耗时列在资源账本。",
    ]
    if not eligible:
        text += [
            "",
            "B3 未启动 BC 或局部 PPO，本轮没有新增可与历史 B1/B2 对比的训练成绩。",
        ]
    text += [
        "",
        "## 图表与记录",
        "",
        "[首轮完整报告](initial_search/RESULTS_REPORT.md) · [实测幅度复验完整报告](paired_excitation/RESULTS_REPORT.md) · [历史 V0.2a 结果](../RESULTS_REPORT.md)",
        "",
        "[回放误差](figures/replay_state_error.png) · [回放有效率](figures/replay_validity_rate.png) · [K16/39D 搜索成功](figures/search_success.png) · [复验逐目标修复](figures/goal_repairs.png) · [终点误差改善](figures/terminal_improvement.png) · [复验成本](figures/simulation_cost.png)",
        "",
        "[回放审计](replay_audit.csv) · [分种子搜索](search_summary.csv) · [全部候选](counterfactual_trials.csv) · [独立目标判别](functional_gap_summary.csv) · [成功见证](verified_witnesses.csv) · [BC 数据及统一划分](bc_dataset.csv) · [资源账本](resource_ledger.csv) · [全部运行与哈希](runs.csv) · [机器核验](verification.json)",
    ]
    position = text.index("## 为什么做两轮等预算搜索")
    both = g.loc[g.classification.isin(["K16 only", "Both"]), "goal"]
    full = g.loc[g.classification.eq("39D only"), "goal"]
    describe = lambda goals: "、".join(str(i) for i in goals) or "无"
    text[position:position] = [
        f"**Reuse。** 目标 {describe(both)} 已由固定 K16 Bank 的新募集路径修复，可保留现有 Bank。",
        "",
        f"**潜在表示缺口。** 目标 {describe(full)} 只出现 39D 成功修正，可用于后续 Same-K Reshape 或 Expand 验证。当前结果不足以证明 K16 无解。",
        "",
    ]
    (ROOT / "RESULTS_REPORT.md").write_text("\n".join(text) + "\n", encoding="utf8")
    print(validation["goal_counts"], per_seed, validation["b3_gate"], costs)


if __name__ == "__main__":
    publish()
