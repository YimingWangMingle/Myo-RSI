"""Rebuild the study report from immutable attempts and per-target evidence."""

import csv
import json
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from myo_rsi.tables import wilson

from .registry import PROTOCOL, ROOT, cohorts, completed, read, sha, table, write

SEEDS = read(PROTOCOL)["source_seeds"]
ARMS = ["B0", "B1", "B2", "B3"]
FIG = ROOT / "figures"


def rows(file):
    with file.open(encoding="utf8") as f:
        return list(csv.DictReader(f))


def paths_all():
    return sorted((ROOT / "runs").glob("*/manifest.json"))


def cohort(rows_, group, reference=None):
    if [int(r["goal"]) for r in rows_] != cohorts()[group]:
        raise ValueError("Incomplete or reordered evaluation goals")
    ids = [(r["goal_hash"], r["reset_hash"]) for r in rows_]
    if len(set(ids)) != len(ids):
        raise ValueError("Repeated target identity")
    if reference is not None and reference != ids:
        raise ValueError("Methods used different paired starts")
    return ids


def bootstrap(d):
    rng = np.random.default_rng(20261119)
    a = np.asarray(d)
    idx = rng.integers(a.shape[0], size=(3000, a.shape[0], 1))
    j = rng.integers(a.shape[1], size=(3000, 1, a.shape[1]))
    return np.quantile(a[idx, j].mean((1, 2)) * 100, [0.025, 0.975]).tolist()


def fig_save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / (name + ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def audit_history():
    runs, costs = [], []
    for p in paths_all():
        m = read(p)
        if sha(p.parent / "config.json") != m["config_sha256"]:
            raise ValueError("Config checksum")
        for key in ("source_checkpoint", "synergy_bank", "final_checkpoint"):
            obj = m.get(key)
            if obj and sha(obj["uri"]) != obj["sha256"]:
                raise ValueError(f"Lineage mismatch: {p} {key}")
        ev = [
            json.loads(x) for x in (p.parent / "events.jsonl").read_text().splitlines()
        ]
        totals = defaultdict(lambda: [0, 0.0])
        for e in ev:
            if e["stage"] in ("start", "end"):
                continue
            v = totals[e["stage"]]
            v[0] += e["transitions"]
            v[1] += e.get("wall_s") or 0
        for stage, (steps, wall) in totals.items():
            costs.append(
                dict(
                    run_id=m["run_id"],
                    seed=m["train_seed"],
                    method=m["method"],
                    status=m["status"],
                    stage=stage,
                    transitions=steps,
                    wall_s=wall,
                    physical_unique=stage != "attributed_diagnosis",
                )
            )
        runs.append(
            dict(
                run_id=m["run_id"],
                method=m["method"],
                seed=m["train_seed"],
                status=m["status"],
                commit=m["git_commit"],
                protocol_sha256=m["protocol_sha256"],
                manifest=p.relative_to(ROOT).as_posix(),
                wall_s=m.get("wall_s"),
                reason=m.get("failure_or_exclusion_reason"),
            )
        )
    table(ROOT / "runs.csv", runs)
    if costs:
        table(ROOT / "resource_ledger.csv", costs)
    return runs, costs


def screening():
    records = []
    for s in SEEDS:
        a = completed("screen", s)
        b = completed("S2-screen", s)
        ref = None
        for site, p in [
            ("S0", a / "T1.csv"),
            ("S1", a / "T3.csv"),
            ("S2", b / "S2_K16.csv"),
        ]:
            rr = rows(p)
            ref = cohort(rr, "screen", ref)
            records.append(
                dict(
                    seed=s,
                    site=site,
                    n=len(rr),
                    success_pct=100 * np.mean([int(x["success"]) for x in rr]),
                    mean_error_m=np.mean([float(x["error"]) for x in rr]),
                    median_error_m=np.median([float(x["error"]) for x in rr]),
                    p90_error_m=np.quantile([float(x["error"]) for x in rr], 0.9),
                    source=p.relative_to(ROOT).as_posix(),
                )
            )
    table(ROOT / "scenario_screen.csv", records)
    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    for s in SEEDS:
        ax.plot(
            ["Normal", "Native weak", "Controlled weak"],
            [r["success_pct"] for r in records if r["seed"] == s],
            "o-",
            label=f"Seed {s}",
        )
    ax.set(
        ylabel="Native success (%)",
        ylim=(0, 100),
        title="Paired screening on 100 Reach goals",
    )
    ax.legend()
    fig_save(fig, "01_baseline")
    return records


def diagnostics():
    agg = defaultdict(list)
    sums = Counter()
    for s in SEEDS:
        d = completed("diagnose", s)
        for f in [
            "failure_manifest",
            "failure_episodes",
            "failure_clusters",
            "functional_profiles",
            "counterfactual_trials",
        ]:
            for r in rows(d / (f + ".csv")):
                agg[f].append(dict(seed=s, run=d.name, **r))
        for r in read(d / "diagnostic_report.json")["per_case"]:
            for label in ("success_witness", "promising_correction", "invalid"):
                sums[(r["space"], label)] += r[label]
    for f, r in agg.items():
        table(ROOT / (f + ".csv"), r)
    d = completed("diagnose", SEEDS[0])
    case = next(
        r
        for r in agg["failure_manifest"]
        if r["selected"] == "True" and r["success"] == "0"
    )
    g = int(case["goal"])
    with np.load(d / "failure_traces.npz") as z:
        i = list(z["goals"]).index(g)
        fig, axes = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
        for ax, (name, lab) in zip(
            axes,
            [
                ("finger", "Tip errors (m)"),
                ("qpos", "Joint position (rad)"),
                ("recruitment", "Recruitment"),
                ("ctrl", "Excitation / activation"),
            ],
        ):
            ax.plot(z[name][:, i], lw=0.7, alpha=0.7)
            ax.set_ylabel(lab)
        axes[-1].plot(z["act"][:, i].mean(1), color="black", label="mean activation")
        axes[-1].legend()
        axes[-1].set_xlabel("Control step")
        axes[0].set_title(f"Seed {SEEDS[0]}, diagnostic goal {g}")
        fig_save(fig, "02_failure")
    candidates = [
        r
        for r in agg["functional_profiles"]
        if float(r["delta_q95"]) == 0.1 and int(r["repeat"]) == 0
    ]
    coverage = Counter(
        (r["seed"], int(r["goal"])) for r in candidates if r["matched"] == "True"
    )
    best_seed, g = coverage.most_common(1)[0][0] if coverage else (SEEDS[0], g)
    profile = [r for r in candidates if r["seed"] == best_seed and int(r["goal"]) == g]
    mat = np.full((3, 16), np.nan)
    marks = []
    for r in profile:
        j = [1, 5, 10].index(int(r["horizon"]))
        k = int(r["component"]) - 1
        if r["matched"] == "True":
            mat[j, k] = float(r["error_slope"])
        if r["clipped"] == "True":
            marks.append((k, j))
    fig, ax = plt.subplots(figsize=(8, 2.8))
    im = ax.imshow(mat, aspect="auto", cmap="coolwarm")
    for k, j in marks:
        if np.isfinite(mat[j, k]):
            ax.text(k, j, "×", ha="center", va="center", fontsize=8)
    ax.set(
        yticks=range(3),
        yticklabels=["1", "5", "10"],
        xticks=range(16),
        xticklabels=range(1, 17),
        xlabel="Bank component",
        ylabel="Horizon",
        title=f"Seed {best_seed}, goal {g}: local error slope (× clipped; blank invalid)",
    )
    fig.colorbar(im, ax=ax)
    fig_save(fig, "03_profile")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for x, space in enumerate(["K16", "muscle39"]):
        rr = [
            r
            for r in agg["counterfactual_trials"]
            if r["space"] == space and r["matched"] == "True"
        ]
        if rr:
            ax.boxplot(
                [float(r["terminal_error"]) for r in rr],
                positions=[x],
                widths=0.38,
                showfliers=False,
            )
        else:
            ax.text(
                x,
                0.5,
                "No valid pairs",
                transform=ax.get_xaxis_transform(),
                ha="center",
                color="dimgray",
            )
    ax.set(
        xticks=[0, 1],
        xticklabels=["Fixed bank (16D)", "Native muscle (39D)"],
        ylabel="Terminal error (m)",
        title="Matched search and source-policy continuation",
    )
    ax.set_xlim(-0.5, 1.5)
    fig_save(fig, "04_search")
    return sums, agg


def final_metrics(site):
    ref = {}
    vals = {}
    progress = []
    seedstats = []
    missing = []
    for s in SEEDS:
        for arm in ARMS:
            try:
                tr = completed(arm, s)
                val = completed(arm + "-validation", s)
                final = completed(arm + "-final", s)
            except FileNotFoundError:
                missing.append(f"{arm}/{s}")
                continue
            m = read(tr / "manifest.json")
            cost = sum(m["budget"].values())
            if cost != (0 if arm == "B0" else 1200000):
                raise ValueError("Unmatched costs")
            for task in ("T1", site):
                for point in read(val / "evaluated.json"):
                    r = rows(val / f"{task}_{point['total']}.csv")
                    ref["validation", task] = cohort(
                        r, "validation", ref.get(("validation", task))
                    )
                    progress.append(
                        dict(
                            task=task,
                            method=arm,
                            seed=s,
                            total=point["total"],
                            ppo=point["ppo"],
                            success_pct=100 * np.mean([int(x["success"]) for x in r]),
                            episodes=len(r),
                        )
                    )
                p = read(final / "evaluated.json")[-1]
                ep = rows(final / f"{task}_{p['total']}.csv")
                ref["final", task] = cohort(ep, "final", ref.get(("final", task)))
                v = np.array([int(x["success"]) for x in ep])
                vals[task, arm, s] = v
                interval = wilson(int(v.sum()), len(v))
                seedstats.append(
                    dict(
                        site=task,
                        method=arm,
                        seed=s,
                        n=len(v),
                        successes=int(v.sum()),
                        success_pct=100 * v.mean(),
                        wilson_low=interval[0],
                        wilson_high=interval[1],
                        error_m=np.mean([float(x["error"]) for x in ep]),
                        total=cost,
                        checkpoint=p["sha256"],
                        episodes=(final / f"{task}_{p['total']}.csv")
                        .relative_to(ROOT)
                        .as_posix(),
                    )
                )
    if seedstats:
        table(ROOT / "final_performance.csv", seedstats)
    if not progress:
        return [], missing
    for r in list(progress):
        if r["method"] == "B0":
            for a in ARMS[1:]:
                if (r["task"], a, r["seed"]) in vals:
                    progress.append(dict(r, method=a))
    table(ROOT / "adaptation_progress.csv", progress)
    summaries = []
    for a in ARMS:
        if not all((site, a, s) in vals for s in SEEDS):
            continue
        offsets = {}
        for task in ("T1", site):
            data = np.array([vals[task, a, s] for s in SEEDS])
            delta = data - np.array([vals[task, "B0", s] for s in SEEDS])
            offsets[task] = dict(
                mean=100 * data.mean(),
                sd=100 * data.mean(1).std(ddof=1),
                delta=100 * delta.mean(),
                ci=bootstrap(delta),
            )
        auc = []
        for s in SEEDS:
            p = sorted(
                (r["total"], r["success_pct"])
                for r in progress
                if r["task"] == site and r["method"] == a and r["seed"] == s
            )
            if a == "B0":
                auc.append(p[0][1])
            else:
                if [x for x, y in p] != [0, 240000, 600000, 1200000]:
                    raise ValueError("Checkpoint budget schedule")
                auc.append(
                    float(np.trapezoid([y for x, y in p], [x for x, y in p]) / 1200000)
                )
        w, n = offsets[site], offsets["T1"]
        summaries.append(
            dict(
                method=a,
                n_sources=3,
                weak_mean_pct=w["mean"],
                weak_sd_pp=w["sd"],
                recovery_pp=w["delta"],
                recovery_ci_low=w["ci"][0],
                recovery_ci_high=w["ci"][1],
                normal_mean_pct=n["mean"],
                normal_sd_pp=n["sd"],
                retention_delta_pp=n["delta"],
                retention_ci_low=n["ci"][0],
                retention_ci_high=n["ci"][1],
                weak_validation_auc_pct=np.mean(auc),
                recovery_pp_per_million=w["delta"] / 1.2 if a != "B0" else None,
            )
        )
    if summaries:
        table(ROOT / "summary.csv", summaries)
    if all((site, "B3", s) in vals for s in SEEDS):
        contrasts = []
        for a in ["B1", "B2"]:
            for task in ("T1", site):
                delta = np.array(
                    [vals[task, "B3", s] - vals[task, a, s] for s in SEEDS]
                )
                lo, hi = bootstrap(delta)
                contrasts.append(
                    dict(
                        task=task,
                        contrast=f"B3 minus {a}",
                        difference_pp=100 * delta.mean(),
                        ci_low=lo,
                        ci_high=hi,
                    )
                )
        table(ROOT / "paired_contrasts.csv", contrasts)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for arm in ARMS:
        for ax, unit in zip(axes, ["total", "ppo"]):
            groups = defaultdict(list)
            for r in progress:
                if r["task"] == site and r["method"] == arm:
                    if unit == "ppo" and arm == "B3" and r["total"] == 0:
                        continue
                    groups[r[unit]].append(r["success_pct"])
            if groups:
                xx = sorted(groups)
                mu = np.array([np.mean(groups[t]) for t in xx])
                sd = np.array([np.std(groups[t]) for t in xx])
                x = np.array(xx) / 1e6
                ax.plot(x, mu, "o-", label=arm)
                ax.fill_between(x, mu - sd, mu + sd, alpha=0.12)
            ax.set(
                xlabel=unit + " transitions (M)",
                ylabel="Weak validation success (%)",
                ylim=(0, 100),
            )
    axes[1].legend()
    fig_save(fig, "05_adaptation")
    fig, ax = plt.subplots(figsize=(5, 4))
    for arm in ARMS:
        seeds_ = [s for s in SEEDS if (site, arm, s) in vals]
        if seeds_:
            ax.scatter(
                [100 * vals["T1", arm, s].mean() for s in seeds_],
                [100 * vals[site, arm, s].mean() for s in seeds_],
                label=arm,
            )
    ax.set(xlabel="Normal retention (%)", ylabel="Weak recovery (%)")
    ax.legend()
    fig_save(fig, "06_retention")
    return summaries, missing


def report():
    FIG.mkdir(parents=True, exist_ok=True)
    runs, costs = audit_history()
    screening_rows = screening()
    sums, diag = diagnostics()
    site = "S2" if not read(ROOT / "screen_gate.json")["degradation"] else "T3"
    summaries, missing = final_metrics(site)
    # This comparison has no valid three-seed B3 estimate if any seed is missing.
    stage = {
        "P0": "verified source/physical excitation",
        "P1": "native weak insufficient; conditional S2 qualified",
        "P2": "per-branch physical state checks",
        "P3": "budgeted search completed",
        "P4": "incomplete" if missing else "complete",
        "P5": "pending",
        "P6": "partial" if missing else "verified",
    }
    sm = []
    for s in SEEDS:
        try:
            ds = completed("structure", s)
            for r in read(ds / "smoke.json"):
                sm.append(dict(seed=s, **r))
        except FileNotFoundError:
            pass
    if sm:
        table(ROOT / "structure_smoke.csv", sm)
        stage["P5"] = "complete" if len(sm) == 6 else "partial"
        import torch

        fig, axes = plt.subplots(2, 3, figsize=(9.5, 6.7))
        # Show the one measured full-muscle witness direction if available.
        seed = next(
            (
                r["seed"]
                for r in sm
                if r["candidate_source"] == "verified full-action correction"
            ),
            SEEDS[0],
        )
        with np.load("artifacts/v0_1r/banks/reach_only/K16/seed11/bank.npz") as f:
            prior = f["W"]
        for ax, op in zip(axes, ["reshape", "expand"], strict=True):
            z = torch.load(
                completed("structure", seed) / (op + ".pt"),
                map_location="cpu",
                weights_only=True,
            )
            w = z["W"].numpy()
            delta = w.copy()
            delta[:, :16] -= prior
            for axes_, v, lab in zip(
                ax, [prior, w, delta], ["Source", "Edited", "Change"], strict=True
            ):
                if lab == "Change":
                    scale = max(float(np.abs(v).max()), 1e-9)
                    im = axes_.imshow(
                        v, aspect="auto", cmap="coolwarm", vmin=-scale, vmax=scale
                    )
                else:
                    im = axes_.imshow(v, aspect="auto", cmap="viridis", vmin=0, vmax=1)
                axes_.set(title=op + " · " + lab, xlabel="Component", ylabel="Muscle")
                fig.colorbar(im, ax=axes_, shrink=0.7)
        fig.suptitle(f"Source seed {seed}: normalized bank weights and changes")
        fig_save(fig, "08_structure")
    dcost = {
        stage: sum(c["transitions"] for c in costs if c["stage"] == stage)
        for stage in (
            "failure_traces",
            "profile_prefix",
            "functional_profile",
            "search_prefix",
            "search_candidates",
            "search_continuation",
        )
    }
    ppocost = sum(c["transitions"] for c in costs if c["stage"] == "ppo_update")
    unique = sum(c["transitions"] for c in costs if c["physical_unique"])
    searchcost = sum(dcost.values())
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(
        ["Diagnosis + search", "PPO", "Screen/eval/smoke"],
        np.array([searchcost, ppocost, unique - searchcost - ppocost]) / 1e6,
    )
    ax.set(
        ylabel="Physical transitions (M)", title="Measured local simulator interactions"
    )
    fig_save(fig, "07_cost")
    bysite = {
        k: np.mean([r["success_pct"] for r in screening_rows if r["site"] == k])
        for k in ("S0", "S1", "S2")
    }
    eligible = {
        str(s): read(completed("diagnose", s) / "diagnostic_report.json")[
            "eligible_goals"
        ]
        for s in SEEDS
    }
    profilevalid = sum(r["matched"] == "True" for r in diag["functional_profiles"])
    dropped = sum(sums[(k, "invalid")] for k in ("K16", "muscle39"))
    report = [
        "# V0.2a · 弱手诊断与修复",
        "",
        f"**{'PARTIAL（存在未完成的主比较）' if missing or len(sm) != 6 else '已完成注册对照'}** · 本报告由已保存的逐目标记录生成。",
        "",
        f"**验证 1 · 肌力变化**　同一批 100 个目标上，三个 16 维源策略的正常手平均成功率为 {bysite['S0']:.1f}%，原生肌少症手（正常峰值力的 50%）为 {bysite['S1']:.1f}%。预注册的 S2 将全部 39 块肌肉的峰值力设为正常值的 37.5%，平均成功率为 {bysite['S2']:.1f}%。S2 达到差距门槛，三个种子都有同物理条件下的 39 维成功参考。",
        "",
        "![配对筛查](figures/01_baseline.png)",
        "",
        "**验证 2 · 固定协同库能否修正失败**　每个源策略记录 100 个诊断目标，按末端误差分层选择 4 个失败和 1 个成功对照，记录最偏离的指尖、误差首次上升及停滞位置。每个分支均从同目标重置并重放原始动作前缀。下表的成功见证要求通过状态匹配检查，并在完整闭环延续后通过原生成功判据。",
        "",
        "| 搜索空间 | 尝试 | 通过匹配与对照检查 | 成功见证 | 误差改善候选 |",
        "|---|---:|---:|---:|---:|",
    ]
    for k in ("K16", "muscle39"):
        n = sum(r["space"] == k for r in diag["counterfactual_trials"])
        report.append(
            f"| {k} | {n} | {n - sums[k, 'invalid']} | {sums[k, 'success_witness']} | {sums[k, 'promising_correction']} |"
        )
    report += [
        "",
        f"共有 {dropped} 个搜索候选未通过物理状态或无干预延续对照检查。功能扰动的 {len(diag['functional_profiles'])} 个配对读数中，{profilevalid} 个通过状态检查；发生限幅的响应另有标识。各源策略可用于 BC 的独立失败目标数为 "
        + ", ".join(str(len(eligible[str(s)])) for s in SEEDS)
        + "（种子顺序 11、29、47）。单目标或无目标不能完成跨案例 BC 验证。有限搜索失败无法确定物理可达性或协同库容量上限。",
        "",
        "![失败定位](figures/02_failure.png)",
        "![局部响应](figures/03_profile.png)",
        "![反事实搜索](figures/04_search.png)",
        "",
        "**验证 3 · 等预算适应和旧技能保留**　每个可训练分支从相同种子的 K=16 Reach checkpoint 开始，固定同一个解码库。B1 全量更新策略与价值网络，B2 仅更新募集残差与价值网络。B3 预留 24 万步用于诊断和候选搜索，其余 96 万步用于 PPO。基线 B0 没有新增训练。正常手和 S2 的最终结果均使用锁定的 200 个新目标；均值和标准差跨三个源策略计算。",
        "",
        "| 方法 | S2 成功率 %（均值 ± SD） | 相对冻结提升 pp | 正常手成功率 % | 保留变化 pp | S2 验证 AUC % |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in summaries:
        report.append(
            f"| {r['method']} | {r['weak_mean_pct']:.1f} ± {r['weak_sd_pp']:.1f} | {r['recovery_pp']:+.1f} | {r['normal_mean_pct']:.1f} ± {r['normal_sd_pp']:.1f} | {r['retention_delta_pp']:+.1f} | {r['weak_validation_auc_pct']:.1f} |"
        )
    if missing:
        report += [
            "",
            "缺少符合条件的完整最终记录："
            + ", ".join(missing)
            + "。B3 的三种子恢复收益、保留和相对 B1/B2 的优势无法估计。详细原因见各诊断记录。",
        ]
    report += [
        "",
        "[逐源策略最终结果与 Wilson 区间](final_performance.csv) · [恢复、保留和配对 95% 区间](summary.csv) · [验证学习曲线数据](adaptation_progress.csv)",
        "",
        "![适应学习曲线](figures/05_adaptation.png)",
        "![恢复与保留](figures/06_retention.png)",
        "",
        f"**验证 4 · Reshape 与 Expand**　完成 {len(sm)}/6 项独立结构测试。种子、权重来源、首步梯度、初始动作偏差、重新加载后的正常手成功数见[结构测试表](structure_smoke.csv)。每项用相同的 20 个正常手目标做配对评测。缺少合格全肌肉修正方向时，扩维只用工程测试向量验证接口。",
        "",
        "![结构变化](figures/08_structure.png)",
        "",
        f"**成本**　本地实际记账 {unique:,} 次环境交互，其中诊断与搜索 {searchcost:,} 次、PPO 训练 {ppocost:,} 次，筛查、评测及其他已记录交互 {unique - searchcost - ppocost:,} 次。诊断成本包括前缀和续跑，重复引用不重复计费。另复用历史 10 万教师步与每个种子 1,800 万步训练所得源策略。外部教师预训练及设备独占时长未知。未把预算终点当成收敛。",
        "",
        "![成本](figures/07_cost.png)",
        "",
        "**可复现性与范围**　实际配置、环境版本、模型/协同库哈希、唯一运行 ID、失败与中断记录独立保存。源策略种子为 11、29、47。评测使用原生末步成功，所有方法共享目标和初始状态；配对区间对源种子和目标交叉重采样，三个源种子限制推断精度。本阶段未检验自主操作选择、记忆或多轮迭代改进。",
        "",
        "[冻结协议](../../experiments/rsi_v02a/protocol.json) · [所有运行](runs.csv) · [成本明细](resource_ledger.csv) · [筛查](scenario_screen.csv) · [失败分类](failure_clusters.csv) · [全部搜索候选](counterfactual_trials.csv) · [审核](verification.json)",
    ]
    (ROOT / "RESULTS_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf8")
    write(
        ROOT / "verification.json",
        dict(
            status="PARTIAL" if missing or len(sm) != 6 else "COMPLETE",
            stages=stage,
            missing=missing,
            eligible_bc_goals=eligible,
            n_attempts=len(runs),
            unmatched_candidates=dropped,
            source_seeds=SEEDS,
            unique_physical_transitions=unique,
            final_cohort=cohorts()["final"],
            code=read(completed("screen", 11) / "manifest.json")["git_commit"],
            protocol_sha256=sha(PROTOCOL),
        ),
    )
    print("Report", ROOT, stage, flush=True)


if __name__ == "__main__":
    report()
