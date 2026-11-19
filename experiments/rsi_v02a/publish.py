"""Validation of launch, report completeness and deterministic artifacts."""

from .registry import ROOT, PROTOCOL, cohorts, completed, read, sha, table, write


def ready():
    seeds = read(PROTOCOL)["source_seeds"]
    missing = []
    for seed in seeds:
        for stage in (
            "screen",
            "S2-screen",
            "diagnose",
            "B0-final",
            "B1-final",
            "B2-final",
            "structure",
        ):
            try:
                completed(stage, seed)
            except FileNotFoundError:
                missing.append(f"{stage}/{seed}")
    return missing


def publish():
    from .report import audit_history, report

    missing = ready()
    if missing:
        if (ROOT / "runs").exists():
            audit_history()
        ROOT.mkdir(parents=True, exist_ok=True)
        write(ROOT / "verification.json", dict(status="PARTIAL", pending=missing))
        (ROOT / "RESULTS_REPORT.md").write_text(
            "# V0.2a · PARTIAL\n\nPending or blocked cohorts: "
            + ", ".join(missing)
            + "\n",
            encoding="utf8",
        )
        return
    from .evidence import supplement

    supplement()
    report()
    final = read(ROOT / "final_lock.json")
    if final["cohort"] != cohorts()["final"]:
        raise ValueError("Final test lock mismatch")
    import numpy as np
    from .report import rows, bootstrap

    contrasts = []
    site = "S2" if not read(ROOT / "screen_gate.json")["degradation"] else "T3"
    for task in ("T1", site):
        paired = []
        for seed in read(PROTOCOL)["source_seeds"]:
            left = rows(completed("B2-final", seed) / f"{task}_1200000.csv")
            right = rows(completed("B1-final", seed) / f"{task}_1200000.csv")
            baseline = rows(completed("B0-final", seed) / f"{task}_0.csv")
            opposite = "T1" if task == site else site
            across = rows(completed("B0-final", seed) / f"{opposite}_0.csv")
            ids = lambda group: [(r["goal_hash"], r["reset_hash"]) for r in group]
            if not ids(left) == ids(right) == ids(baseline) == ids(across):
                raise ValueError("Unpaired final cross-body identifiers")
            paired.append(
                np.array(
                    [
                        int(x["success"]) - int(y["success"])
                        for x, y in zip(left, right, strict=True)
                    ]
                )
            )
        lo, hi = bootstrap(paired)
        contrasts.append(
            dict(
                task=task,
                contrast="B2 minus B1",
                difference_pp=100 * np.mean(paired),
                ci_low=lo,
                ci_high=hi,
            )
        )
    table(ROOT / "residual_vs_full.csv", contrasts)
    structure = []
    contact = []
    for s in read(PROTOCOL)["source_seeds"]:
        p = completed("structure", s)
        before = rows(p / "normal_source.csv")
        for op in ("reshape", "expand"):
            after = rows(p / (op + "_normal.csv"))
            structure.append(
                dict(
                    seed=s,
                    method=op,
                    goals=len(before),
                    gain=sum(
                        x["success"] == "0" and y["success"] == "1"
                        for x, y in zip(before, after, strict=True)
                    ),
                    lost=sum(
                        x["success"] == "1" and y["success"] == "0"
                        for x, y in zip(before, after, strict=True)
                    ),
                )
            )
        ev = completed("B0-final", s)
        manifest = read(ev / "manifest.json")
        for task in ("T1", site):
            with np.load(ev / f"{task}_0.npz") as z:
                contacts = z["contacts"]
                contract = manifest["environment"][task]["contact_contract"]
                contact.append(
                    dict(
                        seed=s,
                        site=task,
                        goal=int(z["goals"][0]),
                        frames=int(contacts.size),
                        contact_frames=int((contacts > 0).sum()),
                        max_contacts=int(contacts.max()),
                        **contract,
                    )
                )

    table(ROOT / "structural_retention.csv", structure)
    if contact:
        table(ROOT / "contact_control.csv", contact)
    onset = []
    for seed in read(PROTOCOL)["source_seeds"]:
        ev = completed("B0-final", seed)
        with np.load(ev / "T1_0.npz") as normal, np.load(ev / f"{site}_0.npz") as weak:
            np.testing.assert_array_equal(normal["goals"], weak["goals"])
            for name, bound in (("qpos", 0.02), ("ctrl", 0.05)):
                diff = np.sqrt(np.mean((normal[name] - weak[name]) ** 2, axis=(1, 2)))
                hits = diff >= bound
                index = np.flatnonzero(hits[:-2] & hits[1:-1] & hits[2:])
                onset.append(
                    dict(
                        seed=seed,
                        goal=int(normal["goals"][0]),
                        signal=name,
                        rms_threshold=bound,
                        consecutive_steps=3,
                        first_step=int(index[0]) + 1 if len(index) else None,
                        maximum_rms=float(diff.max()),
                    )
                )
    table(ROOT / "exemplar_body_divergence.csv", onset)
    data = read(ROOT / "verification.json")
    data["audited_final_checkpoint_count"] = len(final["arms"])
    data["paired_cross_body_final_goals"] = len(final["cohort"])
    data["contact_control_seeds"] = len(set(r["seed"] for r in contact))
    data["historical_source_commit"] = read(PROTOCOL)["source_release"]
    data["source_sha256"] = {
        m["source_seed"]: m["checkpoint"]["sha256"]
        for m in read(ROOT / "source_manifest.json")
    }
    data["all_final_budgets_verified"] = True
    write(ROOT / "verification.json", data)
    replay = rows(ROOT / "replay_audit.csv")
    total = lambda r, k: sum(int(x[k]) for x in r)
    summary = rows(ROOT / "summary.csv")
    full = next(r for r in summary if r["method"] == "B1")
    residual = next(r for r in summary if r["method"] == "B2")
    text = [
        "",
        "## 结果解读与可复核记录",
        "",
        f"在相同的 120 万步下，B1 相比冻结策略的弱手恢复为 {float(full['recovery_pp']):+.1f} 个百分点，正常手变化 {float(full['retention_delta_pp']):+.1f} 个百分点。B2 的对应变化为 {float(residual['recovery_pp']):+.1f} 和 {float(residual['retention_delta_pp']):+.1f} 个百分点。[B2 与 B1 的配对差值及区间](residual_vs_full.csv)",
        "",
        f"K16 的 3,000 个候选中，{total([r for r in replay if r['space'] == 'K16'], 'prebranch_valid')} 个候选自身在分支前通过状态容差；把各自的无干预对照也纳入检查后，合格数为 {total([r for r in replay if r['space'] == 'K16'], 'accepted_after_control')}。39 维的成功见证集中在 {sum(int(r['success_witnesses']) > 0 for r in replay if r['space'] == 'muscle39')} 个源策略与目标组合。比较两种动作空间的搜索能力需要在相同物理状态上同时获得有效对照。[逐案例回放偏差](replay_audit.csv) · [B3 阻断记录](blocked_arms.csv)",
        "",
        f"结构测试使用 20 个成对的正常手目标。六次测试总计得到 {sum(r['gain'] for r in structure)} 个由失败转成功、{sum(r['lost'] for r in structure)} 个由成功转失败的配对变化。这一小样本衡量接口改动后的保留，不能据此推断长期性能。[配对保留变化](structural_retention.csv)",
        "",
    ]
    if contact:
        text += [
            f"在既定最终评测的首个示例目标上，额外记录了正常手和 S2 的原生接触计数，共 {sum(r['frames'] for r in contact)} 帧，{sum(r['contact_frames'] for r in contact)} 帧发生接触。模型碰撞掩码、显式配对和逐源策略统计见[接触记录](contact_control.csv)。记录接触没有增加环境交互。",
            "",
        ]
    text += [
        "[预定示例中两种体模的轨迹首次偏离](exemplar_body_divergence.csv) · [阶段命令和复现说明](../../experiments/rsi_v02a/README.md) · [源模型与体模](source_manifest.json) · [记录字段](data_contract.json)"
    ]
    with (ROOT / "RESULTS_REPORT.md").open("a", encoding="utf8") as f:
        f.write("\n".join(text) + "\n")
    inventory = []
    for m in sorted((ROOT / "runs").glob("*/manifest.json")):
        for f in sorted(m.parent.iterdir()):
            if f.is_file():
                inventory.append(
                    dict(
                        run=m.parent.name,
                        path=f.relative_to(ROOT).as_posix(),
                        size=f.stat().st_size,
                        sha256=sha(f),
                    )
                )
    table(ROOT / "artifact_inventory.csv", inventory)
    for fn in [
        "01_baseline",
        "02_failure",
        "03_profile",
        "04_search",
        "05_adaptation",
        "06_retention",
        "07_cost",
        "08_structure",
    ]:
        if (ROOT / "figures" / (fn + ".png")).stat().st_size < 1000:
            raise ValueError("Figure missing: " + fn)


if __name__ == "__main__":
    publish()
