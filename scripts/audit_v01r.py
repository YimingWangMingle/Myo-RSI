"""Check the evidence chain of the v0.1r run without modifying its observations."""

import csv
import json

import numpy as np
from sklearn.decomposition import NMF
from threadpoolctl import threadpool_limits

from myo_rsi.bank import checksum
from myo_rsi.bank_study import BANKS, capped
from myo_rsi.expert_data import DATA
from myo_rsi.rerun_v01r import OUT, read, save, sha


def records(name):
    with (OUT / name).open(encoding="utf8", newline="") as f:
        return list(csv.DictReader(f))


def audit():
    lock = read(OUT / "run_lock.json")
    pf = read(OUT / "preflight.json")
    assert pf["normal_gate"] and pf["adaptation_gate"]
    assert len(pf["rows"]) == 8 and all(
        r["verification_status"] == "PASS" for r in pf["rows"]
    )
    assert all(r["mapping_max_error"] < 2e-6 for r in pf["rows"])
    eval_rows = records("teacher_eval_episodes.csv")
    validation = set(range(lock["validation"][0], lock["validation"][1] + 1))
    final = set(range(lock["final"][0], lock["final"][1] + 1))
    for summary in records("teacher_eval_summary.csv"):
        group = [r for r in eval_rows if r["task"] == summary["task"]]
        assert len(group) == 100 and {int(r["seed"]) for r in group} == validation
        assert sum(int(r["success"]) for r in group) == int(summary["successes"])
        threshold = 0.7 if summary["task"] == "T2" else 0.0625
        assert all(
            (float(r["terminal_error"]) < threshold) == bool(int(r["success"]))
            for r in group
        )
    ds = read(OUT / "dataset_manifest.json")
    assert (
        ds["gate"] == "PASS"
        and ds["teacher_sha256"]
        == read(OUT / "teacher_manifest.json")[0]["model_sha256"]
    )
    source = {int(r["seed"]): r for r in records("dataset_episodes.csv")}
    assert len(source) == 1000 and len(
        {r["goal_sha256"] for r in source.values()}
    ) == len(source)
    bankdev = set(range(lock["bank_dev"][0], lock["bank_dev"][1] + 1))
    assert not (
        set(source) & (validation | bankdev | final)
        or (validation & bankdev)
        or ((validation | bankdev) & final)
    )
    goals = {r["goal_sha256"] for r in source.values()}
    assert not goals & {r["goal_sha256"] for r in eval_rows if r["task"] == "T1"}
    for split in ds["splits"]:
        path = DATA / f"rollouts_{split['split']}.npz"
        assert sha(path.read_bytes()) == split["sha256"]
        with np.load(path, allow_pickle=False) as d:
            seed, t, ctrl = d["seed"], d["timestep"], d["ctrl"]
            assert (
                ctrl.shape == (split["frames"], 39)
                and len(np.unique(seed)) == split["episodes"]
            )
            np.testing.assert_allclose(
                ctrl,
                1 / (1 + np.exp(-5 * (d["native_action"] - 0.5))),
                rtol=0,
                atol=2e-6,
            )
            np.testing.assert_array_equal(
                d["native_action"], np.clip(d["raw_action"], -1, 1)
            )
            goal = d["goal"]
            for s, idx, n in zip(
                *np.unique(seed, return_index=True, return_counts=True)
            ):
                r = source[int(s)]
                assert r["split"] == split["split"] and n == int(r["length"])
                assert sha(goal[idx].tobytes()) == r["goal_sha256"] and np.all(
                    goal[idx : idx + n] == goal[idx]
                )
                np.testing.assert_array_equal(t[idx : idx + n], np.arange(1, n + 1))
                assert (float(r["terminal_error"]) < 0.0625) == bool(int(r["success"]))
                label = (
                    "success"
                    if int(r["success"])
                    else (
                        "near_success"
                        if float(r["terminal_error"]) < lock["near_success_reach_m"]
                        else "other_failure"
                    )
                )
                assert label == r["outcome"]
            if split["split"] == "train":
                fit = ctrl[capped(seed, t)]
    recon = records("bank_metrics.csv")
    assert len(recon) == 12
    hashes = {r["bank_hash"] for r in recon}
    for r in recon:
        path = BANKS / f"K{r['K']}" / f"seed{r['nmf_seed']}" / "bank.npz"
        assert sha(path.read_bytes()) == r["file_sha256"]
        with np.load(path) as d:
            w = d["W"]
            assert (
                w.shape == (39, int(r["K"])) and np.isfinite(w).all() and (w >= 0).all()
            )
            assert (
                np.allclose(w.max(axis=0), 1)
                and checksum(w, d["actuator_names"]) == r["bank_hash"]
            )
            assert (d["q95"] > 0).all()
            if int(r["K"]) == 4 and int(r["nmf_seed"]) == 11:
                with threadpool_limits(limits=2):
                    refit = NMF(
                        n_components=4,
                        init="random",
                        solver="cd",
                        tol=1e-4,
                        max_iter=600,
                        random_state=11,
                    ).fit(fit)
                h = refit.components_
                np.testing.assert_allclose(
                    w, (h / h.max(axis=1)[:, None]).T, rtol=1e-4, atol=1e-5
                )
    original = records("bank_episodes.csv")
    ref = {
        int(r["seed"]): (r["goal_sha256"], r["reset_obs_sha256"])
        for r in original
        if r["bank_hash"] == "unmodified_teacher"
    }
    assert set(ref) == bankdev and not goals & {pair[0] for pair in ref.values()}
    for data, summary, key in [
        (original, records("bank_functional_coverage.csv"), "successes"),
        (
            records("solver_sensitivity_episodes.csv"),
            records("solver_sensitivity.csv"),
            "successes",
        ),
    ]:
        for row in summary:
            sub = [r for r in data if r["bank_hash"] == row["bank_hash"]]
            assert len(sub) == 100 and all(
                ref[int(r["seed"])] == (r["goal_sha256"], r["reset_obs_sha256"])
                for r in sub
            )
            assert sum(int(r["success"]) for r in sub) == int(row[key])
    assert {r["bank_hash"] for r in records("solver_sensitivity.csv")} == hashes
    sel = read(OUT / "bank_selection.json")
    if sel["status"] == "BANK_INADEQUATE":
        assert sel["selected"] is None and not any(r["passes"] == "True" for r in recon)
    return dict(
        status="PASS",
        environment_probes=8,
        teacher_episodes=len(eval_rows),
        unique_dataset_goals=len(goals),
        audited_frames=sum(s["frames"] for s in ds["splits"]),
        fit_frames_after_phase_cap=len(fit),
        verified_banks=len(hashes),
        refit="K4 seed11 reproduced within rtol=1e-4 and atol=1e-5",
        original_and_projected_episodes=len(original),
        independent_solver_episodes=len(records("solver_sensitivity_episodes.csv")),
        reused_final_seed_ids_in_this_rerun=0,
        selection=sel["status"],
    )


if __name__ == "__main__":
    result = audit()
    save(OUT / "verification.json", result)
    print(json.dumps(result, indent=2))
