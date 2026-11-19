"""Paired trajectory diagnostics and capacity sweep, independent of PPO performance."""

import argparse
import time
import warnings

import numpy as np
import torch
from sklearn.decomposition import NMF
from threadpoolctl import threadpool_limits

from myo_rsi.bank import checksum
from myo_rsi.bank_study import Projection, capped
from myo_rsi.expert_data import DATA
from myo_rsi.rerun_v01r import cpu, load_teacher, rollout

from .common import ART, LOCK, OLD, OUT, atomic_table, bank, read, save, seeds, sha


def fit():
    ds = read(OLD / "dataset_manifest.json")
    for s in ds["splits"]:
        if sha((DATA / f"rollouts_{s['split']}.npz").read_bytes()) != s["sha256"]:
            raise ValueError("Unexpected original dataset")
    with np.load(DATA / "rollouts_train.npz") as d:
        x = d["ctrl"][capped(d["seed"], d["timestep"])]
    with np.load(DATA / "rollouts_val.npz") as d:
        v = d["ctrl"]
    names = bank(8, 11)[2]
    for k in (24, 32):
        for seed in LOCK["nmf_seeds"]:
            folder = ART / "banks" / f"K{k}" / f"seed{seed}"
            folder.mkdir(parents=True, exist_ok=True)
            if (folder / "metrics.json").exists():
                bank(k, seed)
                continue
            tic = time.perf_counter()
            nm = NMF(
                n_components=k,
                init="random",
                solver="cd",
                tol=1e-4,
                max_iter=600,
                random_state=seed,
            )
            with warnings.catch_warnings(record=True) as alerts:
                co = nm.fit_transform(x)
                cv = nm.transform(v)
            scales = nm.components_.max(axis=1)
            if min(scales) <= 0:
                raise ValueError("Empty component")
            w = np.ascontiguousarray((nm.components_ / scales[:, None]).T)
            cf = co * scales
            pred = (cv * scales) @ w.T
            f = folder / "bank.npz"
            np.savez_compressed(
                f,
                W=w,
                q95=np.quantile(cf, 0.95, axis=0).astype(np.float32),
                actuator_names=names,
            )
            meta = dict(
                K=k,
                nmf_seed=seed,
                bank_hash=checksum(w, names),
                file_sha256=sha(f.read_bytes()),
                fit_frames=len(x),
                holdout_frames=len(v),
                fit_relative=float(np.linalg.norm(cf @ w.T - x) / np.linalg.norm(x)),
                holdout_relative=float(np.linalg.norm(pred - v) / np.linalg.norm(v)),
                holdout_mae=float(abs(pred - v).mean()),
                nmf_iter=nm.n_iter_,
                warnings=[str(a.message) for a in alerts],
                fit_wall_s=time.perf_counter() - tic,
                source_sha256=ds["teacher_sha256"],
                data_train_sha256=ds["splits"][0]["sha256"],
            )
            save(folder / "metrics.json", dict(reconstruction=meta))


def first(mask):
    if len(mask) < 3:
        return None
    indices = np.flatnonzero(mask[:-2] & mask[1:-1] & mask[2:])
    return int(indices[0] + 1) if len(indices) else None


def challenge(group, identity_only=False):
    cohort = seeds(group)
    folder = OUT / ("identity" if identity_only else "capacity_" + group)
    folder.mkdir(exist_ok=True)
    if (folder / "complete.json").exists():
        return
    env, policy, source = load_teacher("T1", len(cohort))
    n = len(cohort)
    muscle_names = bank(8, 11)[2]
    joint_names = [
        env.sim.mj_model.joint(i).name.split("/")[-1]
        for i in range(env.sim.mj_model.njnt)
    ]
    items = [("Teacher", 0, 0), ("TeacherRepeat", 0, 0), ("Identity", 39, 0)]
    if not identity_only:
        items.extend(("NMF", k, s) for k in LOCK["k"] for s in LOCK["nmf_seeds"])
    summary = []
    episode_rows = []
    series = []
    anatomy = []
    ref = None
    try:
        for label, k, s in items:
            tic = time.perf_counter()
            if label == "NMF":
                w, _, _, meta = bank(k, s)
                transform = Projection(w)
            elif label == "Identity":
                meta = {"bank_hash": "identity39", "holdout_relative": 0.0}
                transform = Projection(np.eye(39, dtype=np.float32))
            else:
                meta = {"bank_hash": "", "holdout_relative": None}
                transform = None
            traces = {key: [] for key in ("ctrl", "qpos", "requested", "active")}

            def record(t, active, before, raw, env, rwds, errors, goals):
                traces["ctrl"].append(cpu(env.sim.data.ctrl))
                traces["qpos"].append(cpu(env.sim.data.qpos))
                traces["requested"].append(
                    cpu(torch.sigmoid(5 * (raw.clamp(-1, 1) - 0.5)))
                )
                traces["active"].append(active.copy())

            data, steps = rollout(
                env, policy, "T1", cohort, transform=transform, recorder=record
            )
            tr = {key: np.stack(ls) for key, ls in traces.items()}
            match = [(r["goal_sha256"], r["reset_obs_sha256"]) for r in data]
            if ref is None:
                ref, original, goals = tr, data, match
            if match != goals:
                raise ValueError("Mismatch in paired initial conditions")
            nt = min(len(tr["ctrl"]), len(ref["ctrl"]))
            live = tr["active"][:nt] & ref["active"][:nt]
            dc = tr["ctrl"][:nt] - ref["ctrl"][:nt]
            dq = tr["qpos"][:nt] - ref["qpos"][:nt]
            local = (tr["ctrl"] - tr["requested"])[tr["active"]]
            arm = dict(controller=label, K=k, nmf_seed=s)
            rms = lambda d: np.sqrt(np.mean(d * d, axis=-1))
            early = []
            for j, r in enumerate(data):
                ok = live[:, j]
                q = first(rms(dq[ok, j]) >= 0.02)
                u = first(rms(dc[ok, j]) >= 0.05)
                early.append(q)
                episode_rows.append(
                    dict(
                        **arm,
                        **r,
                        first_joint_step=q,
                        first_excitation_step=u,
                        paired_teacher_success=original[j]["success"],
                        paired_steps=int(ok.sum()),
                    )
                )
            for ti in range(nt):
                ok = live[ti]
                if ok.any():
                    series.append(
                        dict(
                            **arm,
                            timestep=ti + 1,
                            pairs=int(ok.sum()),
                            excitation_rmse=float(np.sqrt(np.mean(dc[ti, ok] ** 2))),
                            joint_rmse_rad=float(np.sqrt(np.mean(dq[ti, ok] ** 2))),
                        )
                    )
            for signal, names, diff in (
                ("joint_rad", joint_names, dq),
                ("excitation", muscle_names, dc),
            ):
                if len(names) != diff.shape[-1]:
                    raise ValueError("Unexpected anatomy layout")
                for name, err in zip(names, np.sqrt(np.mean(diff[live] ** 2, axis=0))):
                    anatomy.append(
                        dict(
                            **arm,
                            signal=signal,
                            channel=name,
                            trajectory_rmse=float(err),
                        )
                    )
            diff_count = sum(
                r["success"] != o["success"] for r, o in zip(data, original)
            )
            f = [v for v in early if v is not None]
            summary.append(
                dict(
                    **arm,
                    bank_hash=meta["bank_hash"],
                    heldout_relative=meta["holdout_relative"],
                    successes=sum(r["success"] for r in data),
                    episodes=n,
                    mean_error_m=float(np.mean([r["terminal_error"] for r in data])),
                    terminal_disagreements=diff_count,
                    projected_command_max_abs=float(abs(local).max()),
                    projected_command_rmse=float(np.sqrt(np.mean(local**2))),
                    trajectory_excitation_rmse=float(np.sqrt(np.mean(dc[live] ** 2))),
                    trajectory_joint_rmse_rad=float(np.sqrt(np.mean(dq[live] ** 2))),
                    joint_diverged=len(f),
                    median_first_joint_step=np.median(f).item() if f else None,
                    simulated_steps=steps,
                    wall_s=time.perf_counter() - tic,
                )
            )
            if label == "Identity" and (
                abs(local).max() > 2e-6
                or diff_count > max(3, summary[1]["terminal_disagreements"] + 1)
            ):
                atomic_table(folder / "summary.csv", summary)
                raise ValueError("Identity calibration failed; inspect actual errors")
            print(label, k, s, summary[-1]["successes"], flush=True)
    finally:
        env.close()
    for name, rows in [
        ("summary", summary),
        ("episodes", episode_rows),
        ("trajectories", series),
        ("channels", anatomy),
    ]:
        atomic_table(folder / (name + ".csv"), rows)
    save(
        folder / "complete.json",
        dict(
            teacher=source["model_sha256"],
            group=group,
            scope="paired T1 terminal success",
            status="COMPLETE",
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["identity", "dev", "test"])
    stage = p.parse_args().stage
    with threadpool_limits(limits=2):
        if stage == "identity":
            challenge("projection_dev", True)
        elif stage == "dev":
            fit()
            challenge("projection_dev")
        else:
            challenge("test")
