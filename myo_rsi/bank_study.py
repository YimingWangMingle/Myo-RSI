"""Fit Reach-only NMF candidates and challenge each with matched-state feedback."""

import csv
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from .bank import checksum
from .expert_data import DATA
from .rerun_v01r import OUT, cpu, load_teacher, read, rollout, save, sha, table

BANKS = Path("artifacts/v0_1r/banks/reach_only")
LOW, HIGH = 1 / (1 + np.exp(7.5)), 1 / (1 + np.exp(-2.5))


def readcsv(path):
    with path.open(newline="", encoding="utf8") as f:
        return list(csv.DictReader(f))


def capped(seed, t):
    phase = (t - 1) // 25
    groups = seed * 10 + phase
    _, start, count = np.unique(groups, return_index=True, return_counts=True)
    return np.concatenate(
        [i + np.linspace(0, n - 1, min(20, n), dtype=int) for i, n in zip(start, count)]
    )


def hist(values):
    return np.histogram(values, bins=np.linspace(LOW - 1e-6, HIGH + 1e-6, 31))[
        0
    ].tolist()


def fit_all():
    lock = read(OUT / "run_lock.json")
    dataset = read(OUT / "dataset_manifest.json")
    if dataset["gate"] != "PASS":
        raise RuntimeError("P1 gate closed")
    for s in dataset["splits"]:
        if sha((DATA / f"rollouts_{s['split']}.npz").read_bytes()) != s["sha256"]:
            raise ValueError("Trajectory partition checksum mismatch")
    with np.load(DATA / "rollouts_train.npz") as d:
        train = d["ctrl"][capped(d["seed"], d["timestep"])]
    with np.load(DATA / "rollouts_val.npz") as d:
        held = d["ctrl"]
        classes = d["outcome"]
    ref = next(
        r
        for r in read(OUT / "preflight.json")["rows"]
        if r["backend"] == "mjlab" and r["task_id"] == "T1"
    )
    names = ref["actuator_order"]
    recon, muscles, coverage, decoders = [], [], [], []
    for k in lock["bank_sizes"]:
        for seed in lock["nmf_seeds"]:
            path = BANKS / f"K{k}" / f"seed{seed}"
            path.mkdir(parents=True, exist_ok=True)
            if not (path / "metrics.json").exists():
                start = time.perf_counter()
                nmf = NMF(
                    n_components=k,
                    init="random",
                    solver="cd",
                    tol=1e-4,
                    max_iter=600,
                    random_state=seed,
                )
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    c = nmf.fit_transform(train)
                    d = nmf.transform(held)
                norm = nmf.components_.max(axis=1)
                if np.any(norm <= 0):
                    raise ValueError("Unidentifiable empty NMF column")
                w = np.ascontiguousarray((nmf.components_ / norm[:, None]).T)
                c *= norm
                d *= norm
                pred = d @ w.T
                bounded = pred.clip(LOW, HIGH)
                excess = np.abs(pred - bounded)
                unit = w / np.linalg.norm(w, axis=0)
                similarity = (unit.T @ unit - np.eye(k)).max()
                q95 = np.quantile(c, 0.95, axis=0).astype(np.float32)
                bank_hash = checksum(w, names)
                file = path / "bank.npz"
                np.savez_compressed(file, W=w, q95=q95, actuator_names=names)
                base = dict(K=k, nmf_seed=seed, bank_hash=bank_hash)
                meta = dict(
                    **base,
                    source_sha256=dataset["teacher_sha256"],
                    data_train_sha256=dataset["splits"][0]["sha256"],
                    file_sha256=sha(file.read_bytes()),
                    nmf_iter=int(nmf.n_iter_),
                    convergence_warnings=[str(v.message) for v in caught],
                    fit_frames=len(train),
                    holdout_frames=len(held),
                    fit_relative=float(
                        np.linalg.norm(c @ w.T - train) / np.linalg.norm(train)
                    ),
                    holdout_relative=float(
                        np.linalg.norm(pred - held) / np.linalg.norm(held)
                    ),
                    holdout_mae=float(np.abs(pred - held).mean()),
                    clipped_fraction=float((excess > 1e-7).mean()),
                    mean_clip_excess=float(excess.mean()),
                    max_column_cosine=float(similarity),
                    dead_columns=int(np.sum(np.max(w, axis=0) == 0)),
                    fit_wall_s=time.perf_counter() - start,
                )
                muscle = [
                    dict(**base, muscle=n, holdout_mae=float(v))
                    for n, v in zip(names, np.abs(pred - held).mean(axis=0))
                ]
                cats = []
                for label in np.unique(classes):
                    keep = classes == label
                    cats.append(
                        dict(
                            **base,
                            outcome=str(label),
                            n=int(keep.sum()),
                            relative_error=float(
                                np.linalg.norm((pred - held)[keep])
                                / np.linalg.norm(held[keep])
                            ),
                        )
                    )
                probe = (
                    np.random.default_rng(701).normal(size=(8192, k)).astype(np.float32)
                )
                sat = []
                for decoder, co in [
                    ("teacher", None),
                    ("heldout_NNLS", d),
                    ("legacy_zero", np.full((1, k), 0.15 * np.log(2))),
                    ("calibrated_zero", np.zeros((1, k))),
                    ("legacy_normal", 0.15 * np.logaddexp(0, probe)),
                    ("calibrated_normal", q95 * np.maximum(probe, 0)),
                ]:
                    raw = held if co is None else co @ w.T
                    physical = raw.clip(LOW, HIGH)
                    sat.append(
                        dict(
                            **base,
                            source=decoder,
                            n_values=raw.size,
                            mean_excitation=float(physical.mean()),
                            fraction_clipped=float(
                                (np.abs(raw - physical) > 1e-7).mean()
                            ),
                            frac_at_low=float((physical <= LOW + 1e-6).mean()),
                            frac_at_high=float((physical >= HIGH - 1e-6).mean()),
                            histogram=hist(physical),
                        )
                    )
                save(
                    path / "metrics.json",
                    dict(
                        reconstruction=meta, muscles=muscle, outcome=cats, decoder=sat
                    ),
                )
            info = read(path / "metrics.json")
            record = info["reconstruction"]
            if (
                record["data_train_sha256"] != dataset["splits"][0]["sha256"]
                or record["source_sha256"] != dataset["teacher_sha256"]
                or sha((path / "bank.npz").read_bytes()) != record["file_sha256"]
            ):
                raise ValueError("Stale or corrupted candidate cache")
            recon.append(record)
            muscles.extend(info["muscles"])
            coverage.extend(info["outcome"])
            decoders.extend(info["decoder"])
            table(OUT / "bank_reconstruction.csv", recon)
            print("fit", k, seed, recon[-1]["holdout_relative"], flush=True)
    table(OUT / "bank_muscle_mae.csv", muscles)
    table(OUT / "bank_outcome_coverage.csv", coverage)
    table(OUT / "decoder_saturation.csv", decoders)


class Projection:
    """Batched NNLS followed by projected descent on the clipped loss (local stationarity)."""

    def __init__(self, w):
        self.w = w
        self.gram = w.T @ w
        self.step = 1 / np.linalg.eigvalsh(self.gram)[-1]
        self.pinv = np.linalg.pinv(w.T)
        self.measure = []
        self.histogram = np.zeros(30, dtype=np.int64)

    def __call__(self, action):
        target = cpu(torch.sigmoid(5 * (action.clamp(-1, 1) - 0.5)))
        tic = time.perf_counter()
        rhs = target @ self.w
        c = np.maximum(target @ self.pinv, 0)
        y = c
        for i in range(48):
            new = np.maximum(0, y - self.step * (y @ self.gram - rhs))
            y = new + (i / (i + 3)) * (new - c)
            c = new
        # A squared clipped residual has the same Lipschitz step within each active region.
        # Guard uphill steps explicitly at the piecewise boundaries.
        for j in range(240):
            raw = c @ self.w.T
            diff = raw.clip(LOW, HIGH) - target
            # At the lower boundary use the right derivative, allowing a silent muscle to recruit.
            gradient = (diff * ((raw >= LOW) & (raw <= HIGH))) @ self.w
            candidate = np.maximum(0, c - self.step * gradient)
            distance = np.abs(candidate - c).max(axis=1, keepdims=True)
            if np.all(distance < 2e-5):
                break
            obj = np.sum(diff * diff, axis=1, keepdims=True)
            trial = np.sum(
                ((candidate @ self.w.T).clip(LOW, HIGH) - target) ** 2,
                axis=1,
                keepdims=True,
            )
            for _ in range(12):
                bad = trial > obj + 1e-7
                if not bad.any():
                    break
                candidate = np.where(bad, (candidate + c) * 0.5, candidate)
                trial = np.sum(
                    ((candidate @ self.w.T).clip(LOW, HIGH) - target) ** 2,
                    axis=1,
                    keepdims=True,
                )
            c = np.where(trial <= obj + 1e-7, candidate, c)
        raw = c @ self.w.T
        phys = raw.clip(LOW, HIGH)
        grad = ((phys - target) * ((raw >= LOW) & (raw <= HIGH))) @ self.w
        resid = np.abs(np.maximum(0, c - self.step * grad) - c).max(axis=1)
        elapsed = time.perf_counter() - tic
        self.measure.append(
            (
                float(((phys - target) ** 2).mean()),
                float(np.mean(np.abs(phys - raw) > 1e-7)),
                elapsed,
                j + 1,
                float(np.mean(resid <= 2e-5)),
                float(np.max(resid)),
            )
        )
        self.histogram += hist(phys)
        native = np.clip(0.5 + np.log(phys / (1 - phys)) / 5, -1, 1)
        return torch.as_tensor(native, dtype=action.dtype, device=action.device)


def functional():
    lock = read(OUT / "run_lock.json")
    dataset = read(OUT / "dataset_manifest.json")
    seeds = list(range(lock["bank_dev"][0], lock["bank_dev"][1] + 1))
    env, policy, source = load_teacher("T1", len(seeds))
    if source["model_sha256"] != dataset["teacher_sha256"]:
        raise ValueError("Teacher changed")
    history = OUT / "bank_episodes.csv"
    episodes = readcsv(history) if history.exists() else []
    summary = OUT / "bank_functional_coverage.csv"
    measured = readcsv(summary) if summary.exists() else []
    reference = None
    try:
        recon = readcsv(OUT / "bank_reconstruction.csv")
        for row in [dict(K=0, nmf_seed=0, bank_hash="unmodified_teacher")] + recon:
            k, seed = int(row["K"]), int(row["nmf_seed"])
            label = row["bank_hash"]
            present = [r for r in episodes if r["bank_hash"] == label]
            if not any(r["bank_hash"] == label for r in measured):
                projector = None
                if k:
                    path = BANKS / f"K{k}" / f"seed{seed}" / "bank.npz"
                    with np.load(path) as d:
                        w = d["W"]
                        if checksum(w, d["actuator_names"]) != label:
                            raise ValueError("Candidate hash failed")
                    projector = Projection(w)
                start = time.perf_counter()
                results, steps = rollout(env, policy, "T1", seeds, transform=projector)
                if (
                    reference
                    and [(r["goal_sha256"], r["reset_obs_sha256"]) for r in results]
                    != reference
                ):
                    raise RuntimeError("Unmatched bank development initial states")
                stats = np.asarray(projector.measure) if projector else np.zeros((1, 6))
                base = dict(K=k, nmf_seed=seed, bank_hash=label)
                present = [dict(**base, **r) for r in results]
                episodes = [r for r in episodes if r["bank_hash"] != label] + present
                measured.append(
                    dict(
                        **base,
                        successes=sum(r["success"] for r in results),
                        episodes=len(seeds),
                        mean_error_m=float(
                            np.mean([r["terminal_error"] for r in results])
                        ),
                        simulated_steps=steps,
                        wall_s=time.perf_counter() - start,
                        solver_s=float(stats[:, 2].sum()),
                        solver_excitation_rmse=float(np.sqrt(stats[:, 0].mean())),
                        solver_clip_fraction=float(stats[:, 1].mean()),
                        solver_mean_iterations=float(stats[:, 3].mean()),
                        solver_stationary_fraction=float(stats[:, 4].mean())
                        if projector
                        else None,
                        max_projected_residual=float(stats[:, 5].max()),
                        physical_histogram=projector.histogram.tolist()
                        if projector
                        else None,
                    )
                )
                table(history, episodes)
                table(summary, measured)
                print("functional", k, seed, measured[-1]["successes"], flush=True)
            if len(present) != len(seeds):
                raise RuntimeError("Incomplete functional episode table")
            if not k:
                reference = [(r["goal_sha256"], r["reset_obs_sha256"]) for r in present]
    finally:
        env.close()
    original = int(measured[0]["successes"])
    if not original:
        raise ValueError("Zero matched teacher reference")
    metrics = []
    for row, fn in zip(recon, measured[1:], strict=True):
        if row["bank_hash"] != fn["bank_hash"]:
            raise ValueError("Mismatched reconstruction and control results")
        ratio = int(fn["successes"]) / original
        passed = (
            ratio >= lock["relative_success_min"]
            and float(row["holdout_relative"]) <= 0.5
            and float(row["mean_clip_excess"]) <= 0.05
        )
        metrics.append(
            dict(
                **row,
                projected_successes=int(fn["successes"]),
                teacher_successes=original,
                relative_success=ratio,
                mean_error_m=fn["mean_error_m"],
                passes=passed,
            )
        )
    selected = None
    for k in lock["bank_sizes"]:
        group = [r for r in metrics if int(r["K"]) == k]
        eligible = [r for r in group if r["passes"]]
        if len(eligible) >= 2:
            middle = np.median([r["projected_successes"] for r in group])
            selected = min(
                eligible,
                key=lambda r: (
                    abs(r["projected_successes"] - middle),
                    int(r["nmf_seed"]),
                ),
            )
            break
    table(OUT / "bank_metrics.csv", metrics)
    save(
        OUT / "bank_selection.json",
        dict(
            status="PASS" if selected else "BANK_INADEQUATE",
            selected=selected,
            teacher_successes=original,
            cohort=lock["bank_dev"],
            rule=lock["bank_selection"],
        ),
    )


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        fit_all()
        functional()
