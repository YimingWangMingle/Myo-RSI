"""Independent double-precision solver check on the same bank-development targets."""

import time

import numpy as np
import torch
from scipy.optimize import minimize, nnls
from threadpoolctl import threadpool_limits

from .bank_study import BANKS, HIGH, LOW, readcsv
from .bank import checksum
from .rerun_v01r import OUT, cpu, load_teacher, read, rollout, table


class RefinedProjection:
    def __init__(self, w):
        self.w = w.astype(float)
        self.step = 1 / np.linalg.norm(self.w, 2) ** 2
        self.stats = []

    def __call__(self, action):
        target = cpu(torch.sigmoid(5 * (action.clamp(-1, 1) - 0.5))).astype(float)
        projected = np.empty_like(target)
        tic = time.perf_counter()
        residuals, gains = [], []
        for i, u in enumerate(target):
            w = self.w

            def objective(c):
                raw = w @ c
                diff = raw.clip(LOW, HIGH) - u
                return 0.5 * (diff @ diff), w.T @ (
                    diff * ((raw >= LOW) & (raw <= HIGH))
                )

            c, _ = nnls(w, u)
            initial, _ = objective(c)
            result = minimize(
                objective,
                c,
                jac=True,
                method="L-BFGS-B",
                bounds=[(0, None)] * len(c),
                options={"maxiter": 250, "ftol": 1e-13, "gtol": 1e-8, "maxls": 30},
            )
            if result.fun <= initial:
                c = result.x
            final, gradient = objective(c)
            residuals.append(np.abs(c - np.maximum(0, c - self.step * gradient)).max())
            gains.append(initial - final)
            projected[i] = (w @ c).clip(LOW, HIGH)
        self.stats.append(
            (
                np.mean(np.array(residuals) <= 2e-5),
                max(residuals),
                time.perf_counter() - tic,
                np.mean((target - projected) ** 2),
                np.mean(gains),
            )
        )
        native = (0.5 + np.log(projected / (1 - projected)) / 5).clip(-1, 1)
        return torch.as_tensor(native, dtype=action.dtype, device=action.device)


def run():
    lock = read(OUT / "run_lock.json")
    seeds = list(range(lock["bank_dev"][0], lock["bank_dev"][1] + 1))
    expected = [
        (r["goal_sha256"], r["reset_obs_sha256"])
        for r in readcsv(OUT / "bank_episodes.csv")
        if r["bank_hash"] == "unmodified_teacher"
    ]
    filename = OUT / "solver_sensitivity.csv"
    details = OUT / "solver_sensitivity_episodes.csv"
    completed = readcsv(filename) if filename.exists() else []
    episodes = readcsv(details) if details.exists() else []
    env, policy, _ = load_teacher("T1", len(seeds))
    try:
        for r in readcsv(OUT / "bank_metrics.csv"):
            label = r["bank_hash"]
            if any(m["bank_hash"] == label for m in completed):
                continue
            with np.load(
                BANKS / f"K{r['K']}" / f"seed{r['nmf_seed']}" / "bank.npz"
            ) as data:
                if checksum(data["W"], data["actuator_names"]) != label:
                    raise ValueError("Bank checksum mismatch")
                project = RefinedProjection(data["W"])
            start = time.perf_counter()
            obs, steps = rollout(env, policy, "T1", seeds, transform=project)
            if [(v["goal_sha256"], v["reset_obs_sha256"]) for v in obs] != expected:
                raise ValueError("Unmatched sensitivity cohort")
            stats = np.asarray(project.stats)
            base = dict(K=r["K"], nmf_seed=r["nmf_seed"], bank_hash=label)
            completed.append(
                dict(
                    **base,
                    successes=sum(v["success"] for v in obs),
                    episodes=len(obs),
                    original_projection_successes=r["projected_successes"],
                    mean_error_m=np.mean([v["terminal_error"] for v in obs]),
                    stationary_fraction=stats[:, 0].mean(),
                    max_projected_residual=stats[:, 1].max(),
                    solver_s=stats[:, 2].sum(),
                    solver_excitation_rmse=np.sqrt(stats[:, 3].mean()),
                    mean_objective_gain_over_nnls=stats[:, 4].mean(),
                    simulated_steps=steps,
                    wall_s=time.perf_counter() - start,
                )
            )
            episodes = [v for v in episodes if v["bank_hash"] != label] + [
                dict(**base, **v) for v in obs
            ]
            table(details, episodes)
            table(filename, completed)
            print(
                "sensitivity",
                r["K"],
                r["nmf_seed"],
                completed[-1]["successes"],
                flush=True,
            )
    finally:
        env.close()


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
