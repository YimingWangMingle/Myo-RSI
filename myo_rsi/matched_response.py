"""One matched action per bank and target, evaluated before any feedback divergence."""

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .bank_study import BANKS, Projection, readcsv
from .rerun_v01r import OUT, cpu, load_teacher, read, table


@torch.inference_mode()
def measure():
    lock = read(OUT / "run_lock.json")
    seeds = range(lock["bank_dev"][0], lock["bank_dev"][1] + 1)
    env, policy, _ = load_teacher("T1", len(seeds))
    rows = []
    ref = None
    try:
        for row in [dict(K=0, nmf_seed=0, bank_hash="unmodified_teacher")] + readcsv(
            OUT / "bank_metrics.csv"
        ):
            for lane, seed in enumerate(seeds):
                obs, _ = env.reset(
                    seed=seed, env_ids=torch.tensor([lane], device=env.device)
                )
            before = obs["actor"].clone()
            command = policy(before)
            torch.testing.assert_close(command, policy(before), rtol=0, atol=0)
            if not torch.isfinite(command).all():
                raise ValueError("Non-finite teacher action")
            if int(row["K"]):
                with np.load(
                    BANKS / f"K{row['K']}" / f"seed{row['nmf_seed']}" / "bank.npz"
                ) as data:
                    command = Projection(data["W"])(command)
            _, _, end, trunc, _ = env.step(command)
            if (end | trunc).any():
                raise ValueError("Unexpected termination at first step")
            after = {k: cpu(getattr(env.sim.data, k)) for k in ("ctrl", "act", "qpos")}
            if ref is None:
                ref = (cpu(before), after)
            if not np.array_equal(cpu(before), ref[0]):
                raise ValueError("Initial observations differ in open-loop check")
            rows.append(
                dict(
                    K=row["K"],
                    nmf_seed=row["nmf_seed"],
                    bank_hash=row["bank_hash"],
                    targets=len(seeds),
                    **{
                        k + "_rmse": float(np.sqrt(np.mean((v - ref[1][k]) ** 2)))
                        for k, v in after.items()
                    },
                )
            )
        table(OUT / "one_step_response.csv", rows)
    finally:
        env.close()


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        measure()
