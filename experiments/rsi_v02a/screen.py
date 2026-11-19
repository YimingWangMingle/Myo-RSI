"""P0/P1 checks run before any adaptation, separately for each source seed."""

import numpy as np
import torch

from .adapters import install
from .measurement import controller, episodes, metadata, paired
from .registry import PROTOCOL, ROOT, attempt, cohorts, read, source, write


def screen(seed):
    with attempt("screen", seed) as (loc, m):
        goals = cohorts()["screen"]
        sites = {}
        models = {}
        for task in ("T1", "T3"):
            with controller(task, seed, len(goals)) as (env, runner):
                models[task] = metadata(env)
                actor = runner.alg.get_policy()
                obs = env.get_observations()
                with torch.no_grad():
                    original = actor(obs).clone()
                    normalizer = {
                        k: v.clone()
                        for k, v in actor.obs_normalizer.state_dict().items()
                    }
                    std = {
                        k: v.clone() for k, v in actor.distribution.state_dict().items()
                    }
                install(runner)
                with torch.no_grad():
                    torch.testing.assert_close(actor(obs), original, atol=0, rtol=0)
                    torch.testing.assert_close(
                        std, actor.distribution.state_dict(), atol=0, rtol=0
                    )
                    torch.testing.assert_close(
                        normalizer, actor.obs_normalizer.state_dict(), atol=0, rtol=0
                    )
                sites[task] = episodes(env, runner, goals, loc, task)
        paired(sites["T1"], sites["T3"])
        s0, s1 = [models[k]["model"] for k in ("T1", "T3")]
        if (
            s0["actuator_order"] != s1["actuator_order"]
            or s0["model_hash"] == s1["model_hash"]
        ):
            raise ValueError("Invalid body intervention")
        ratios = np.array(s1["peak_force"]) / s0["peak_force"]
        if np.any(ratios >= 1) or np.any(ratios <= 0):
            raise ValueError("Peak force change violates body contract")
        with controller("T3", seed, len(goals), 39) as (env, runner):
            reference = episodes(env, runner, goals, loc, "T3_direct39")
        paired(sites["T3"], reference)
        m["environment"] = models
        result = dict(
            normal=float(np.mean([x["success"] for x in sites["T1"]])),
            weak=float(np.mean([x["success"] for x in sites["T3"]])),
            normal_error=float(np.mean([x["error"] for x in sites["T1"]])),
            weak_error=float(np.mean([x["error"] for x in sites["T3"]])),
            direct_reference=source(seed, 39),
            force_ratio=ratios.tolist(),
            direct_witness=sum(
                y["success"] and not x["success"]
                for x, y in zip(sites["T3"], reference)
            ),
        )
        write(loc / "screen.json", result)
        print(
            "SCREEN",
            seed,
            result["normal"],
            result["weak"],
            result["direct_witness"],
            flush=True,
        )


if __name__ == "__main__":
    from .registry import completed

    for seed in read(PROTOCOL)["source_seeds"]:
        try:
            completed("screen", seed)
        except FileNotFoundError:
            screen(seed)
    all_rows = [
        read(completed("screen", s) / "screen.json")
        for s in read(PROTOCOL)["source_seeds"]
    ]
    gap = np.mean([r["normal"] - r["weak"] for r in all_rows])
    err = np.mean([r["weak_error"] / r["normal_error"] - 1 for r in all_rows])
    valid = (
        sum(
            r["weak"] < r["normal"] or r["weak_error"] > r["normal_error"]
            for r in all_rows
        )
        >= 2
    )
    lock = read(PROTOCOL)["gate"]
    write(
        ROOT / "screen_gate.json",
        dict(
            competent=all(r["normal"] >= lock["min_normal"] for r in all_rows),
            degradation=bool(
                valid
                and (
                    gap >= lock["mean_gap_pp"] / 100
                    or err >= lock["relative_error_increase"]
                )
            ),
            viable=all(r["direct_witness"] > 0 for r in all_rows),
            gap_pp=float(gap * 100),
            relative_error_increase=float(err),
        ),
    )
