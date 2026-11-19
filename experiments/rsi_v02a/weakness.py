"""Prespecified contingency S2, scaled at model compilation before GPU allocation."""

import csv

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from myosuite.core.muscle_conditions import apply_sarcopenia_to_spec

from experiments.synergy_budget.ppo import RecruitmentEnv
from myo_rsi.probe import describe_model, versions
from myo_rsi.train import agent

from .registry import PROTOCOL, ROOT, read


def secondary(k, seed, n):
    env_id = read(PROTOCOL)["sites"]["T3"]
    cfg = load_env_cfg(env_id)
    cfg.scene.num_envs = n
    cfg.seed = seed
    entity = cfg.scene.entities["robot"]
    original = entity.spec_fn
    entity.spec_fn = lambda: apply_sarcopenia_to_spec(original(), force_scale=0.75)
    raw = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    try:
        m = describe_model(raw.sim.mj_model)
        ref = read("results/v0_1r/preflight.json")
        weak = next(
            r for r in ref["rows"] if r["task_id"] == "T3" and r["backend"] == "mjlab"
        )
        if (
            versions() != ref["host"]["packages"]
            or m["hand_contract_hash"] != weak["hand_contract_hash"]
            or m["actuator_order"] != weak["actuator_order"]
        ):
            raise ValueError("Conditional body contract mismatch")
        np.testing.assert_allclose(
            m["peak_force"], np.array(weak["peak_force"]) * 0.75, rtol=1e-6
        )
        env = RecruitmentEnv(raw, k)
        torch.manual_seed(seed)
        return env, agent(env, "T3", seed)
    except BaseException:
        raw.close()
        raise


def active_gate():
    initial = read(ROOT / "screen_gate.json")
    if initial["degradation"]:
        return "T3", initial
    return "S2", read(ROOT / "S2_gate.json")


if __name__ == "__main__":
    from .measurement import controller, episodes, metadata
    from .registry import attempt, cohorts, completed, write

    first = read(ROOT / "screen_gate.json")
    if first["degradation"] or not first["competent"]:
        raise ValueError("S2 not indicated")
    records = []
    for s in read(PROTOCOL)["source_seeds"]:
        try:
            prev = completed("S2-screen", s)
        except FileNotFoundError:
            with attempt("S2-screen", s) as (loc, m):
                prior = completed("screen", s)
                with (prior / "T1.csv").open() as f:
                    normal = list(csv.DictReader(f))
                tests = {}
                for k in (16, 39):
                    with controller("S2", s, 100, k) as (env, runner):
                        m["environment"] = metadata(env)
                        r = episodes(env, runner, cohorts()["screen"], loc, f"S2_K{k}")
                    if [(x["goal_hash"], x["reset_hash"]) for x in r] != [
                        (x["goal_hash"], x["reset_hash"]) for x in normal
                    ]:
                        raise ValueError("S2 nonpaired initial states")
                    tests[k] = r
                data = dict(
                    seed=s,
                    normal=np.mean([int(x["success"]) for x in normal]),
                    weak=np.mean([x["success"] for x in tests[16]]),
                    normal_error=np.mean([float(x["error"]) for x in normal]),
                    weak_error=np.mean([x["error"] for x in tests[16]]),
                    direct_witness=sum(
                        y["success"] and not x["success"]
                        for x, y in zip(tests[16], tests[39])
                    ),
                )
                print("S2 SCREEN", data, flush=True)
                write(loc / "screen.json", data)
            prev = loc
        records.append(read(prev / "screen.json"))
    gap = np.mean([r["normal"] - r["weak"] for r in records])
    err = np.mean([r["weak_error"] / r["normal_error"] - 1 for r in records])
    consistent = (
        sum(
            r["weak"] < r["normal"] or r["weak_error"] > r["normal_error"]
            for r in records
        )
        >= 2
    )
    write(
        ROOT / "S2_gate.json",
        dict(
            competent=first["competent"],
            degradation=bool(consistent and (gap >= 0.1 or err >= 0.1)),
            viable=all(r["direct_witness"] > 0 for r in records),
            gap_pp=float(100 * gap),
            relative_error_increase=float(err),
        ),
    )
