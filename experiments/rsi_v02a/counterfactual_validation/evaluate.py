"""Run immutable same-lane no-op audits, then matched interventions behind the gate."""

import argparse
import csv
import hashlib
import json
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch

from ..measurement import controller, metadata, reset
from ..registry import git, read, sha, source, write
from .replay import (
    Snapshot,
    errors,
    native_result,
    restore_route,
    rng_hash,
    rng_state,
    state,
    step,
)

HERE = Path(__file__).parent
ROOT = Path("results/rsi_v02a/counterfactual_validation")
PLAN = HERE / "protocol.json"
BANK = Path("artifacts/v0_1r/banks/reach_only/K16/seed11/bank.npz")


def append(loc, name, rows):
    if not rows:
        return
    path = loc / (name + ".csv")
    new = not path.exists()
    with path.open("a", newline="", encoding="utf8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        if new:
            writer.writeheader()
        writer.writerows(rows)


@contextmanager
def cost(loc, stage, seed, counter, **extra):
    begin, start = time.perf_counter(), counter[0]
    try:
        yield
    finally:
        append(
            loc,
            "resource_ledger",
            [
                dict(
                    seed=seed,
                    stage=stage,
                    transitions=counter[0] - start,
                    wall_s=time.perf_counter() - begin,
                    details=json.dumps(extra, sort_keys=True),
                )
            ],
        )


def array(value):
    return value.detach().cpu().numpy()


def panel(success, hashes, failures, controls):
    selected = []
    for label, n in [(False, failures), (True, controls)]:
        unique = set()
        for i, hit in enumerate(success):
            if hit == label and hashes[i] not in unique:
                unique.add(hashes[i])
                selected.append(i)
                if len(unique) == n:
                    break
    return sorted(selected)


def run(seed, pilot=False):
    p = read(PLAN)
    n = 4 if pilot else p["worlds"]
    start = 109000 if pilot else p["goals"][str(seed)]
    goals = list(range(start, start + n))
    loc = (
        ROOT
        / "runs"
        / (("pilot" if pilot else "validation") + f"-s{seed}-" + uuid4().hex)
    )
    loc.mkdir(parents=True, exist_ok=False)
    cnt = [0]
    manifest = dict(
        run_id=loc.name,
        seed=seed,
        pilot=pilot,
        status="RUNNING",
        commit=git("rev-parse", "HEAD"),
        source=source(seed),
        bank_sha256=sha(BANK),
        protocol_sha256=sha(PLAN),
        code_sha256={f.name: sha(f) for f in sorted(HERE.glob("*.py"))},
        goal_first=start,
        worlds=n,
        configuration=p,
    )
    write(loc / "manifest.json", manifest)
    print("RUN", loc, flush=True)
    try:
        with torch.inference_mode(), controller("S2", seed, n) as (env, runner):
            raw = env.unwrapped
            raw.cfg.auto_reset = False
            actor = runner.alg.get_policy().eval()
            manifest["environment"] = metadata(env)
            manifest["bank_decoder_hash"] = env.bank_hash
            write(loc / "manifest.json", manifest)
            initial = raw.common_step_counter, raw._sim_step_counter, rng_state()
            tol, t0, horizon = p["state_atol"], p["branch"], p["horizon"]
            commands, reference = [], []
            with cost(loc, "source_episode", seed, cnt):
                obs = reset(raw, goals)
                aim = array(raw.command_manager.get_command("reach")).copy()
                hashes = [hashlib.sha256(a.tobytes()).hexdigest() for a in aim]
                for t in range(1, horizon + 1):
                    native = env.native(actor(obs))
                    obs = step(raw, native, t, cnt)
                    if t <= t0:
                        commands.append(native.clone())
                    if t >= t0:
                        reference.append(state(raw, obs, tol))
                    if t == t0:
                        snap = Snapshot(raw)
                base_hit, base_err = map(array, native_result(raw))
            lanes = panel(base_hit, hashes, p["failures"], p["controls"])
            append(
                loc,
                "source_episodes",
                [
                    dict(
                        seed=seed,
                        goal=g,
                        goal_hash=hashes[i],
                        lane=i,
                        terminal_error_m=float(base_err[i]),
                        success=int(base_hit[i]),
                        selected=int(i in lanes),
                    )
                    for i, g in enumerate(goals)
                ],
            )
            manifest["absent_zero_width_fields"] = sorted(
                tol.keys() - reference[0].keys()
            )
            manifest["integration_state_width"] = snap.state.shape[1]
            routes = {}
            for route in p["replay_routes"]:
                unresolved = [i for i in lanes if i not in routes]
                if not unresolved:
                    break
                all_pass = np.ones(n, dtype=bool)
                for rep in range(p["noop_repeats"]):
                    with cost(loc, "no_op", seed, cnt, route=route, repeat=rep):
                        obs = restore_route(
                            raw, route, snap, initial, goals, commands, cnt
                        )
                        rng_ok = rng_hash(snap.rng) == rng_hash(rng_state())
                        branch = errors(reference[0], state(raw, obs, tol))
                        suffix = {k: torch.zeros_like(v) for k, v in branch.items()}
                        for j in range(1, horizon - t0 + 1):
                            obs = step(raw, env.native(actor(obs)), t0 + j, cnt)
                            d = errors(reference[j], state(raw, obs, tol))
                            suffix = {
                                k: torch.maximum(m, d[k]) for k, m in suffix.items()
                            }
                        hit, err = map(array, native_result(raw))
                        branch, suffix = [
                            {k: array(v) for k, v in a.items()}
                            for a in (branch, suffix)
                        ]
                        rows = []
                        for i in unresolved:
                            ok = bool(
                                rng_ok
                                and hit[i] == base_hit[i]
                                and abs(err[i] - base_err[i])
                                <= p["terminal_error_atol_m"]
                                and all(
                                    branch[k][i] <= tol[k] and suffix[k][i] <= tol[k]
                                    for k in branch
                                )
                            )
                            all_pass[i] &= ok
                            rows.append(
                                dict(
                                    seed=seed,
                                    goal=goals[i],
                                    goal_hash=hashes[i],
                                    lane=i,
                                    source_success=int(base_hit[i]),
                                    route=route,
                                    repeat=rep,
                                    valid=int(ok),
                                    rng_equal=int(rng_ok),
                                    ref_error_m=float(base_err[i]),
                                    noop_error_m=float(err[i]),
                                    terminal_delta_m=float(abs(err[i] - base_err[i])),
                                    noop_success=int(hit[i]),
                                    **{
                                        "branch_" + k: float(v[i])
                                        for k, v in sorted(branch.items())
                                    },
                                    **{
                                        "suffix_" + k: float(v[i])
                                        for k, v in sorted(suffix.items())
                                    },
                                )
                            )
                        append(loc, "replay_audit", rows)
                for i in unresolved:
                    if all_pass[i]:
                        routes[i] = route
                print("REPLAY", seed, route, len(routes), "/", len(lanes), flush=True)
            manifest["verified_lanes"] = {
                str(goals[i]): route for i, route in routes.items()
            }
            write(loc / "manifest.json", manifest)
            if routes and not pilot:
                from .search import search

                search(
                    env,
                    actor,
                    loc,
                    seed,
                    p,
                    cnt,
                    initial,
                    goals,
                    hashes,
                    commands,
                    snap,
                    reference[0],
                    routes,
                    base_hit,
                    base_err,
                )
            manifest["status"] = "COMPLETED" if routes else "REPLAY_BLOCKED"
    except BaseException:
        manifest["status"] = "FAILED"
        manifest["failure"] = traceback.format_exc()
        raise
    finally:
        manifest["transitions"] = cnt[0]
        write(loc / "manifest.json", manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=[11, 29, 47], required=True)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    run(args.seed, args.pilot)
