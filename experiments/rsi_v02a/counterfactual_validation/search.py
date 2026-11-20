"""Matched random shooting with physical excitation calibration and native terminal verification."""

import numpy as np
import torch

from ..registry import sha
from .replay import (
    errors,
    native_result,
    restore_route,
    rng_hash,
    rng_state,
    state,
    step,
)


def excitation(native):
    return torch.sigmoid(5 * (native - 0.5))


def muscle_proposal(native, template, direction, measured_target=None):
    """Choose bounded coordinates in 39D, then match the template's physical norm."""
    base = excitation(native)
    desired = (excitation(template) - base).norm(dim=1, keepdim=True)
    count = (template.abs() >= 0.99999).sum(1, keepdim=True)
    if measured_target is not None:
        desired, count = measured_target
    low, high = excitation(base.new_tensor([-1.0, 1.0]))
    margin = 0.00002
    upper = high - base - (base - low) < 0.18 * direction
    edge = torch.where(upper, high, low)
    score = (edge - base).abs() - 0.035 * direction.abs()
    fixed = score.argsort(dim=1).argsort(dim=1) < count
    if measured_target is not None:
        too_large = (
            torch.where(fixed, edge - base, 0.0).norm(dim=1, keepdim=True) > desired
        )
        nearest = torch.where(base - low < high - base, low, high)
        cheapest = (nearest - base).abs().argsort(dim=1).argsort(dim=1) < count
        edge = torch.where(too_large, nearest, edge)
        fixed = torch.where(too_large, cheapest, fixed)
    interior = base.clamp(low + margin, high - margin)
    lo, hi = torch.zeros_like(desired), torch.full_like(desired, 8.0)
    for _ in range(15):
        mid = (lo + hi) / 2
        x = torch.where(
            fixed, edge, (interior + mid * direction).clamp(low + margin, high - margin)
        )
        small = (x - base).norm(dim=1, keepdim=True) < desired
        lo, hi = torch.where(small, mid, lo), torch.where(small, hi, mid)
    x = torch.where(
        fixed, edge, (interior + hi * direction).clamp(low + margin, high - margin)
    )
    return (0.5 + torch.logit(x) / 5).clamp(-1, 1)


def search(
    env,
    actor,
    loc,
    seed,
    p,
    counter,
    initial,
    goals,
    hashes,
    commands,
    snap,
    branch_ref,
    routes,
    base_hit,
    base_err,
):
    from .evaluate import append, array, cost

    raw, n, start = env.unwrapped, len(goals), p["branch"]
    for route in p["replay_routes"]:
        lanes = [i for i, r in routes.items() if r == route]
        if not lanes:
            continue
        mask = torch.zeros(n, dtype=torch.bool, device=raw.device)
        mask[lanes] = True
        for candidate in range(p["candidates"]):
            gen = np.random.default_rng(
                np.random.SeedSequence([350001, seed, candidate])
            )
            z, direction = (
                torch.tensor(gen.normal(size=(2, n, k)).astype("f"), device=raw.device)
                for k in (16, 39)
            )
            scale = [0.25, 0.5, 0.8, 1.1][candidate % 4]
            records, stored, measured_targets = {}, {}, {}
            for space in ("K16", "muscle39"):
                for rep in range(p["independent_repeats"]):
                    trace = []
                    with cost(
                        loc,
                        "candidate",
                        seed,
                        counter,
                        space=space,
                        repeat=rep,
                        candidate=candidate,
                        route=route,
                    ):
                        obs = restore_route(
                            raw, route, snap, initial, goals, commands, counter
                        )
                        deviations = errors(
                            branch_ref, state(raw, obs, p["state_atol"])
                        )
                        valid = torch.full_like(
                            mask, rng_hash(rng_state()) == rng_hash(snap.rng)
                        )
                        for k, d in deviations.items():
                            valid &= d <= p["state_atol"][k]
                        active = mask & valid
                        rms, sat = (torch.zeros(n, device=raw.device) for _ in range(2))
                        for j in range(p["horizon"] - start):
                            base = actor(obs)
                            native = env.native(base)
                            corrected, cmd = base, native
                            if j < 10:
                                target = base + scale * z[j // 5]
                                if space == "K16":
                                    corrected = torch.where(
                                        active[:, None], target, base
                                    )
                                    cmd = env.native(corrected)
                                else:
                                    full = muscle_proposal(
                                        native,
                                        env.native(target),
                                        direction[j // 5],
                                        measured_targets.get((rep, j)),
                                    )
                                    cmd = torch.where(active[:, None], full, native)
                            if space == "K16" and rep == 0:
                                values = {
                                    "observation": obs["actor"],
                                    "original_raw": base,
                                    "corrected_raw": corrected,
                                    "original_recruitment": base.relu() * env.scale,
                                    "corrected_recruitment": corrected.relu()
                                    * env.scale,
                                    "qpos": raw.sim.data.qpos,
                                    "qvel": raw.sim.data.qvel,
                                    "activation": raw.sim.data.act,
                                    "prior_excitation": raw.sim.data.ctrl,
                                    "command_excitation": excitation(cmd),
                                    "goal": raw.command_manager.get_command("reach"),
                                }
                                trace.append(
                                    {
                                        k: array(v[lanes]).copy()
                                        for k, v in values.items()
                                    }
                                )
                            obs = step(raw, cmd, start + j + 1, counter)
                            if j < 10:
                                measured = raw.sim.data.ctrl[:]
                                torch.testing.assert_close(
                                    measured, excitation(cmd), rtol=0, atol=2e-6
                                )
                                if space == "K16" and p.get(
                                    "match_recorded_excitation"
                                ):
                                    measured_targets[rep, j] = (
                                        (measured - excitation(native)).norm(
                                            dim=1, keepdim=True
                                        ),
                                        (cmd.abs() >= 0.99999).sum(1, keepdim=True),
                                    )
                                rms += (measured - excitation(native)).square().mean(
                                    1
                                ).sqrt() / 10
                                sat += (cmd.abs() >= 0.99999).float().mean(1) / 10
                        hit, err = native_result(raw)
                        for m, i in enumerate(lanes):
                            records[space, rep, i] = {
                                "seed": seed,
                                "goal": goals[i],
                                "goal_hash": hashes[i],
                                "lane": i,
                                "source_success": int(base_hit[i]),
                                "route": route,
                                "candidate": candidate,
                                "space": space,
                                "repeat": rep,
                                "branch_valid": int(valid[i]),
                                "native_success": int(hit[i]),
                                "source_error_m": float(base_err[i]),
                                "terminal_error_m": float(err[i]),
                                "improvement_m": float(base_err[i] - float(err[i])),
                                "excitation_rms": float(rms[i]),
                                "saturation": float(sat[i]),
                            }
                            if trace and bool(hit[i]) and not base_hit[i]:
                                stored[i] = {
                                    k: np.stack([v[k][m] for v in trace])
                                    for k in trace[0]
                                }
            witnesses = []
            for i in lanes:
                matched = all(
                    records[s, r, i]["branch_valid"]
                    for s in ("K16", "muscle39")
                    for r in (0, 1)
                )
                for r in (0, 1):
                    a, b = (records[s, r, i] for s in ("K16", "muscle39"))
                    matched &= abs(a["excitation_rms"] - b["excitation_rms"]) <= 0.01
                    matched &= abs(a["saturation"] - b["saturation"]) <= 0.05
                for space in ("K16", "muscle39"):
                    a, b = (records[space, r, i] for r in (0, 1))
                    stable = (
                        a["native_success"] == b["native_success"]
                        and abs(a["terminal_error_m"] - b["terminal_error_m"])
                        <= p["terminal_error_atol_m"]
                    )
                    verified = bool(
                        matched and stable and not base_hit[i] and a["native_success"]
                    )
                    for row in (a, b):
                        row.update(
                            paired_calibrated=int(matched),
                            repeat_stable=int(stable),
                            success_witness=int(verified),
                        )
                    if verified:
                        uri, digest = "", ""
                        if space == "K16":
                            path = loc / f"trajectory-{goals[i]}-{candidate}.npz"
                            np.savez_compressed(
                                path,
                                **stored[i],
                                seed=seed,
                                goal_id=goals[i],
                                goal_hash=hashes[i],
                                native_success=True,
                                terminal_error_m=a["terminal_error_m"],
                                source_run=loc.name,
                                intervention_ticks=np.arange(start + 1, start + 11),
                            )
                            uri, digest = path.as_posix(), sha(path)
                        witnesses.append(
                            {
                                "seed": seed,
                                "goal": goals[i],
                                "goal_hash": hashes[i],
                                "space": space,
                                "candidate": candidate,
                                "terminal_error_m": a["terminal_error_m"],
                                "trajectory": uri,
                                "trajectory_sha256": digest,
                            }
                        )
            append(loc, "counterfactual_trials", list(records.values()))
            append(loc, "verified_witnesses", witnesses)
            print(
                "SEARCH", seed, route, candidate + 1, "/", p["candidates"], flush=True
            )
