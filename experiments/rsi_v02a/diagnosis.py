"""Matched prefix replay, local finite differences and budgeted random shooting."""

import time
from collections import defaultdict

import numpy as np
import torch

from .measurement import controller, cpu, episodes, metadata, reset, tips
from .registry import PROTOCOL, attempt, cohorts, completed, event, read, table, write
from .weakness import active_gate


def selected(rows):
    bad = sorted(
        (i for i, r in enumerate(rows) if not r["success"]),
        key=lambda i: (rows[i]["error"], rows[i]["goal"]),
    )
    good = sorted(
        (i for i, r in enumerate(rows) if r["success"]),
        key=lambda i: (rows[i]["error"], rows[i]["goal"]),
    )
    if len(bad) < 4 or not good:
        raise ValueError(
            "Insufficient failures or successful controls for the registered diagnostic panel"
        )
    return [bad[int(len(bad) * p)] for p in (0.125, 0.375, 0.625, 0.875)] + [
        good[len(good) // 2]
    ]


def fulltrace(loc, seed):
    goals = cohorts()["diagnosis"]
    with controller(active_gate()[0], seed, len(goals)) as (env, runner):
        m = metadata(env)
        rows = episodes(env, runner, goals, loc, "failure_traces", record=True)
    table(loc / "failure_episodes.csv", rows)
    with np.load(loc / "failure_traces.npz") as f:
        data = {k: f[k] for k in f.files}
    errors = np.linalg.norm(data["finger"], axis=2)
    keys = selected(rows)
    panel = []
    for i, r in enumerate(rows):
        dom = int(np.argmax(data["finger"][-1, i]))
        rise = np.flatnonzero(errors[5:, i] - errors[:-5, i] > 0.003)
        stalled = np.flatnonzero(np.abs(errors[10:, i] - errors[:-10, i]) < 0.001)
        panel.append(
            dict(
                goal=r["goal"],
                success=r["success"],
                terminal_error=r["error"],
                worst_tip=dom + 1,
                first_rise=int(rise[0] + 6) if len(rise) else None,
                first_plateau=int(stalled[0] + 11) if len(stalled) else None,
                near_failure=bool(not r["success"] and r["error"] < 0.075),
                selected=i in keys,
                branch_step=30 + 10 * keys.index(i) if i in keys else None,
            )
        )
    table(loc / "failure_manifest.csv", panel)
    groups = defaultdict(list)
    for r in panel:
        groups[(r["success"], r["worst_tip"])].append(r)
    table(
        loc / "failure_clusters.csv",
        [
            dict(success=s, worst_tip=j, count=len(v), fraction=len(v) / len(rows))
            for (s, j), v in sorted(groups.items())
        ],
    )
    return rows, data, keys, m


@torch.inference_mode()
def prefix(env, goal, data, lane, start, n):
    raw = env.unwrapped
    # Historical cross-lane diagnostic. New capability and BC evidence uses the
    # same-lane no-op gate in counterfactual_validation/evaluate.py.
    obs = reset(raw, [goal] * n)
    for t in range(start):
        cmd = (
            torch.tensor(data["native"][t, lane], device=env.device)
            .expand(n, -1)
            .contiguous()
        )
        obs, *_ = raw.step(cmd)
    checks = {}
    valid = np.ones(n, dtype=bool)
    for name, tol in (
        ("qpos", 1e-4),
        ("qvel", 1e-3),
        ("ctrl", 1e-4),
        ("act", 1e-4),
        ("goal", 0),
        ("qacc_warmstart", 0.1),
        ("time", 1e-5),
        ("actuator_length", 1e-4),
        ("actuator_velocity", 1e-3),
    ):
        if name not in data:
            raise ValueError(f"Missing required archived replay field {name}")
        actual = (
            raw.command_manager.get_command("reach")
            if name == "goal"
            else getattr(raw.sim.data, name)
        )
        delta = np.abs(cpu(actual) - data[name][start - 1, lane]).reshape(n, -1).max(1)
        checks[name] = float(delta.max())
        valid &= delta <= tol
    delta = np.abs(cpu(obs["actor"]) - data["obs"][start, lane]).reshape(n, -1).max(1)
    checks["obs"] = float(delta.max())
    valid &= delta <= 1e-3
    return obs, valid, checks


@torch.inference_mode()
def profile(env, runner, seed, goal, data, lane, start, loc):
    n = env.num_envs
    tic = time.perf_counter()
    raw = env.unwrapped
    obs, valid, checks = prefix(env, goal, data, lane, start, n)
    # Hold matched base raw mean constant. Identical reset+prefix replicates
    # quantify numerical variability, not independent animals or policy seeds.
    base = runner.alg.get_policy()(obs)[0].clone()
    states = [(j, d, sign) for d in (0.1, 0.25) for j in range(16) for sign in (-1, 1)]
    design = [(-1, 0.0, 0)] + states + states[:32] + [(-1, 0.0, 0)] * 3
    val = base.relu().expand(n, -1).clone()
    for i, (j, d, sgn) in enumerate(design):
        if j >= 0:
            val[i, j] += sgn * d
    lower = cpu((val < 0).any(1))
    pre = (val.relu() * env.scale) @ env.W.T
    from myo_rsi.bank_study import HIGH, LOW

    saturated = cpu(((pre < LOW) | (pre > HIGH)).any(1))
    native = env.native(val)
    profiles = []
    effects = {}
    for horizon in range(1, 11):
        obs, *_ = raw.step(native)
        if horizon not in (1, 5, 10):
            continue
        pos = cpu(tips(raw)).reshape(n, 15)
        aim = cpu(raw.command_manager.get_command("reach"))
        er = np.linalg.norm(pos - aim, axis=1)
        q = cpu(raw.sim.data.qpos)
        act = cpu(raw.sim.data.act)
        for j, d, _ in states[::2]:
            for rep in range(2 if d == 0.1 else 1):
                ids = [i for i, x in enumerate(design) if x == (j, d, -1)]
                lo = ids[rep]
                hi = lo + 1
                step = 2 * d * float(env.scale[j])
                valid_fd = bool(valid[lo] and valid[hi])
                xyz = (pos[hi] - pos[lo]) / step
                effects[f"{horizon}-{j}-{d}-{rep}"] = np.concatenate(
                    [xyz, (q[hi] - q[lo]) / step, (act[hi] - act[lo]) / step]
                )
                profiles.append(
                    dict(
                        goal=goal,
                        phase=start,
                        horizon=horizon,
                        component=j + 1,
                        delta_q95=d,
                        repeat=rep,
                        matched=valid_fd,
                        clipped=bool(
                            lower[lo] or lower[hi] or saturated[lo] or saturated[hi]
                        ),
                        tip_response_norm=float(np.linalg.norm(xyz)),
                        error_slope=float((er[hi] - er[lo]) / step),
                        joint_response=float(np.linalg.norm(q[hi] - q[lo]) / step),
                        activation_response=float(
                            np.linalg.norm(act[hi] - act[lo]) / step
                        ),
                    )
                )
    np.savez_compressed(loc / f"responses_{goal}.npz", **effects)
    event(loc, "profile_prefix", n * start)
    event(loc, "functional_profile", n * 10, time.perf_counter() - tic)
    return profiles, checks


@torch.inference_mode()
def shooting(env, runner, goal, seed, data, lane, start, loc, source_success):
    tic = time.perf_counter()
    n = env.num_envs
    # Preserve the published group-control estimator for historical reproduction.
    # Its proposals do not qualify for the CF verified correction dataset.
    half = n // 2
    rng = np.random.default_rng(np.random.SeedSequence([400000, seed, goal]))
    raw = env.unwrapped
    obs, match, checks = prefix(env, goal, data, lane, start, n)
    actor = runner.alg.get_policy()
    noise = []
    for dim, scale in [(16, 0.65), (39, 0.28)]:
        z = rng.normal(0, scale, (2, half, dim)).astype("f")
        z[:, :2] = 0
        noise.append(torch.tensor(z, device=env.device))
    samples = {k: [] for k in ["obs", "latent", "excitation_delta", "error"]}
    saturation = np.zeros(n)
    native_energy = np.zeros(n)
    for t in range(start, 100):
        base = actor(obs)
        native = env.native(base)
        if t < start + 10:
            b = (t - start) // 5
            fixed = base[:half] + noise[0][b]
            cmd = torch.cat(
                [env.native(fixed), (native[half:] + noise[1][b]).clamp(-1, 1)]
            )
            samples["obs"].append(cpu(obs["actor"]))
            samples["latent"].append(cpu(fixed))
            samples["excitation_delta"].append(
                cpu(
                    torch.sigmoid(5 * (cmd[half:] - 0.5))
                    - torch.sigmoid(5 * (native[half:] - 0.5))
                )
            )
            saturation += cpu((cmd.abs() >= 0.99999).float().mean(1))
            native_energy += cpu((cmd - native).square().mean(1))
        else:
            cmd = native
        obs, _, term, trunc, _ = raw.step(cmd)
        if bool((term | trunc).any()) != (t == 99):
            raise ValueError("Search continuation terminal mismatch")
        err = cpu(
            -raw.reward_manager._step_reward[
                :, raw.reward_manager.active_terms.index("reach")
            ]
            / raw.reward_manager.cfg["reach"].weight
        )
        samples["error"].append(err)
        if t == start + 9:
            short = err.copy()
    success = cpu(
        raw.metrics_manager._step_values[
            :, raw.metrics_manager.active_terms.index("success")
        ]
    ).astype(int)
    np.testing.assert_array_equal(success, err < 0.0625)
    trials = []
    bc = None
    direction = None
    for offset, space in ((0, "K16"), (half, "muscle39")):
        reference = offset
        control_ok = bool(
            match[reference]
            and success[reference] == source_success
            and abs(err[reference] - np.linalg.norm(data["finger"][-1, lane])) <= 0.001
        )
        candidates = []
        for i in range(offset, offset + half):
            ok = bool(match[i] and control_ok)
            witness = ok and not source_success and bool(success[i])
            promising = (
                ok
                and short[i] < short[reference] - 0.001
                and err[i] < err[reference] - 0.001
            )
            label = (
                "invalid"
                if not ok
                else "success_witness"
                if witness
                else "promising_correction"
                if promising
                else "unresolved"
            )
            score = float(
                short[i] + 0.001 * saturation[i] / 10 + 0.0001 * native_energy[i] / 10
            )
            trials.append(
                dict(
                    goal=goal,
                    phase=start,
                    space=space,
                    candidate=i - offset,
                    matched=ok,
                    label=label,
                    short_error=float(short[i]),
                    terminal_error=float(err[i]),
                    success=int(success[i]),
                    score=score,
                    correction_mse=float(native_energy[i] / 10),
                    saturation=float(saturation[i] / 10),
                )
            )
            if not source_success and label in (
                "success_witness",
                "promising_correction",
            ):
                candidates.append((not witness, float(err[i]), score, i))
        if candidates:
            best = min(candidates)[-1]
            if space == "K16":
                bc = (
                    np.array(samples["obs"])[:, best],
                    np.array(samples["latent"])[:, best],
                    goal,
                )
            elif success[best]:
                direction = np.array(samples["excitation_delta"])[:, best - half].mean(
                    0
                )
    np.savez_compressed(
        loc / f"search_{goal}.npz",
        error=np.array(samples["error"]),
        raw_noise=cpu(noise[0]),
        native_noise=cpu(noise[1]),
        match=match,
    )
    event(loc, "search_prefix", n * start)
    event(loc, "search_candidates", n * 10)
    event(loc, "search_continuation", n * (90 - start), time.perf_counter() - tic)
    return trials, bc, direction, checks


def diagnose(seed):
    site, gate = active_gate()
    if not all(gate[k] for k in ("competent", "degradation", "viable")):
        raise ValueError("G0 not qualified")
    with attempt("diagnose", seed) as (loc, m):
        rows, data, keys, m["environment"] = fulltrace(loc, seed)
        profiles, trials, targets, directions = [], [], [], []
        checks = {}
        for mode, n in [("profile", 100), ("search", 400)]:
            with controller(active_gate()[0], seed, n) as (env, runner):
                env.unwrapped.cfg.auto_reset = False
                runner.alg.get_policy().eval()
                for j, i in enumerate(keys):
                    g = rows[i]["goal"]
                    start = 30 + 10 * j
                    if mode == "profile":
                        rs, cmp = profile(env, runner, seed, g, data, i, start, loc)
                        profiles.extend(rs)
                    else:
                        rs, bc, vec, cmp = shooting(
                            env,
                            runner,
                            g,
                            seed,
                            data,
                            i,
                            start,
                            loc,
                            rows[i]["success"],
                        )
                        trials.extend(rs)
                        if bc is not None:
                            targets.append(bc)
                        if vec is not None:
                            directions.append(vec)
                    checks[f"{mode}_{g}"] = cmp
        table(loc / "functional_profiles.csv", profiles)
        table(loc / "counterfactual_trials.csv", trials)
        np.savez_compressed(
            loc / "corrections.npz",
            obs=np.array([x[0] for x in targets]),
            raw=np.array([x[1] for x in targets]),
            goals=[x[2] for x in targets],
            full_directions=np.array(directions),
        )
        write(
            loc / "diagnostic_report.json",
            dict(
                matches=checks,
                rows=len(trials),
                eligible_goals=[x[2] for x in targets],
                per_case=[
                    dict(
                        goal=rows[i]["goal"],
                        space=space,
                        **{
                            label: sum(
                                t["goal"] == rows[i]["goal"]
                                and t["space"] == space
                                and t["label"] == label
                                for t in trials
                            )
                            for label in (
                                "success_witness",
                                "promising_correction",
                                "invalid",
                                "unresolved",
                            )
                        },
                    )
                    for i in keys
                    for space in ("K16", "muscle39")
                ],
            ),
        )
        print("DIAGNOSIS", seed, "BC goals", len(targets), flush=True)


if __name__ == "__main__":
    for s in read(PROTOCOL)["source_seeds"]:
        try:
            completed("diagnose", s)
        except FileNotFoundError:
            diagnose(s)
