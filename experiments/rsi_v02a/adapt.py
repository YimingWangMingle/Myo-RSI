"""Source-matched adaptation and independent, frozen-policy evaluations."""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .adapters import install
from .measurement import controller, episodes, metadata, td
from .registry import (
    PROTOCOL,
    ROOT,
    attempt,
    cohorts,
    completed,
    event,
    read,
    sha,
    write,
)
from .weakness import active_gate


def frozen_digest(actor):
    import hashlib

    digest = hashlib.sha256()
    for name, v in actor.state_dict().items():
        if not name.startswith("mlp.delta."):
            digest.update(name.encode())
            digest.update(v.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def bc_fit(actor, file, loc):
    with np.load(file) as z:
        goals, obs, target = z["goals"], z["obs"], z["raw"]
    if (
        len(np.unique(goals)) < 2
        or obs.shape[:2] != target.shape[:2]
        or target.shape[-1] != 16
    ):
        raise ValueError("No adequate independent correction cohorts for BC")
    device = next(actor.parameters()).device
    x = torch.tensor(obs, device=device)
    y = torch.tensor(target, device=device)
    optim = torch.optim.Adam(actor.mlp.delta.parameters(), lr=0.001)

    def loss(obs, raw):
        a = obs.flatten(0, 1)
        return torch.nn.functional.mse_loss(actor(td(a)), raw.flatten(0, 1))

    tic = time.perf_counter()
    with torch.no_grad():
        initial = float(loss(x[-1:], y[-1:]))
    for _ in range(200):
        optim.zero_grad()
        err = loss(x[:-1], y[:-1])
        err.backward()
        optim.step()
    with torch.no_grad():
        out = dict(
            train_goals=goals[:-1].tolist(),
            validation_goal=int(goals[-1]),
            train_pairs=int(y[:-1].numel() // 16),
            validation_pairs=len(y[-1]),
            validation_initial_mse=initial,
            validation_final_mse=float(loss(x[-1:], y[-1:])),
            train_mse=float(loss(x[:-1], y[:-1])),
            gradient_steps=200,
            source_sha256=sha(file),
        )
    write(loc / "bc.json", out)
    event(loc, "bc_compute", 0, time.perf_counter() - tic, device="cuda:0")


def check_budget(method, diagnosis, ppo):
    if method == "B0":
        if diagnosis or ppo:
            raise ValueError("Frozen controller has training cost")
    elif method in ("B1", "B2", "B3"):
        if (
            diagnosis + ppo != 1200000
            or (method == "B3" and diagnosis != 240000)
            or (method != "B3" and diagnosis)
        ):
            raise ValueError("Unmatched total budget")
    else:
        raise ValueError("Unknown arm")


def train(seed, method):
    site, gate = active_gate()
    if not all(gate[k] for k in ("competent", "degradation", "viable")):
        raise ValueError("Body-screening gate is closed")
    with attempt(method, seed) as (loc, m):
        cost = 0
        if method == "B0":
            with controller(site, seed, 1) as (env, runner):
                m["environment"] = metadata(env)
            m["final_checkpoint"] = m["source_checkpoint"]
            check_budget(method, 0, 0)
            return
        if method == "B3":
            diag = completed("diagnose", seed)
            cost = sum(read(diag / "manifest.json")["budget"].values())
            check_budget(method, cost, 960000)
            m["diagnostic_parent"] = diag.name
            event(
                loc,
                "attributed_diagnosis",
                cost,
                shared_run=diag.name,
                charge_only=True,
            )
            if len(read(diag / "diagnostic_report.json")["eligible_goals"]) < 2:
                write(
                    loc / "blocked.json",
                    dict(
                        reason="Fewer than two independent eligible BC goals",
                        source=diag.name,
                    ),
                )
                raise ValueError("Insufficient verified corrections for B3")
        ticks = (1200000 - cost) // 12000
        ck = Path("checkpoints/rsi_v02a") / loc.name
        ck.mkdir(parents=True, exist_ok=False)
        with controller(site, seed, 500) as (env, runner):
            m["environment"] = metadata(env)
            actor = runner.alg.get_policy()
            actor.obs_normalization = False
            torch.manual_seed(200000 + seed)
            if method in ("B2", "B3"):
                actor = install(runner)
            invariant = frozen_digest(actor)
            if method == "B3":
                bc_fit(actor, diag / "corrections.npz", loc)
            env.unwrapped.reset(seed=200000 + seed)
            torch.manual_seed(200000 + seed)
            history = []

            def checkpoint(it):
                torch.cuda.synchronize()
                if method in ("B2", "B3") and frozen_digest(actor) != invariant:
                    raise ValueError(
                        "Frozen source weights/normalization/variance changed"
                    )
                path = ck / f"policy_{it}.pt"
                data = runner.alg.save()
                data.update(
                    iter=it,
                    infos=dict(
                        method=method,
                        seed=seed,
                        bank_hash=env.bank_hash,
                        parent_sha256=m["source_checkpoint"]["sha256"],
                        protocol_sha256=m["protocol_sha256"],
                    ),
                    torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state(),
                )
                tmp = path.with_suffix(".tmp")
                torch.save(data, tmp)
                tmp.replace(path)
                history.append(
                    dict(
                        iteration=it,
                        ppo=it * 12000,
                        diagnosis=cost,
                        total=it * 12000 + cost,
                        uri=path.as_posix(),
                        sha256=sha(path),
                    )
                )
                write(loc / "checkpoints.json", history)

            checkpoint(0)
            previous = time.perf_counter()

            def monitor(**kw):
                nonlocal previous
                it = kw["it"] + 1
                event(
                    loc,
                    "ppo_update",
                    12000,
                    time.perf_counter() - previous,
                    iteration=it,
                    learning_rate=float(kw["learning_rate"]),
                )
                previous = time.perf_counter()
                if it * 12000 + cost in (240000, 600000, 1200000):
                    checkpoint(it)

            runner.logger.log = monitor
            runner.learn(ticks, init_at_random_ep_len=True)
            if history[-1]["iteration"] != ticks:
                raise ValueError("Incomplete PPO run")
            check_budget(method, cost, ticks * 12000)
            m["final_checkpoint"] = history[-1]
            print("TRAIN", method, seed, history[-1]["total"], flush=True)


def evaluate(seed, method, group):
    trainloc = completed(method, seed)
    parent = read(trainloc / "manifest.json")
    if group == "final" and not (ROOT / "final_lock.json").exists():
        raise ValueError("Lock all eligible checkpoints first")
    with attempt(method + "-" + group, seed, {"trained_run": trainloc.name}) as (
        loc,
        m,
    ):
        goals = cohorts()[group]
        points = (
            [dict(iteration=0, ppo=0, total=0, **parent["source_checkpoint"])]
            if method == "B0"
            else read(trainloc / "checkpoints.json")
        )
        if group == "final":
            points = points[-1:]
            if (
                read(ROOT / "final_lock.json")["arms"][f"{method}-{seed}"]
                != points[0]["sha256"]
            ):
                raise ValueError("Final checkpoint differs from lock")
        elif method != "B0":
            points = [p for p in points if p["total"] > 0]
        meta = {}
        for task in ("T1", active_gate()[0]):
            with controller(task, seed, len(goals)) as (env, runner):
                meta[task] = metadata(env)
                if method in ("B2", "B3"):
                    install(runner)
                for p in points:
                    if sha(p["uri"]) != p["sha256"]:
                        raise ValueError("Evaluation checkpoint changed")
                    runner.load(p["uri"], load_cfg={"actor": True})
                    tag = f"{task}_{p['total']}"
                    episodes(env, runner, goals, loc, tag, exemplar=group == "final")
        m["environment"] = meta
        write(loc / "evaluated.json", points)


def already(method, seed):
    try:
        completed(method, seed)
        return True
    except FileNotFoundError:
        return False


def eligible(seed):
    result = ["B0", "B1", "B2"]
    try:
        diag = completed("diagnose", seed)
    except FileNotFoundError:
        return result
    if len(read(diag / "diagnostic_report.json")["eligible_goals"]) >= 2:
        result.append("B3")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["base", "train", "validation", "final"])
    args = p.parse_args()
    seeds = read(PROTOCOL)["source_seeds"]
    if args.stage == "final":
        lock = {
            f"{a}-{s}": read(completed(a, s) / "manifest.json")["final_checkpoint"][
                "sha256"
            ]
            for s in seeds
            for a in eligible(s)
        }
        for s in seeds:
            for a in eligible(s):
                completed(a + "-validation", s)
        f = ROOT / "final_lock.json"
        if f.exists():
            if read(f)["arms"] != lock:
                raise ValueError("Final lock immutable")
        else:
            write(f, dict(arms=lock, cohort=cohorts()["final"]))
    for s in seeds:
        for a in ["B0"] if args.stage == "base" else eligible(s):
            if args.stage in ("base", "train"):
                if not already(a, s):
                    train(s, a)
                if a == "B0" and not already("B0-validation", s):
                    evaluate(s, a, "validation")
            elif not already(a + "-" + args.stage, s):
                evaluate(s, a, args.stage)
