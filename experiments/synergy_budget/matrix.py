"""Run Reach first, then Pose with identical frozen Reach banks."""

import subprocess
import sys

import numpy as np

from .common import CKPT, LOCK, OUT, read, readcsv, save


def run(mode, task, k, seed, target):
    cmd = [
        sys.executable,
        "-X",
        "utf8",
        "-u",
        "-m",
        "experiments.synergy_budget.ppo",
        mode,
        task,
        str(k),
        str(seed),
        str(target),
    ]
    print(" ".join(cmd[4:]), flush=True)
    folder = CKPT / "logs"
    folder.mkdir(exist_ok=True)
    with (folder / f"{task}-K{k}-s{seed}-{mode}-{target}.log").open("ab") as stream:
        subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, check=True)


def matrix(task):
    arms = list(LOCK["primary_actions"])
    if task == "T2":
        arms = read(OUT / "T1_budget.json")["arms"]

    def stage(ks, target):
        for seed in LOCK["training_seeds"]:
            for k in ks:
                run("train", task, k, seed, target)
                run("validation", task, k, seed, target)

    def avg(k, it, field):
        return float(
            np.mean(
                [
                    read(OUT / "cells" / f"{task}-K{k}-s{s}" / f"validation_{it}.json")[
                        field
                    ]
                    for s in LOCK["training_seeds"]
                ]
            )
        )

    stage(arms, 1000)
    if task == "T1" and avg(39, 1000, "successes") - avg(16, 1000, "successes") > 10:
        rates = {
            int(r["K"]): int(r["successes"])
            for r in readcsv(OUT / "capacity_projection_dev" / "summary.csv")
            if r["nmf_seed"] == "11"
        }
        options = [k for k in (24, 32) if rates[k] - rates[16] >= 15]
        if options:
            extra = min(options)
            arms.append(extra)
            stage([extra], 1000)
    decisions = []
    for k in LOCK["primary_actions"]:
        a, b = avg(k, 500, "successes"), avg(k, 1000, "successes")
        x, y = avg(k, 500, "mean_terminal_error"), avg(k, 1000, "mean_terminal_error")
        decisions.append(dict(K=k, gain_pp=b - a, error_reduction=1 - y / x))
    target = (
        1500
        if any(d["gain_pp"] >= 5 or d["error_reduction"] >= 0.1 for d in decisions)
        else 1000
    )
    data = dict(task=task, arms=arms, iterations=target, decisions=decisions)
    if (OUT / f"{task}_budget.json").exists() and read(
        OUT / f"{task}_budget.json"
    ) != data:
        raise RuntimeError("Frozen evaluation budget changed")
    save(OUT / f"{task}_budget.json", data)
    if target > 1000:
        stage(arms, target)
    for k in arms:
        for seed in LOCK["training_seeds"]:
            run("test", task, k, seed, target)
    save(OUT / f"{task}_complete.json", data)


if __name__ == "__main__":
    for task in LOCK["tasks"]:
        if not (OUT / f"{task}_complete.json").exists():
            matrix(task)
    subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8",
            "-m",
            "experiments.synergy_budget.capacity",
            "test",
        ],
        check=True,
    )
