"""Run the preregistered matrix serially on one GPU, resuming completed cells."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from .protocol import TRAIN_SEEDS


def stage(method, task):
    from .train import folder

    logs = Path(".runtime/matrix")
    logs.mkdir(parents=True, exist_ok=True)
    for seed in TRAIN_SEEDS:
        out = folder(task, method, seed)
        meta = out / "run.json"
        completed = (
            meta.exists()
            and json.loads(meta.read_text(encoding="utf8"))["status"] == "COMPLETE"
        )
        actions = ["evaluate"] if completed else ["train", "evaluate"]
        for action in actions:
            log = logs / f"{task}-{method}-{seed}-{action}.log"
            print(action, task, method, seed, flush=True)
            with log.open("a", encoding="utf8") as f:
                subprocess.run(
                    [
                        sys.executable,
                        "-X",
                        "utf8",
                        "-u",
                        "-m",
                        f"myo_rsi.{action}",
                        task,
                        method,
                        str(seed),
                    ],
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    check=True,
                )


def validated(method):
    from .train import folder

    initial, final = [], []
    for seed in TRAIN_SEEDS:
        out = folder("T1", method, seed)
        items = json.loads((out / "eval_T1.json").read_text(encoding="utf8"))["points"]
        curve = [p for p in items if p["split"] == "validation"]
        rates = [
            sum(e["success"] for e in p["episodes"]) / len(p["episodes"]) for p in curve
        ]
        initial.append(rates[0])
        final.append(rates[-1])
    if sum(final) <= sum(initial) or sum(final) == 0:
        raise RuntimeError(
            f"Reach {method} learning gate failed on validation: {initial} -> {final}"
        )


def main():
    from .bank import fit
    from .train import BANK

    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["all", "A", "B", "C"], default="all")
    args = p.parse_args()
    if args.stage in ("A", "all"):
        stage("A", "T1")
        stage("A", "T2")
        validated("A")
    if args.stage in ("B", "all"):
        validated("A")
        if not BANK.exists():
            tic = time.perf_counter()
            files = sorted(
                Path("datasets/direct_rollouts/reach_random_train").glob("T1-A-*.npz")
            )
            info = fit(files, BANK)
            (BANK.parent / "nmf.json").write_text(
                json.dumps(dict(bank=info, wall_s=time.perf_counter() - tic), indent=2),
                encoding="utf8",
            )
        stage("B", "T1")
        stage("B", "T2")
        validated("B")
    if args.stage in ("C", "all"):
        validated("B")
        for method in ("C0", "C1", "C2"):
            stage(method, "T3")


if __name__ == "__main__":
    main()
