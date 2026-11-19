"""Dispatch the registered stages. Completed attempts are reused on restart."""

import subprocess
import sys

from .registry import ROOT, read


def stage(name, *args):
    subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8",
            "-u",
            "-m",
            f"experiments.rsi_v02a.{name}",
            *args,
        ],
        check=True,
    )


def main():
    stage("screen")
    gate = read(ROOT / "screen_gate.json")
    if gate["competent"] and not gate["degradation"]:
        stage("weakness")
        gate = read(ROOT / "S2_gate.json")
    if not all(gate[k] for k in ("competent", "degradation", "viable")):
        stage("structure")
        stage("publish")
        raise SystemExit("The registered body screen did not qualify for adaptation.")
    stage("diagnosis")
    stage("adapt", "train")
    stage("structure")
    stage("adapt", "validation")
    stage("adapt", "final")
    stage("publish")


if __name__ == "__main__":
    main()
