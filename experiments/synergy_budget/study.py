"""End-to-end entry point; existing scientific artifacts are validated or resumed."""

import subprocess
import sys

from .common import OUT


def run(module, *args):
    subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8",
            "-u",
            "-m",
            "experiments.synergy_budget." + module,
            *args,
        ],
        check=True,
    )


if __name__ == "__main__":
    run("provenance")
    run("capacity", "identity")
    run("capacity", "dev")
    if not (OUT / "divergence_onset.csv").exists():
        run("onset")
    run("matrix")
    run("pose_reference")
    run("report")
