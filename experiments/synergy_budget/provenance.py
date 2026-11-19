"""Archive registered environment settings and the immutable study inputs."""

from dataclasses import asdict
from importlib import metadata

import myosuite.envs.myo.backends.mjlab  # noqa: F401
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.os import dump_yaml

from myo_rsi.protocol import TASKS
from .common import HERE, LOCK, OLD, OUT, read, save, sha


def capture():
    directory = OUT / "config"
    directory.mkdir(exist_ok=True)
    for task in LOCK["tasks"]:
        cfg = load_env_cfg(TASKS[task])
        cfg.scene.num_envs = LOCK["n_envs"]
        cfg.seed = LOCK["training_seeds"][0]
        dump_yaml(directory / f"{task}_training.yaml", asdict(cfg))
    save(
        directory / "sources.json",
        dict(
            base_commit=LOCK["base_commit"],
            protocol_sha256=sha((HERE / "protocol.json").read_bytes()),
            study_source_sha256={
                p.name: sha(p.read_bytes()) for p in sorted(HERE.glob("*.py"))
            },
            environment_yaml_sha256={
                p.name: sha(p.read_bytes()) for p in sorted(directory.glob("*.yaml"))
            },
            frozen_teacher_sources=read(OLD / "teacher_manifest.json"),
            historical_data=read(OLD / "dataset_manifest.json"),
            environment_notes="Training YAML is the registered seed-11 configuration. Each other cell overrides only seed. Evaluation uses 100 lanes, explicit manual terminal reset, and per-target reset seeds.",
            analysis_versions={
                p: metadata.version(p)
                for p in [
                    "myosuite",
                    "mjlab",
                    "mujoco",
                    "mujoco-warp",
                    "rsl-rl-lib",
                    "torch",
                    "numpy",
                    "scipy",
                    "scikit-learn",
                    "threadpoolctl",
                    "matplotlib",
                ]
            },
        ),
    )


if __name__ == "__main__":
    capture()
