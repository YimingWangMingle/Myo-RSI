import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from myo_rsi.bank import fit, load_bank
from myo_rsi.protocol import TASKS, TRAIN_SEEDS


class BankTests(unittest.TestCase):
    def test_training_only_and_order(self):
        # Temporary synthetic unit-test fixtures never enter results or rollout datasets.
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            paths = []
            names = [f"test_muscle_{i}" for i in range(39)]
            for seed in TRAIN_SEEDS:
                meta = {
                    "task_id": "T1",
                    "env_id": TASKS["T1"],
                    "method": "Direct PPO",
                    "split": "train",
                    "signal": "data.ctrl",
                    "model_hash": "fixture",
                    "actuator_names": names,
                    "run_id": f"fixture-{seed}",
                    "seed": seed,
                }
                path = folder / f"{seed}.npz"
                np.savez(
                    path,
                    metadata=json.dumps(meta),
                    episode_id=np.repeat(np.arange(6), 10),
                    timestep=np.tile(np.arange(10), 6),
                    physical_excitation=np.random.default_rng(seed).uniform(
                        0.01, 0.9, (60, 39)
                    ),
                )
                paths.append(path)
            target = folder / "bank.npz"
            info = fit(paths, target)
            weights, _ = load_bank(target, names)
            self.assertEqual(weights.shape, (39, 8))
            self.assertFalse(weights.flags.writeable)
            self.assertEqual(
                info["train"]["samples"] + info["validation"]["samples"], 180
            )
            with self.assertRaises(FileExistsError):
                fit(paths, target)
            with self.assertRaises(ValueError):
                load_bank(target, names[::-1])
            with np.load(paths[0]) as src:
                data = dict(src)
            wrong = json.loads(data["metadata"].item())
            for key, value in [
                ("split", "eval"),
                ("task_id", "T2"),
                ("signal", "raw_action"),
            ]:
                data["metadata"] = json.dumps(dict(wrong, **{key: value}))
                np.savez(paths[0], **data)
                with self.assertRaises(ValueError):
                    fit(paths, folder / "rejected.npz")
