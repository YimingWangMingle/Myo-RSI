"""Replay the published tables without local training outputs or simulator runs."""

import hashlib
import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.make_figures import build

PROJECT = Path(__file__).resolve().parents[1]


class PublishedResultsTests(unittest.TestCase):
    @unittest.skipUnless(
        (PROJECT / "results/verification.json").exists(),
        "Complete published matrix unavailable",
    )
    def test_csv_only_reproduction(self):
        original = PROJECT / "results"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            exported = root / "results"
            shutil.copytree(original / "tables", exported / "tables")
            for p in [*original.glob("exp_*.csv"), original / "preflight.json"]:
                shutil.copy2(p, exported / p.name)
            shutil.copytree(PROJECT / "artifacts", root / "artifacts")
            names = build(exported)
            self.assertEqual(len(names), 7)
            pd.testing.assert_frame_equal(
                pd.read_csv(original / "summary.csv"),
                pd.read_csv(exported / "summary.csv"),
            )
            for name in names:
                for ext in ("png", "pdf"):
                    self.assertGreater(
                        (exported / "figures" / f"{name}.{ext}").stat().st_size, 1000
                    )
                # PNG is deterministic in the pinned plotting stack. PDF metadata
                # contains a creation timestamp, so a bytewise PDF match is unsuitable.
                a = (original / "figures" / f"{name}.png").read_bytes()
                b = (exported / "figures" / f"{name}.png").read_bytes()
                self.assertEqual(
                    hashlib.sha256(a).digest(), hashlib.sha256(b).digest(), name
                )
