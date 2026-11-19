import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.rsi_v02a import publish
from experiments.rsi_v02a.registry import ROOT, completed, read
from experiments.rsi_v02a.report import cohort


class PublishedRepairEvidence(unittest.TestCase):
    def test_partial_report_preserves_missing_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch.object(publish, "ROOT", Path(tmp)),
                patch.object(publish, "ready", return_value=["B1-final/47"]),
            ):
                publish.publish()
            self.assertEqual(read(Path(tmp) / "verification.json")["status"], "PARTIAL")
            self.assertIn(
                "B1-final/47",
                (Path(tmp) / "RESULTS_REPORT.md").read_text(encoding="utf8"),
            )

    def test_reject_incomplete_final(self):
        with self.assertRaises(ValueError):
            cohort([], "final")

    @unittest.skipUnless(
        (ROOT / "replay_audit.csv").exists(),
        "No measured diagnostic release in this checkout",
    )
    def test_witness_requires_matching_control_and_real_success(self):
        for seed in (11, 29, 47):
            folder = completed("diagnose", seed)
            with (folder / "counterfactual_trials.csv").open() as stream:
                trials = list(csv.DictReader(stream))
            self.assertEqual(len(trials), 2000)
            for g in sorted({r["goal"] for r in trials}):
                with np.load(folder / f"search_{g}.npz") as archive:
                    terminal = archive["error"][-1]
                    matched = archive["match"]
                for r in (t for t in trials if t["goal"] == g):
                    offset = 0 if r["space"] == "K16" else 200
                    i = offset + int(r["candidate"])
                    self.assertAlmostEqual(
                        float(terminal[i]), float(r["terminal_error"])
                    )
                    if r["matched"] == "True":
                        self.assertTrue(matched[i] and matched[offset])
                    if r["label"] == "success_witness":
                        self.assertEqual(r["matched"], "True")
                        self.assertEqual(int(r["success"]), 1)
                        self.assertLess(float(terminal[i]), 0.0625)
