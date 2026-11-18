import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from myo_rsi.tables import wilson, write_csv


class ResultTests(unittest.TestCase):
    def test_individual_wilson_intervals(self):
        lo, hi = wilson(0, 100)
        self.assertAlmostEqual(lo, 0)
        self.assertAlmostEqual(hi, 0.0369935, places=6)
        lo, hi = wilson(50, 100)
        self.assertAlmostEqual(lo, 1 - hi)
        self.assertLess(lo, 0.5)

    def test_missing_measurement_stays_empty(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "status.csv"
            write_csv(
                path,
                "method status success_rate",
                [{"method": "C0 Frozen", "status": "NOT_RUN", "success_rate": None}],
            )
            self.assertEqual(path.read_text().splitlines()[1], "C0 Frozen,NOT_RUN,")
