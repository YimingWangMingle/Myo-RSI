import unittest

import numpy as np

from myo_rsi.actions import excitation, native_action
from myo_rsi.protocol import (
    FINAL_SEEDS,
    PROBE_SEED,
    TASKS,
    TRAIN_SEEDS,
    VALIDATION_SEEDS,
    K,
)


class ProtocolTests(unittest.TestCase):
    def test_native_roundtrip(self):
        actions = np.linspace(-1, 1, 257)
        recovered, distortion = native_action(excitation(actions))
        np.testing.assert_allclose(recovered, actions, atol=1e-12)
        np.testing.assert_array_equal(distortion, 0)

    def test_projection_reports_unreachable_excitation(self):
        desired = np.array([-3, 0, 0.5, 1, 3])
        action, error = native_action(desired)
        self.assertTrue(np.all(np.abs(action) <= 1))
        np.testing.assert_allclose(excitation(action), desired + error)
        self.assertGreater(error[1], 0)
        self.assertLess(error[-2], 0)
        for value in (float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                native_action(value)

    def test_disjoint_seeds(self):
        self.assertEqual(len(FINAL_SEEDS), 100)
        self.assertEqual(len(set(TRAIN_SEEDS)), 3)
        self.assertTrue(
            set(FINAL_SEEDS).isdisjoint((*TRAIN_SEEDS, *VALIDATION_SEEDS, PROBE_SEED))
        )
        self.assertEqual(K, 8)
        self.assertEqual(len(set(TASKS.values())), 4)
