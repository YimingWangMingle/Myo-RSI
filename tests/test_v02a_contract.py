import tempfile
import unittest
from pathlib import Path

import torch

from experiments.rsi_v02a.adapters import ResidualMean
from experiments.rsi_v02a.registry import allocate, cohorts, source


class StudyContract(unittest.TestCase):
    def test_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertNotEqual(
                allocate("screen", 11, Path(tmp)), allocate("screen", 11, Path(tmp))
            )

    def test_splits_and_sources(self):
        self.assertEqual(len(cohorts()["final"]), 200)
        for s in (11, 29, 47):
            self.assertEqual(len(source(s)["sha256"]), 64)
        with self.assertRaises(ValueError):
            source(93)

    def test_residual_initialization(self):
        base = torch.nn.Linear(115, 16)
        base.requires_grad_(False)
        net = ResidualMean(base, 115, 16)
        obs = torch.randn(8, 115)
        torch.testing.assert_close(net(obs), base(obs), atol=0, rtol=0)
        net(obs).sum().backward()
        self.assertGreater(net.delta[-1].weight.grad.abs().sum().item(), 0)
        self.assertIsNone(base.weight.grad)
