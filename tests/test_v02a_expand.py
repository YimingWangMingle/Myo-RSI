import torch
import unittest

from experiments.rsi_v02a.structure import physical


class ExpandGradient(unittest.TestCase):
    def test_zero_recruitment_has_source_output_and_live_right_derivative(self):
        weights = torch.tensor([[1.0, 0.0, 0.7], [0.0, 1.0, 0.2]])
        base = torch.tensor([[0.2, 0.4]])
        extra = torch.zeros(1, 1, requires_grad=True)
        source = physical(base, weights[:, :2], torch.ones(2), False)
        out = physical(torch.cat([base, extra], dim=1), weights, torch.ones(3), True)
        torch.testing.assert_close(source, out, atol=0, rtol=0)
        out.sum().backward()
        self.assertAlmostEqual(float(extra.grad), 0.9, places=5)
