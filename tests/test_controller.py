import unittest
from types import SimpleNamespace

import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from myo_rsi.actions import excitation
from myo_rsi.controller import HandVecEnv, install_local


class ControllerTests(unittest.TestCase):
    def test_decoder_extremes_and_frozen_weights(self):
        weights = torch.rand(39, 8)
        before = weights.clone()
        low, high = excitation([-1, 1])
        state = SimpleNamespace(
            weights=weights, low=low, high=high, saturation=torch.zeros(3)
        )
        raw = torch.tensor([[-1000.0] * 8, [1000.0] * 8])
        native = HandVecEnv.native(state, raw, track=True)
        self.assertEqual(native.shape, (2, 39))
        self.assertTrue(torch.isfinite(native).all())
        self.assertTrue((native.abs() <= 1).all())
        self.assertEqual(int(state.saturation[0]), 78)
        torch.testing.assert_close(before, state.weights, atol=0, rtol=0)

    def test_zero_residual_and_base_freeze(self):
        obs = TensorDict({"actor": torch.randn(4, 5)}, batch_size=[4])
        dist = {"class_name": "rsl_rl.modules:GaussianDistribution", "init_std": 0.5}
        actor = MLPModel(obs, {"actor": ["actor"]}, "actor", 8, (8,), "elu", True, dist)
        optimizer = torch.optim.Adam(actor.parameters())
        runner = SimpleNamespace(
            alg=SimpleNamespace(get_policy=lambda: actor, optimizer=optimizer),
            env=SimpleNamespace(get_observations=lambda: obs, device="cpu"),
        )
        with torch.no_grad():
            before = actor(obs).clone()
        install_local(runner)
        base = [p.detach().clone() for p in actor.mlp.base.parameters()]
        torch.testing.assert_close(actor(obs), before)
        raw = actor(obs, stochastic_output=True)
        nll = -actor.get_output_log_prob(raw.detach()).mean()
        optimizer.zero_grad()
        nll.backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(nll))
        self.assertFalse(torch.equal(actor(obs), before))
        for new, old in zip(actor.mlp.base.parameters(), base, strict=True):
            torch.testing.assert_close(new, old, atol=0, rtol=0)
