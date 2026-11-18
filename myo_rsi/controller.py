"""RSL-RL samples raw Gaussian actions. Decoding happens only at the environment boundary."""

import torch
from mjlab.rl import RslRlVecEnvWrapper
from torch import nn

from .actions import excitation
from .protocol import K


class HandVecEnv(RslRlVecEnvWrapper):
    def __init__(self, env, weights=None, collect=False, collect_from=0):
        super().__init__(env, clip_actions=None)
        self.weights = None
        if weights is not None:
            self.weights = torch.tensor(
                weights.copy(), dtype=torch.float32, device=self.device
            )
            self.num_actions = K
        self.low, self.high = excitation([-1, 1]).tolist()
        self.calls = 0
        self.collect = collect
        self.collect_from = collect_from
        self.records = []
        self.episodes = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.tick = torch.zeros_like(self.episodes)
        self.saturation = torch.zeros(3, device=self.device, dtype=torch.float64)

    def native(self, action, track=False):
        if self.weights is None:
            return action.clamp(-1, 1)
        # W has unit column maxima. A zero latent recruits about 0.10 per component.
        coeff = 0.15 * torch.nn.functional.softplus(action)
        desired = coeff @ self.weights.T
        u = desired.clamp(self.low, self.high)
        if track:
            self.saturation[0] += (desired != u).sum()
            self.saturation[1] += (desired - u).abs().sum()
            self.saturation[2] += u.numel()
        return (0.5 + torch.logit(u) / 5.0).clamp(-1, 1)

    def step(self, actions):
        native = self.native(actions, track=True)
        # Lock data collection to 8 evenly spaced lanes and every 8th step of
        # the last quarter of learning. A terminal autoreset has no terminal ctrl.
        sample = (
            self.collect and self.calls >= self.collect_from and self.calls % 8 == 0
        )
        if sample:
            env = self.unwrapped
            lanes = torch.arange(
                0, self.num_envs, max(1, self.num_envs // 8), device=self.device
            )[:8]
            goal = env.command_manager.get_command("reach")[lanes].clone()
        obs, reward, done, extras = super().step(native)
        if sample:
            valid = ~done[lanes].bool()
            lanes = lanes[valid]

            def cpu(tensor):
                return tensor.detach().cpu().numpy().copy()

            self.records.append(
                dict(
                    episode_id=cpu(self.episodes[lanes] * self.num_envs + lanes),
                    timestep=cpu(self.tick[lanes]),
                    goal=cpu(goal[valid]),
                    raw_action=cpu(actions[lanes]),
                    native_action=cpu(native[lanes]),
                    physical_excitation=cpu(env.sim.data.ctrl[lanes]),
                    activation=cpu(env.sim.data.act[lanes]),
                    qpos=cpu(env.sim.data.qpos[lanes]),
                    reward=cpu(reward[lanes]),
                    observation=cpu(obs["actor"][lanes]),
                )
            )
        self.calls += 1
        if self.collect:
            self.tick += 1
            self.episodes += done.long()
            self.tick.masked_fill_(done.bool(), 0)
        return obs, reward, done, extras


class LocalMean(nn.Module):
    def __init__(self, original, dim):
        super().__init__()
        self.base = original
        self.delta = nn.Sequential(nn.Linear(dim, 16), nn.Tanh(), nn.Linear(16, K))
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(self, inputs):
        return self.base(inputs) + self.delta(inputs)


def install_local(runner):
    actor = runner.alg.get_policy()
    actor.requires_grad_(False)
    # The frozen normalizer stays in forward(); this flag disables statistic updates.
    actor.obs_normalization = False
    obs = runner.env.get_observations()
    with torch.no_grad():
        before = actor(obs).clone()
    actor.mlp = LocalMean(actor.mlp, actor.obs_dim).to(runner.env.device)
    with torch.no_grad():
        torch.testing.assert_close(before, actor(obs), rtol=0, atol=1e-7)
    runner.alg.optimizer.add_param_group({"params": actor.mlp.delta.parameters()})
