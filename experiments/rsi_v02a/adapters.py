import torch
from torch import nn


class ResidualMean(nn.Module):
    """Residual in source Gaussian coordinates; source normalizer remains upstream."""
    def __init__(self, original, inputs, actions):
        super().__init__()
        self.base = original
        self.delta = nn.Sequential(nn.Linear(inputs,32),nn.Tanh(),nn.Linear(32,actions))
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(self,x):
        return self.base(x) + self.delta(x)


def install(runner):
    actor=runner.alg.get_policy()
    actor.obs_normalization=False
    actor.requires_grad_(False)
    obs=runner.env.get_observations()
    with torch.no_grad():
        old=actor(obs).clone()
    actor.mlp=ResidualMean(actor.mlp,actor.obs_dim,old.shape[-1]).to(old.device)
    with torch.no_grad():
        torch.testing.assert_close(actor(obs),old,rtol=0,atol=0)
    pars=[p for p in list(actor.parameters())+list(runner.alg.critic.parameters()) if p.requires_grad]
    runner.alg.optimizer=type(runner.alg.optimizer)(pars,**runner.alg.optimizer.defaults)
    return actor
