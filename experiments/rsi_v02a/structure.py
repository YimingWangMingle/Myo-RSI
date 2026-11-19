"""B4/B5 functional tests with saved bank versions and paired old-body rollouts."""

from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from .measurement import controller, episodes, metadata, paired, td
from .registry import PROTOCOL, attempt, completed, read, write, sha
from myo_rsi.bank_study import LOW, HIGH


def new_bank(w, operation, directions):
    base = w.clone()
    if operation == "reshape":
        base[::2, 0] *= 0.98
        base[:, 0] /= base[:, 0].max()
        return base, "bounded alternating-weight edit to column 1"
    candidates = [
        (torch.as_tensor(v, device=w.device).relu(), "verified full-action correction")
        for v in directions
    ]
    # An engineering fixture tests the interface when no witness supplies an admissible column.
    eye = torch.eye(w.shape[0], device=w.device)
    for v in eye:
        candidates.append((v, "unit-vector interface fixture"))
    for v, label in candidates:
        if float(v.max()) <= 1e-7:
            continue
        v = v / v.max()
        if torch.nn.functional.cosine_similarity(w, v[:, None], dim=0).max() < 0.97:
            return torch.cat([base, v[:, None]], dim=1), label
    raise ValueError("No admissible nonduplicate column")


class RecruitmentHead(nn.Module):
    def __init__(self, actor, expanding):
        super().__init__()
        self.actor = actor
        self.actor.requires_grad_(False)
        self.expanding = expanding
        self.head = nn.Linear(actor.obs_dim, 1 if expanding else 16)
        nn.init.zeros_(self.head.bias)
        nn.init.zeros_(self.head.weight)

    @property
    def mlp(self):
        return self.head

    def forward(self, obs):
        z = self.actor(obs)
        delta = self.head(self.actor.get_latent(obs))
        return torch.cat([z, delta], dim=1) if self.expanding else z + delta


def physical(z, w, q, expand):
    # clamp_min has derivative 1 at exactly zero, unlike ReLU. Check that
    # this derivative reaches a physical muscle channel in the actual decoder.
    c = (
        torch.cat([z[:, :-1].relu(), z[:, -1:].clamp_min(0)], dim=1)
        if expand
        else z.relu()
    ) * q
    return (c @ w.T).clamp(LOW, HIGH)


def smoke(seed):
    with attempt("structure", seed) as (loc, m):
        diag = None
        try:
            diag = completed("diagnose", seed)
        except FileNotFoundError:
            pass
        directions = []
        if diag:
            with np.load(diag / "corrections.npz") as z:
                directions = z["full_directions"].copy()
        with controller("T1", seed, 20) as (env, runner):
            m["environment"] = metadata(env)
            base = runner.alg.get_policy()
            base.eval()
            source_w = env.W.clone()
            q = env.scale.clone()
            baseline = episodes(
                env, runner, list(range(84000, 84020)), loc, "normal_source"
            )
            episodes(
                env,
                runner,
                list(range(83500, 83520)),
                loc,
                "calibration",
                exemplar=True,
            )
            with np.load(loc / "calibration.npz") as z:
                x = td(torch.tensor(z["obs"][:, 0], device=env.device))
            with torch.no_grad():
                old = physical(base(x), source_w, q, False).detach()
            details = []
            for op in ("reshape", "expand"):
                expand = op == "expand"
                w, provenance = new_bank(source_w, op, directions)
                scale = torch.cat([q, q.median()[None]]) if expand else q.clone()
                assert (w >= 0).all() and torch.allclose(
                    w.max(0).values, torch.ones(w.shape[1], device=w.device)
                )
                head = RecruitmentHead(base, expand).to(env.device)
                opt = torch.optim.Adam(head.head.parameters(), lr=0.0003)
                before = physical(head(x), w, scale, expand)
                initial = float((before - old).abs().max().detach())
                if expand and initial > 2e-6:
                    raise ValueError("Expansion changed source action")
                target = (old + 0.01 * w[:, -1]).clamp(LOW, HIGH) if expand else old
                trace = []
                for it in range(60):
                    out = physical(head(x), w, scale, expand)
                    loss = (out - target).square().mean()
                    opt.zero_grad()
                    loss.backward()
                    grad = float(head.head.weight.grad.norm())
                    if it == 0:
                        first_grad = grad
                    opt.step()
                    trace.append(float(loss.detach()))
                if first_grad <= 0 or min(trace[1:]) >= trace[0]:
                    raise ValueError("New head has no measured useful gradient")
                archive = loc / (op + ".pt")
                torch.save(
                    dict(
                        W=w.cpu(),
                        q95=scale.cpu(),
                        actuator_order=m["environment"]["model"]["actuator_order"],
                        parent=m["source_checkpoint"],
                        head=head.head.state_dict(),
                        optimizer=opt.state_dict(),
                        operation=op,
                    ),
                    archive,
                )
                saved = torch.load(archive, map_location=env.device, weights_only=True)
                reloaded = RecruitmentHead(base, expand).to(env.device)
                reloaded.head.load_state_dict(saved["head"])
                with torch.no_grad():
                    torch.testing.assert_close(reloaded(x), head(x), atol=0, rtol=0)
                if (
                    saved["actuator_order"]
                    != m["environment"]["model"]["actuator_order"]
                ):
                    raise ValueError("Order changed")
                env.W, env.scale = saved["W"], saved["q95"]
                # Forward values of clamp_min(0) and relu are identical for deployment.
                policy = SimpleNamespace(
                    alg=SimpleNamespace(get_policy=lambda: reloaded)
                )
                measured = episodes(
                    env, policy, list(range(84000, 84020)), loc, op + "_normal"
                )
                paired(baseline, measured)
                details.append(
                    dict(
                        operation=op,
                        k=w.shape[1],
                        candidate_source=provenance,
                        initial_excitation_delta_max=initial,
                        parent_column_linf=float((w[:, :16] - source_w).abs().max()),
                        first_grad=first_grad,
                        first_loss=trace[0],
                        last_loss=trace[-1],
                        parent_success=sum(r["success"] for r in baseline),
                        smoke_success=sum(r["success"] for r in measured),
                        n=len(measured),
                        archive_sha256=sha(archive),
                    )
                )
                env.W, env.scale = source_w, q
            write(loc / "smoke.json", details)


if __name__ == "__main__":
    for s in read(PROTOCOL)["source_seeds"]:
        try:
            completed("structure", s)
        except FileNotFoundError:
            smoke(s)
