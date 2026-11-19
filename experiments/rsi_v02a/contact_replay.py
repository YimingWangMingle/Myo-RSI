"""Independent contact and repeatability measurements on the selected cases.

Diagnostic-only: these new samples cannot contribute demonstrations to B3.
"""

import csv
import numpy as np
import torch
import warp as wp

from .measurement import controller, cpu
from .registry import PROTOCOL, attempt, completed, event, read, table, write
from .weakness import active_gate


@torch.inference_mode()
def contact_audit(seed):
    parent = completed("diagnose", seed)
    with (parent / "failure_manifest.csv").open() as f:
        cases = [r for r in csv.DictReader(f) if r["selected"] == "True"]
    with np.load(parent / "failure_traces.npz") as f:
        data = {k: f[k] for k in f.files}
    with attempt("contact-replay", seed, {"diagnosis": parent.name}) as (loc, m):
        with controller(active_gate()[0], seed, 100) as (env, runner):
            from .measurement import metadata, reset
            import time

            m["environment"] = metadata(env)
            raw = env.unwrapped
            raw.cfg.auto_reset = False
            # Preserve the complete original batch and lane assignment on replay.
            goals = list(map(int, data["goals"]))
            obs = reset(raw, goals)
            fields = raw.sim.wp_data
            counter = wp.to_torch(fields.nacon)
            ids = wp.to_torch(fields.contact.worldid)
            traces = []
            tic = time.perf_counter()
            steps = 0
            try:
                for t in range(100):
                    native = torch.as_tensor(data["native"][t], device=env.device)
                    obs, *_ = raw.step(native)
                    steps += len(goals)
                    n = int(counter[0])
                    world = cpu(ids[:n])
                    contacts = np.bincount(world, minlength=100)
                    for case in cases:
                        goal = int(case["goal"])
                        i = goals.index(goal)
                        qpos = float(
                            np.max(
                                np.abs(cpu(raw.sim.data.qpos[i]) - data["qpos"][t, i])
                            )
                        )
                        qvel = float(
                            np.max(
                                np.abs(cpu(raw.sim.data.qvel[i]) - data["qvel"][t, i])
                            )
                        )
                        traces.append(
                            dict(
                                goal=goal,
                                step=t + 1,
                                physical_contacts=int(contacts[i]),
                                qpos_linf=qpos,
                                qvel_linf=qvel,
                                active_control_linf=float(
                                    np.max(
                                        np.abs(
                                            cpu(raw.sim.data.ctrl[i])
                                            - data["ctrl"][t, i]
                                        )
                                    )
                                ),
                                activation_linf=float(
                                    np.max(
                                        np.abs(
                                            cpu(raw.sim.data.act[i]) - data["act"][t, i]
                                        )
                                    )
                                ),
                            )
                        )
            finally:
                event(
                    loc, "independent_contact_replay", steps, time.perf_counter() - tic
                )
            table(loc / "contact_replay.csv", traces)
            write(
                loc / "summary.json",
                dict(
                    physical_steps=steps,
                    selected_cases=len(cases),
                    contact_frames=sum(r["physical_contacts"] > 0 for r in traces),
                    max_contacts=max(r["physical_contacts"] for r in traces),
                    used_for_search_or_bc=False,
                ),
            )


if __name__ == "__main__":
    for seed in read(PROTOCOL)["source_seeds"]:
        try:
            completed("contact-replay", seed)
        except FileNotFoundError:
            contact_audit(seed)
