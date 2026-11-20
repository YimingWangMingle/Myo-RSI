"""Restore MuJoCo integration state and task state in the original world lanes."""

import hashlib
import pickle
import random

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from ..measurement import reset, tips


def rng_state():
    return (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        torch.cuda.get_rng_state(),
    )


def rng_hash(state):
    py, np_state, cpu, gpu = state
    return hashlib.sha256(
        pickle.dumps((py, np_state), protocol=4)
        + cpu.numpy().tobytes()
        + gpu.numpy().tobytes()
    ).hexdigest()


def set_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    torch.cuda.set_rng_state(state[3])


def task_buffers(raw):
    cmd = raw.command_manager.get_term("reach")
    buffers = {
        "episode": raw.episode_length_buf,
        "pending": raw._manual_reset_pending,
        "goal": cmd.command,
        "command_time": cmd.time_left,
        "command_counter": cmd.command_counter,
    }
    objects = [raw.action_manager, *raw.action_manager._terms.values()]
    for index, obj in enumerate(objects):
        for name, tensor in vars(obj).items():
            if (
                isinstance(tensor, torch.Tensor)
                and tensor.ndim
                and tensor.shape[0] == raw.num_envs
            ):
                buffers[f"action_{index}_{name}"] = tensor
    return buffers


class Snapshot:
    def __init__(self, raw):
        model = raw.sim.mj_model
        if model.npluginstate:
            raise RuntimeError("Unimplemented Warp plugin state")
        self.sig = int(mjw.State.INTEGRATION)
        self.state = torch.empty(
            (raw.num_envs, mujoco.mj_stateSize(model, self.sig)), device=raw.device
        )
        mjw.get_state(
            raw.sim.wp_model, raw.sim.wp_data, wp.from_torch(self.state), self.sig
        )
        self.buffers = {k: v.clone() for k, v in task_buffers(raw).items()}
        self.counters = raw.common_step_counter, raw._sim_step_counter
        self.rng = rng_state()

    def restore(self, raw):
        mjw.set_state(
            raw.sim.wp_model, raw.sim.wp_data, wp.from_torch(self.state), self.sig
        )
        for k, v in task_buffers(raw).items():
            v.copy_(self.buffers[k])
        raw.common_step_counter, raw._sim_step_counter = self.counters
        raw.sim.forward()
        raw.sim.sense()
        # Forward computes derived quantities, but may update integrator warm-start.
        mjw.set_state(
            raw.sim.wp_model, raw.sim.wp_data, wp.from_torch(self.state), self.sig
        )
        set_rng(self.rng)
        raw.observation_manager._obs_buffer = None
        raw.obs_buf = raw.observation_manager.compute(update_history=False)
        return raw.obs_buf


def step(raw, cmd, tick, counter):
    obs, _, term, trunc, _ = raw.step(cmd)
    counter[0] += raw.num_envs
    if not bool(((term | trunc) == (tick == raw.max_episode_length)).all()):
        raise ValueError(f"Unexpected native termination at tick {tick}")
    return obs


def state(raw, obs, tolerances):
    cmd = raw.command_manager.get_term("reach")
    values = {
        "obs": obs["actor"],
        "goal": cmd.command,
        "tips": tips(raw),
        "episode_length": raw.episode_length_buf,
        "command_counter": cmd.command_counter,
        "command_time_left": cmd.time_left,
    }
    for k in tolerances.keys() - values.keys():
        w = getattr(raw.sim.wp_data, k)
        if w.size == 0:
            continue
        values[k] = wp.to_torch(w)
    return {k: v.detach().clone() for k, v in values.items()}


def errors(ref, curr):
    if ref.keys() != curr.keys():
        raise ValueError("State schema changed during replay")
    return {
        k: (v.to(torch.float64) - curr[k]).abs().reshape(len(v), -1).amax(1)
        for k, v in ref.items()
    }


def native_result(raw):
    m, r = raw.metrics_manager, raw.reward_manager
    success = m._step_values[:, m.active_terms.index("success")].bool().clone()
    err = (
        -r._step_reward[:, r.active_terms.index("reach")] / r.cfg["reach"].weight
    ).clone()
    if not torch.equal(success, err < 0.0625):
        raise ValueError("Native metric disagrees with registered Reach criterion")
    return success, err


def restore_route(raw, route, snap, initial, goals, commands, counter):
    if route == "integration":
        return snap.restore(raw)
    # Restore counters/RNG to original pre-reset conditions. Replay actual native
    # actions from initialization, in the same lane with the original batch width.
    raw.common_step_counter, raw._sim_step_counter = initial[:2]
    set_rng(initial[2])
    obs = reset(raw, goals)
    for i, act in enumerate(commands, 1):
        obs = step(raw, act, i, counter)
    return obs
