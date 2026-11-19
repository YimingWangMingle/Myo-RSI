"""Verified source models and task-native first-episode measurements."""
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from experiments.synergy_budget.ppo import setup
from myo_rsi.probe import describe_model, versions
from myo_rsi.rerun_v01r import cpu, sha as hash_bytes
from .registry import event, read, source, table, write
from .adapters import install


@contextmanager
def controller(task,seed,n,k=16):
    env,runner=setup(task,k,seed,n)
    try:
        parent=source(seed,k)
        info=runner.load(parent["uri"],load_cfg={"actor":True,"critic":True})
        if info != dict(bank_hash=env.bank_hash,task="T1",seed=seed):
            raise ValueError(f"Parent provenance {info}")
        yield env,runner
    finally:
        env.close()


def metadata(env):
    raw=env.unwrapped
    m=describe_model(raw.sim.mj_model)
    obs=env.get_observations()
    info=dict(device=torch.cuda.get_device_name(),versions=versions(),model=m,
              observation={name:list(obs[name].shape[1:]) for name in obs.keys()},
              term_names=list(raw.reward_manager.active_terms),
              reward_weights={k:raw.reward_manager.cfg[k].weight for k in raw.reward_manager.active_terms},
              success="native terminal metric",horizon=raw.max_episode_length,
              termination=list(raw.termination_manager.active_terms))
    if info["observation"]["actor"]!=[115] or info["horizon"]!=100:
        raise ValueError("Reach observation or time limit mismatch")
    return info


def reset(raw,goals):
    for lane,g in enumerate(goals):
        obs,_=raw.reset(seed=int(g),env_ids=torch.tensor([lane],device=raw.device))
    return obs


def tips(raw):
    cmd=raw.command_manager.get_term("reach")
    return cmd._accessor.site_xpos(cmd._tip_ids).reshape(raw.num_envs,5,3)


def td(obs):
    return TensorDict({"actor":obs},batch_size=[len(obs)])


@torch.inference_mode()
def episodes(env,runner,goals,loc,tag,record=False,exemplar=False):
    """Keep all physics ticks, including the final step before any reset."""
    raw=env.unwrapped
    raw.cfg.auto_reset=False
    obs=reset(raw,goals)
    aim=cpu(raw.command_manager.get_command("reach")).reshape(len(goals),5,3)
    init=cpu(obs["actor"])
    actor=runner.alg.get_policy()
    actor.eval()
    rewards=np.zeros(len(goals))
    sat=np.zeros(len(goals))
    energy=np.zeros(len(goals))
    trace={}
    tic=time.perf_counter()
    steps=0
    try:
        for t in range(1,101):
            pre=obs["actor"]
            act=actor(obs)
            native=env.native(act)
            if env.k!=39 and hasattr(actor.mlp,"base"):
                base=actor.mlp.base(actor.get_latent(obs))
                energy+=cpu((act-base).square().mean(1))
            obs,r,term,trunc,_=raw.step(native)
            steps+=len(goals)
            if bool((term|trunc).any()) != (t==100) or (t==100 and not bool((term|trunc).all())):
                raise ValueError("Unexpected episode termination")
            rewards+=cpu(r)
            ctrl=raw.sim.data.ctrl
            expected=torch.sigmoid(5*(native-.5))
            torch.testing.assert_close(ctrl,expected,rtol=0,atol=2e-6)
            sat+=cpu(((native.abs()>=0.99999).float()).mean(1))
            if record or exemplar or t==100:
                tip=cpu(tips(raw))
                finger=np.linalg.norm(tip-aim,axis=2)
                if record or exemplar:
                    take=slice(None) if record else slice(0,1)
                    cmd=raw.command_manager.get_command("reach")
                    vals=dict(obs=pre,raw=act,native=native,ctrl=ctrl,act=raw.sim.data.act,
                              qpos=raw.sim.data.qpos,qvel=raw.sim.data.qvel,
                              reward=raw.reward_manager._step_reward,goal=cmd)
                    if env.k!=39:
                        vals["recruitment"]=act.relu()*env.scale
                    for name in ("actuator_length","actuator_velocity","qacc_warmstart","time"):
                        value=getattr(raw.sim.data,name,None)
                        if torch.is_tensor(value) and value.ndim and len(value)==len(goals):
                            vals[name]=value
                    for name,val in vals.items():
                        trace.setdefault(name,[]).append(cpu(val[take]))
                    trace.setdefault("tips",[]).append(tip[take])
                    trace.setdefault("finger",[]).append(finger[take])
        err=cpu(-raw.reward_manager._step_reward[:,raw.reward_manager.active_terms.index("reach")]/raw.reward_manager.cfg["reach"].weight)
        success=cpu(raw.metrics_manager._step_values[:,raw.metrics_manager.active_terms.index("success")]).astype(int)
        np.testing.assert_array_equal(success,err<0.0625)
        np.testing.assert_allclose(np.linalg.norm(finger,axis=1),err,atol=2e-5)
        rows=[dict(goal=int(g),goal_hash=hash_bytes(aim[i].tobytes()),reset_hash=hash_bytes(init[i].tobytes()),
                   success=int(success[i]),error=float(err[i]),length=t,episode_return=float(rewards[i]),
                   saturation=float(sat[i]/t),residual_mean_square=float(energy[i]/t),
                   **{f"tip_{j+1}_m":float(val) for j,val in enumerate(finger[i])}) for i,g in enumerate(goals)]
        table(loc/(tag+".csv"),rows)
        if trace:
            np.savez_compressed(loc/(tag+".npz"),goals=goals if record else goals[:1],**{k:np.stack(v) for k,v in trace.items()})
        return rows
    finally:
        event(loc,tag,steps,time.perf_counter()-tic,device="cuda:0")


def paired(a,b):
    if [(x["goal"],x["goal_hash"],x["reset_hash"]) for x in a] != [(x["goal"],x["goal_hash"],x["reset_hash"]) for x in b]:
        raise ValueError("Unmatched initial goals or observations")
