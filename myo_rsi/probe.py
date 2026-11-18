"""Run paired, seeded environment traces on the native CPU and CUDA backends.

Each trace uses a fixed open-loop control pattern. It is a compatibility
measurement, never a training trajectory or a PPO performance estimate.
"""

import argparse
import csv
import hashlib
import json
import platform
import subprocess
from importlib import metadata
from pathlib import Path

from .protocol import PROBE_SEED, TASKS, UPSTREAM


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def versions():
    packages = {}
    for name in (
        "myosuite",
        "myo-sim",
        "gymnasium",
        "numpy",
        "mujoco",
        "mjlab",
        "mujoco-warp",
        "warp-lang",
        "rsl-rl-lib",
        "torch",
    ):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return packages


def host():
    try:
        gpu = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            encoding="utf8",
            timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        gpu = None
    return {
        "python": platform.python_version(),
        "platform": platform.system(),
        "architecture": platform.machine(),
        "gpu": gpu,
        "packages": versions(),
        "myosuite_release_reference": UPSTREAM,
    }


def describe_model(model):
    import mujoco
    import numpy as np

    names = [model.actuator(i).name.split("/")[-1] for i in range(model.nu)]
    if len(names) != 39 or len(set(names)) != 39 or model.na != 39:
        raise ValueError(
            f"Expected 39 distinct hand muscle channels, got {model.nu}/{model.na}"
        )
    if not np.all(model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE):
        raise ValueError("A hand channel is not muscle driven")
    # Compiled MJB includes the task scene. The second signature omits goal sites
    # and force magnitudes and compares the mechanical hand across task variants.
    buf = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, buf)
    skeleton = {}
    for attr in (
        "jnt_type",
        "jnt_axis",
        "jnt_pos",
        "jnt_range",
        "actuator_trntype",
        "actuator_trnid",
        "actuator_dyntype",
        "actuator_dynprm",
        "actuator_ctrlrange",
        "tendon_adr",
        "tendon_num",
        "wrap_type",
        "wrap_prm",
    ):
        skeleton[attr] = np.asarray(getattr(model, attr)).tolist()
    skeleton["joints"] = [model.joint(i).name.split("/")[-1] for i in range(model.njnt)]
    skeleton["muscles"] = names
    force = model.actuator_gainprm[:, 2].copy()
    auto = force < 0
    force[auto] = model.actuator_gainprm[auto, 3] / model.actuator_acc0[auto]
    return {
        "model_hash": hashlib.sha256(buf.tobytes()).hexdigest(),
        "hand_contract_hash": digest(skeleton),
        "actuator_order": names,
        "actuator_order_hash": digest(names),
        "native_action_dim": model.nu,
        "policy_action_dim": model.nu,
        "nq": model.nq,
        "nv": model.nv,
        "na": model.na,
        "physics_dt": model.opt.timestep,
        "peak_force": force.tolist(),
    }


def array(value):
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value).copy()


def trace(env, backend, seed, steps, dimensions):
    import numpy as np

    from .actions import excitation, native_action

    obs, _ = env.reset(seed=seed)
    gpu = backend == "mjlab"
    states = [array(obs["actor"] if gpu else obs).ravel()]
    metrics = []
    expected_errors = []
    samples = []
    # Actuator-specific patterns detect permutations; both endpoints are sampled.
    sweep = np.linspace(-1, 1, dimensions, dtype=np.float32)
    for i in range(steps):
        action = np.roll(sweep, 7 * i)
        if gpu:
            import torch

            obs, reward, terminated, truncated, _ = env.step(
                torch.as_tensor(action, device=env.device)[None, :]
            )
            solved = dict(env.metrics_manager.get_active_iterable_terms(0))["success"][
                0
            ]
            ctrl = array(env.sim.data.ctrl)[0]
            activation = array(env.sim.data.act)[0]
            obs = obs["actor"]
        else:
            obs, reward, terminated, truncated, info = env.step(action)
            solved = info["solved"]
            ctrl = array(env.unwrapped.data.ctrl)
            activation = array(env.unwrapped.data.act)
        done = bool(terminated) or bool(truncated)
        states.append(array(obs).ravel())
        metrics.append(
            [float(reward), float(solved), float(terminated), float(truncated)]
        )
        # mjlab resets done lanes internally. Their post-step ctrl/act belong to the next episode.
        if not (gpu and done):
            expected = excitation(action)
            native, _ = native_action(ctrl)
            np.testing.assert_allclose(ctrl, expected, rtol=0, atol=2e-6)
            np.testing.assert_allclose(native, action, rtol=0, atol=2e-5)
            if (
                not np.isfinite(activation).all()
                or np.any(activation < -1e-5)
                or np.any(activation > 1.00001)
            ):
                raise ValueError("Illegal activation state")
            expected_errors.append(float(np.max(np.abs(ctrl - expected))))
            samples.append(
                {
                    "step": i + 1,
                    "ctrl_min": float(ctrl.min()),
                    "ctrl_max": float(ctrl.max()),
                    "act_min": float(activation.min()),
                    "act_max": float(activation.max()),
                    "ctrl_act_max_gap": float(np.abs(ctrl - activation).max()),
                }
            )
        if done:
            break
    if not samples or not np.isfinite(states).all() or not np.isfinite(metrics).all():
        raise ValueError("Missing or nonfinite probe data")
    return np.stack(states), np.asarray(metrics), samples, max(expected_errors)


def probe(env_id, backend, seed, steps):
    import numpy as np
    from myosuite import make_env

    gpu = backend == "mjlab"
    if gpu:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                f"torch.cuda.is_available() is false (build CUDA {torch.version.cuda})"
            )
        env = make_env(env_id, backend="mjlab", num_envs=1, device="cuda:0", seed=seed)
    else:
        env = make_env(env_id)
    try:
        base = env if gpu else env.unwrapped
        model = env.sim.mj_model if gpu else base.model
        row = describe_model(model)
        if gpu:
            from myosuite.envs.myo.backends.mjlab.tasks.mdp.action_map import (
                action_actuator_names,
            )

            names = action_actuator_names(env.action_manager.get_term("muscles"), env)
            if names != row["actuator_order"]:
                raise ValueError("GPU action term order differs from model")
            spec = env.metrics_manager.cfg["success"]
            row.update(
                episode_horizon=int(env.max_episode_length),
                control_dt=float(env.step_dt),
                reward_components=list(env.reward_manager.active_terms),
                termination_terms=list(env.termination_manager.active_terms),
                success_key="Episode_Metrics/success",
                success_function=spec.func.__name__,
                goals=list(env.cfg.commands),
                cuda=torch.version.cuda,
            )
        else:
            np.testing.assert_array_equal(env.action_space.low, -np.ones(model.nu))
            np.testing.assert_array_equal(env.action_space.high, np.ones(model.nu))
            row.update(
                episode_horizon=int(env.spec.max_episode_steps),
                control_dt=base.dt,
                reward_components=base.rwd_keys_wt,
                observation_terms=base.obs_keys,
                success_key="info.solved",
            )
            if "Pose" in env_id:
                row.update(
                    success_criterion=f"L2 joint error < {base.pose_thd} rad",
                    goal_ranges=array(base._target_jnt_range).tolist(),
                    termination_terms=["far pose error", "time limit"],
                )
            else:
                row.update(
                    success_criterion=f"L2 stacked tip error < {0.0125 * len(base.tip_sids)} m",
                    goal_ranges={
                        k: array(v).tolist() for k, v in base.target_reach_range.items()
                    },
                    far_distance=base.far_th * len(base.tip_sids),
                    termination_terms=[
                        "far tip error after two control steps",
                        "time limit",
                    ],
                )
        first, values, samples, err = trace(env, backend, seed, steps, model.nu)
        second, repeated, _, _ = trace(env, backend, seed, steps, model.nu)
        row.update(
            obs_dim=first.shape[-1],
            probe_seed=seed,
            steps_per_trace=len(values),
            probe_transitions=2 * len(values),
            probe_samples=samples,
            repeat_obs_max_diff=float(np.max(np.abs(first - second))),
            mapping_max_error=err,
            verification_status="PASS",
        )
        try:
            np.testing.assert_allclose(first[0], second[0], rtol=0, atol=2e-5)
            np.testing.assert_array_equal(values[:, 1:], repeated[:, 1:])
            if not gpu:
                np.testing.assert_allclose(first, second, rtol=1e-5, atol=2e-5)
                np.testing.assert_allclose(values, repeated, rtol=1e-5, atol=2e-5)
            else:
                # One vector mixes radians, velocity*dt, metres and activation.
                # Compare named observation groups in their own units; retain
                # the initial uniform-tolerance diagnostic in the run record.
                angular = np.deg2rad(0.5)
                widths = {
                    "qpos": (model.nq, angular),
                    "qvel": (model.nv, angular),
                    "pose_err": (model.nq, angular),
                    "tip_pos": (15, 0.001),
                    "reach_err": (15, 0.001),
                    "act": (model.na, 2e-5),
                }
                terms = list(env.cfg.observations["actor"].terms)
                offset = 0
                row["repeat_errors_by_term"] = {}
                for name in terms:
                    width, tolerance = widths[name]
                    err = float(
                        np.max(
                            np.abs(
                                first[:, offset : offset + width]
                                - second[:, offset : offset + width]
                            )
                        )
                    )
                    row["repeat_errors_by_term"][name] = {
                        "max_abs": err,
                        "tolerance": tolerance,
                    }
                    offset += width
                if offset != first.shape[1]:
                    raise ValueError("Unrecognized observation layout")
                for name, check in row["repeat_errors_by_term"].items():
                    if check["max_abs"] > check["tolerance"]:
                        raise ValueError(
                            f"{name} drift exceeds its physical tolerance: {check}"
                        )
                # Independent, complete episodes exercise the native terminal
                # decision beyond the brief action/kinematics probe.
                row["endpoint_repeats"] = []
                for heldout in (709, 719):
                    one, m1, _, _ = trace(
                        env, backend, heldout, row["episode_horizon"], model.nu
                    )
                    two, m2, _, _ = trace(
                        env, backend, heldout, row["episode_horizon"], model.nu
                    )
                    row["probe_transitions"] += len(m1) + len(m2)
                    equal = np.array_equal(m1[-1, 1:], m2[-1, 1:]) and len(m1) == len(
                        m2
                    )
                    row["endpoint_repeats"].append(
                        dict(
                            seed=heldout,
                            steps=[len(m1), len(m2)],
                            terminal_outcomes=[
                                m1[-1, 1:].tolist(),
                                m2[-1, 1:].tolist(),
                            ],
                            returns=[float(m1[:, 0].sum()), float(m2[:, 0].sum())],
                            matched=bool(equal),
                        )
                    )
                    np.testing.assert_allclose(one[0], two[0], rtol=0, atol=2e-5)
                    if not equal or not any(m1[-1, 2:]):
                        raise ValueError(
                            f"Native terminal outcome or horizon differs for seed {heldout}"
                        )
        except (ValueError, AssertionError) as exc:
            row.update(verification_status="BLOCKED", error=str(exc))
        return row
    finally:
        env.close()


def check_shared_hand(rows):
    import numpy as np

    for backend in ("cpu", "mjlab"):
        by_task = {r["task_id"]: r for r in rows if r["backend"] == backend}
        source = by_task.get("T1", {})
        if source.get("verification_status") != "PASS":
            continue
        for task in ("T0", "T2", "T3"):
            row = by_task.get(task, {})
            if row.get("verification_status") != "PASS":
                continue
            try:
                if source["hand_contract_hash"] != row["hand_contract_hash"]:
                    raise ValueError("Mechanical hand/action signature differs from T1")
                scale = 0.5 if task == "T3" else 1
                np.testing.assert_allclose(
                    row["peak_force"],
                    np.array(source["peak_force"]) * scale,
                    rtol=1e-4,
                    atol=1e-6,
                )
                row["verified_force_ratio_to_T1"] = scale
            except (ValueError, AssertionError) as exc:
                row.update(verification_status="BLOCKED", error=str(exc))


MANIFEST = (
    "task_id env_id backend package_versions model_hash actuator_order_hash obs_dim "
    "policy_action_dim native_action_dim success_key episode_horizon verification_status error"
).split()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["cpu", "mjlab", "both"], default="both")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--steps", type=int, default=6)
    args = parser.parse_args()
    if args.steps < 2:
        parser.error("At least two steps are required")
    system = host()
    rows = []
    backends = ("cpu", "mjlab") if args.backend == "both" else (args.backend,)
    for backend in backends:
        for task, env_id in TASKS.items():
            row = dict(task_id=task, env_id=env_id, backend=backend)
            try:
                row.update(probe(env_id, backend, PROBE_SEED, args.steps))
            except Exception as exc:
                row.update(
                    verification_status="BLOCKED", error=f"{type(exc).__name__}: {exc}"
                )
            rows.append(row)
            print(
                task,
                backend,
                row["verification_status"],
                row.get("error", ""),
                flush=True,
            )
    if any(r["verification_status"] == "PASS" for r in rows):
        check_shared_hand(rows)
        for tid in TASKS:
            pair = [
                r
                for r in rows
                if r["task_id"] == tid and r["verification_status"] == "PASS"
            ]
            if len(pair) == 2:
                for key in (
                    "actuator_order_hash",
                    "native_action_dim",
                    "obs_dim",
                    "episode_horizon",
                    "control_dt",
                ):
                    if pair[0][key] != pair[1][key]:
                        pair[1].update(
                            verification_status="BLOCKED",
                            error=f"CPU/GPU differ in {key}",
                        )
    normal = [r for r in rows if r["task_id"] != "T3"]

    def gate(group):
        return set(backends) == {"cpu", "mjlab"} and all(
            r["verification_status"] == "PASS" for r in group
        )

    normal_ok = gate(normal)
    full_ok = normal_ok and gate(rows)
    out = args.results_dir
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "preflight.json").write_text(
        json.dumps(
            dict(
                host=system, normal_gate=normal_ok, adaptation_gate=full_ok, rows=rows
            ),
            indent=2,
        ),
        encoding="utf8",
    )
    with (out / "tables/env_manifest.csv").open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, MANIFEST, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                dict(
                    row, package_versions=json.dumps(system["packages"], sort_keys=True)
                )
            )
    return 0 if full_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
