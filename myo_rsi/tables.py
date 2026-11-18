"""Normalize run artifacts without treating seeds as independent episodes."""

import csv
import hashlib
import json
import math

SCHEMAS = {
    "checkpoint_manifest": "run_id task_id method training_seed final_checkpoint sha256 source_checkpoint_id bank_hash actor_params trainable_actor_params train_steps projection_fraction projection_mae",
    "train_progress": "task_id method training_seed checkpoint_id environment_steps ppo_iteration train_return eval_success_n eval_episodes elapsed_wall_s gpu_hours",
    "eval_episodes": "task_id method training_seed checkpoint_id eval_seed episode_id body_variant success episode_return episode_length terminal_reason split goal_hash reset_obs_hash",
    "bank_metrics": "bank_id bank_hash source_task source_training_seeds K num_muscles num_actions_fit nmf_train_recon nmf_val_recon excitation_clip_fraction actuator_order_hash",
    "adaptation_progress": "source_checkpoint_id training_seed adapt_method adapt_steps eval_task eval_success_n eval_episodes wall_s gpu_hours split checkpoint_id",
    "resource_cost": "stage task_id method seed environment_steps sim_search_steps eval_steps wall_s gpu_hours source_run_id",
    "exp_ab_comparison": "task_id method seed checkpoint_id bank_hash environment_steps gpu_hours success_n episodes success_rate wilson_low wilson_high",
    "exp_c_adaptation": "source_checkpoint_id task_id seed method checkpoint_id adaptation_steps t3_before t3_after t1_before t1_after time_s gpu_hours status",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf8"))


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf8", newline="") as stream:
        writer = csv.DictWriter(stream, fields.split(), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def wilson(n, count):
    z = 1.959963984540054
    p = n / count
    den = 1 + z * z / count
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / den
    centre = (p + z * z / (2 * count)) / den
    return max(0, centre - half), min(1, centre + half)


def collect(root, results):
    data = {k: [] for k in SCHEMAS}
    cost = data["resource_cost"]
    # Account for compatibility checks outside the randomized PPO comparisons.
    diag = results / "diagnostics"
    for run_id, source in (
        ("cpu_probe", diag / "cpu_only_preflight.json"),
        ("strict_probe", diag / "initial_uniform_tolerance.json"),
        ("physical_probe", results / "preflight.json"),
    ):
        if source.exists():
            for r in read_json(source)["rows"]:
                steps = r.get("probe_transitions")
                if (
                    steps is None
                    and run_id == "strict_probe"
                    and "among violations" in r.get("error", "")
                ):
                    # The strict comparison is reached only after both six-step traces.
                    steps = 12
                cost.append(
                    dict(
                        stage="probe",
                        task_id=r["task_id"],
                        method=r["backend"],
                        seed=701,
                        environment_steps=steps,
                        sim_search_steps=0,
                        eval_steps=0,
                        source_run_id=run_id,
                    )
                )
    if (diag / "ppo_smoke.json").exists():
        smoke = read_json(diag / "ppo_smoke.json")
        cost.append(
            dict(
                stage="ppo_smoke",
                task_id=smoke["task_id"],
                method="Direct PPO",
                seed=smoke["seed"],
                environment_steps=smoke["transitions"],
                eval_steps=0,
                sim_search_steps=0,
                wall_s=smoke["logged_train_wall_s"],
                gpu_hours=smoke["logged_train_wall_s"] / 3600,
                source_run_id="upstream_smoke",
            )
        )
    for path in sorted((root / "datasets/diagnostic_rollouts").glob("*/manifest.json")):
        d = read_json(path)
        cost.append(
            dict(
                stage="diagnostic_rollout",
                task_id=d["task_id"],
                method=d["method"],
                seed=d["training_seed"],
                environment_steps=d["environment_steps"],
                sim_search_steps=0,
                eval_steps=0,
                wall_s=d["wall_s"],
                gpu_hours=d["wall_s"] / 3600,
                source_run_id=d["run_id"],
            )
        )
    goals = {}
    seen_banks = set()
    effective_runs = {}
    for meta_file in sorted((root / "checkpoints").glob("*/*/*/run.json")):
        meta = read_json(meta_file)
        if meta.get("status") != "COMPLETE":
            continue
        run = meta_file.parent
        task = meta["task_id"]
        method = meta["method"]
        seed = meta["seed"]
        tag = meta["run_id"]
        if meta["bank_hash"]:
            seen_banks.add(meta["bank_hash"])
        cfg = read_json(run / "ppo.json")
        cfg.update(seed=seed, logger="tensorboard", save_interval=10**9)
        effective_runs[tag] = dict(
            runner_config=cfg,
            train_iterations=meta["budget_iterations"],
            parallel_envs=meta["num_envs"],
            source_checkpoint_id=meta["source_checkpoint_id"],
            bank_hash=meta["bank_hash"],
            device="cuda:0",
        )
        points = read_json(run / "points.json")
        final = points[-1]
        total = final["environment_steps"]
        checkpoint = run / f"policy_{meta['budget_iterations']}.pt"
        with checkpoint.open("rb") as stream:
            sha = hashlib.file_digest(stream, "sha256").hexdigest()
        data["checkpoint_manifest"].append(
            dict(
                run_id=tag,
                task_id=task,
                method=method,
                training_seed=seed,
                final_checkpoint=checkpoint.relative_to(root).as_posix(),
                sha256=sha,
                source_checkpoint_id=meta["source_checkpoint_id"],
                bank_hash=meta["bank_hash"],
                actor_params=meta["actor_params"],
                trainable_actor_params=meta["trainable_actor_params"],
                train_steps=total,
                projection_fraction=meta["projection_fraction"],
                projection_mae=meta["projection_mae"],
            )
        )
        cost.append(
            dict(
                stage=("frozen_setup" if meta["code"] == "C0" else "adapt_train")
                if task == "T3"
                else "train",
                task_id=task,
                method=method,
                seed=seed,
                environment_steps=total,
                sim_search_steps=0,
                eval_steps=0,
                wall_s=meta["wall_s"],
                gpu_hours=meta["wall_s"] / 3600,
                source_run_id=tag,
            )
        )
        scores = {}
        time_by_it = {p["iteration"]: p for p in points}
        for epfile in sorted(run.glob("eval_T*.json")):
            evaluation = read_json(epfile)
            eval_task = evaluation["points"][0]["eval_task"]
            nstep = 0
            for entry in evaluation["points"]:
                episodes = entry["episodes"]
                split = entry["split"]
                it = entry["iteration"]
                expected = (
                    range(40_000, 40_100) if split == "final" else range(20_000, 20_100)
                )
                if sorted(e["eval_seed"] for e in episodes) != list(expected):
                    raise ValueError(f"Incomplete held-out list at {tag}/{it}")
                n = sum(e["success"] for e in episodes)
                count = len(episodes)
                scores[eval_task, it, split] = (n, count)
                nstep += entry["sim_transitions"]
                for ep in episodes:
                    identity = (eval_task, split, ep["eval_seed"])
                    if goals.setdefault(identity, ep["goal_hash"]) != ep["goal_hash"]:
                        raise ValueError(
                            f"Evaluation goal mismatch at {tag}: {identity}"
                        )
                    data["eval_episodes"].append(
                        dict(
                            ep,
                            task_id=eval_task,
                            method=method,
                            training_seed=seed,
                            checkpoint_id=entry["checkpoint_id"],
                            split=split,
                            body_variant="sarcopenic"
                            if eval_task == "T3"
                            else "normal",
                        )
                    )
                if task == "T3":
                    data["adaptation_progress"].append(
                        dict(
                            source_checkpoint_id=meta["source_checkpoint_id"],
                            training_seed=seed,
                            adapt_method=method,
                            adapt_steps=entry["environment_steps"],
                            eval_task=eval_task,
                            eval_success_n=n,
                            eval_episodes=count,
                            wall_s=time_by_it[it]["elapsed_wall_s"],
                            gpu_hours=time_by_it[it]["gpu_hours"],
                            split=split,
                            checkpoint_id=entry["checkpoint_id"],
                        )
                    )
            cost.append(
                dict(
                    stage="evaluate",
                    task_id=eval_task,
                    method=method,
                    seed=seed,
                    environment_steps=0,
                    sim_search_steps=0,
                    eval_steps=nstep,
                    wall_s=evaluation["stage_wall_s"],
                    gpu_hours=evaluation["stage_wall_s"] / 3600,
                    source_run_id=tag,
                )
            )
        for p in points:
            it = p["iteration"]
            n, count = scores.get((task, it, "validation"), (None, None))
            data["train_progress"].append(
                dict(
                    task_id=task,
                    method=method,
                    training_seed=seed,
                    checkpoint_id=f"{tag}/policy_{it}.pt",
                    environment_steps=p["environment_steps"],
                    ppo_iteration=it,
                    train_return=p["train_return"],
                    eval_success_n=n,
                    eval_episodes=count,
                    elapsed_wall_s=p["elapsed_wall_s"],
                    gpu_hours=p["gpu_hours"],
                )
            )
        last = meta["budget_iterations"]
        if task in ("T1", "T2") and (task, last, "final") in scores:
            n, count = scores[task, last, "final"]
            lo, hi = wilson(n, count)
            data["exp_ab_comparison"].append(
                dict(
                    task_id=task,
                    method=method,
                    seed=seed,
                    checkpoint_id=f"{tag}/policy_{last}.pt",
                    bank_hash=meta["bank_hash"],
                    environment_steps=total,
                    gpu_hours=meta["wall_s"] / 3600,
                    success_n=n,
                    episodes=count,
                    success_rate=n / count,
                    wilson_low=lo,
                    wilson_high=hi,
                )
            )
        if task == "T3" and all(
            (t, it, "final") in scores for t in ("T1", "T3") for it in (0, last)
        ):

            def rate(t, it):
                n, count = scores[t, it, "final"]
                return n / count

            data["exp_c_adaptation"].append(
                dict(
                    source_checkpoint_id=meta["source_checkpoint_id"],
                    task_id=task,
                    seed=seed,
                    method=method,
                    checkpoint_id=f"{tag}/policy_{last}.pt",
                    adaptation_steps=total,
                    t3_before=rate("T3", 0),
                    t3_after=rate("T3", last),
                    t1_before=rate("T1", 0),
                    t1_after=rate("T1", last),
                    time_s=0.0 if meta["code"] == "C0" else meta["wall_s"],
                    gpu_hours=0.0 if meta["code"] == "C0" else meta["wall_s"] / 3600,
                    status="RUN",
                )
            )
    bank_file = root / "artifacts/nmf.json"
    if bank_file.exists():
        payload = read_json(bank_file)
        bank = payload["bank"]
        manifest = read_json(results / "preflight.json")
        ref = next(
            r
            for r in manifest["rows"]
            if r["task_id"] == "T1" and r["backend"] == "mjlab"
        )
        if seen_banks - {bank["bank_hash"]}:
            raise ValueError("More than one bank used in v0.1")
        data["bank_metrics"].append(
            dict(
                bank_id="T1-K8",
                bank_hash=bank["bank_hash"],
                source_task="T1",
                source_training_seeds=json.dumps(bank["training_seeds"]),
                K=bank["K"],
                num_muscles=len(bank["actuator_names"]),
                num_actions_fit=bank["train"]["samples"],
                nmf_train_recon=bank["train"]["relative_frobenius"],
                nmf_val_recon=bank["validation"]["relative_frobenius"],
                excitation_clip_fraction=bank["validation"]["clip_fraction"],
                actuator_order_hash=ref["actuator_order_hash"],
            )
        )
        cost.append(
            dict(
                stage="nmf_cpu",
                task_id="T1",
                method="NMF",
                environment_steps=0,
                eval_steps=0,
                sim_search_steps=0,
                wall_s=payload["wall_s"],
                gpu_hours=0,
                source_run_id=",".join(r["run_id"] for r in bank["source_runs"]),
            )
        )
    (results / "experiment_config.json").write_text(
        json.dumps(
            dict(
                runs=effective_runs,
                evaluation_mode="deterministic",
                validation_seeds=list(range(20_000, 20_100)),
                final_seeds=list(range(40_000, 40_100)),
            ),
            indent=2,
        )
        + "\n",
        encoding="utf8",
    )
    for name, schema in SCHEMAS.items():
        path = (
            results / (name + ".csv")
            if name.startswith("exp_")
            else results / "tables" / (name + ".csv")
        )
        write_csv(path, schema, data[name])
    for name, method in (
        ("exp_a_direct_ppo", "Direct PPO"),
        ("exp_b_synergy_ppo", "Synergy PPO"),
    ):
        write_csv(
            results / (name + ".csv"),
            SCHEMAS["exp_ab_comparison"],
            [row for row in data["exp_ab_comparison"] if row["method"] == method],
        )
    return data
