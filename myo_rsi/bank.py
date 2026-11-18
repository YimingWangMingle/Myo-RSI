"""Fit one K=8 bank to physical T1 training controls; split by rollout episode."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import NMF

from .actions import native_action
from .protocol import K, TASKS, TRAIN_SEEDS


def checksum(weights, names):
    data = np.ascontiguousarray(weights, dtype="<f8").tobytes()
    return hashlib.sha256(data + json.dumps(list(names)).encode()).hexdigest()


def load_bank(path, actuator_names):
    with np.load(path, allow_pickle=False) as src:
        weights = src["W"]
        meta = json.loads(src["metadata"].item())
    if meta["source_env"] != TASKS["T1"] or list(actuator_names) != meta["actuator_names"]:
        raise ValueError("Incompatible bank source or actuator order")
    if (weights.shape != (len(actuator_names), K) or not np.isfinite(weights).all()
            or np.any(weights < 0) or not np.allclose(weights.max(axis=0), 1.0)):
        raise ValueError("Invalid normalized synergy bank")
    if checksum(weights, actuator_names) != meta["bank_hash"]:
        raise ValueError("Bank hash mismatch")
    weights.flags.writeable = False
    return weights, meta


def fit(paths, out, seed=31415):
    if out.exists():
        raise FileExistsError(f"Refusing to replace the frozen bank at {out}")
    source = []
    episodes = []
    uuids = set()
    contracts = []
    provenances = []
    seeds = set()
    rejected = 0
    for path in sorted(paths):
        with np.load(path, allow_pickle=False) as f:
            spec = json.loads(f["metadata"].item())
            if (spec["task_id"], spec["env_id"], spec["method"], spec["split"],
                spec["signal"]) != ("T1", TASKS["T1"], "Direct PPO", "train", "data.ctrl"):
                raise ValueError(f"Invalid NMF data provenance in {path.name}")
            if spec["seed"] not in TRAIN_SEEDS or spec["run_id"] in uuids:
                raise ValueError("Unregistered seed or duplicate training run")
            uuids.add(spec["run_id"])
            seeds.add(spec["seed"])
            contracts.append((spec["model_hash"], spec["actuator_names"]))
            x = f["physical_excitation"]
            ep, step = f["episode_id"], f["timestep"]
            if x.ndim != 2 or x.shape != (len(ep),len(spec["actuator_names"])) or step.shape != ep.shape:
                raise ValueError("Inconsistent trajectory fields")
            keep = np.isfinite(x).all(axis=1) & (x >= 0).all(axis=1) & (x <= 1).all(axis=1)
            rejected += int((~keep).sum())
            source.append(x[keep])
            episodes.extend(f'{spec["run_id"]}/{i}' for i in ep[keep])
            with path.open("rb") as stream:
                provenances.append({"run_id": spec["run_id"], "file":path.name,
                                    "sha256":hashlib.file_digest(stream,"sha256").hexdigest()})
    if seeds != set(TRAIN_SEEDS) or not contracts or any(c != contracts[0] for c in contracts):
        raise ValueError("The prespecified three Reach runs and a common body are required")
    samples = np.concatenate(source)
    groups = np.unique(episodes)
    if len(groups) < 5:
        raise ValueError("Need at least five distinct training episodes")
    rng = np.random.default_rng(seed)
    valid = np.isin(episodes, rng.permutation(groups)[:max(1,len(groups)//5)])
    training = samples[~valid]
    nmf = NMF(n_components=K, init="nndsvda", random_state=seed, max_iter=1000)
    c_train = nmf.fit_transform(training)
    c_val = nmf.transform(samples[valid])
    scales = nmf.components_.max(axis=1)
    if np.any(scales <= 0):
        raise ValueError("NMF produced an empty component")
    weights = np.ascontiguousarray((nmf.components_ / scales[:,None]).T, dtype=float)
    def diagnostics(c, x):
        pred = (c * scales) @ weights.T
        _, distortion = native_action(pred)
        return {"relative_frobenius":float(np.linalg.norm(pred-x) / max(np.linalg.norm(x),1e-12)),
                "clip_fraction":float(np.mean(np.abs(distortion)>1e-7)),
                "projection_mae":float(np.abs(distortion).mean()), "samples":len(x)}
    meta = dict(source_env=TASKS["T1"], source_split="train", source_runs=provenances,
                training_seeds=sorted(seeds), model_hash=contracts[0][0],
                actuator_names=contracts[0][1], K=K, seed=seed, rejected_samples=rejected,
                nmf_iterations=int(nmf.n_iter_), nmf_max_iter=1000, nmf_init="nndsvda",
                normalization="unit maximum per column; multiply coefficients by column scale",
                scales=scales.tolist(), bank_hash=checksum(weights,contracts[0][1]),
                train=diagnostics(c_train,training), validation=diagnostics(c_val,samples[valid]),
                validation_episodes=sorted(np.unique(np.asarray(episodes)[valid]).tolist()))
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("xb") as stream:
        np.savez_compressed(stream, W=weights, metadata=json.dumps(meta,sort_keys=True))
    load_bank(out, contracts[0][1])
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("rollouts", type=Path, nargs="+")
    parser.add_argument("--output",type=Path, default=Path("artifacts/synergy_bank_k8.npz"))
    args = parser.parse_args()
    print(json.dumps(fit(args.rollouts,args.output),indent=2))
