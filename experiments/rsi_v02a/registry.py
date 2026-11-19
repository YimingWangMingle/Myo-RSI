"""Immutable attempts, atomic metadata and append-only event accounting."""
import csv
import hashlib
import json
import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

ROOT = Path("results/rsi_v02a")
PROTOCOL = Path("experiments/rsi_v02a/protocol.json")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf8")
    os.replace(temp, path)


def table(path, rows):
    if not rows:
        raise ValueError("Cannot infer schema from empty observations")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    with tmp.open("w", newline="", encoding="utf8") as f:
        out = csv.DictWriter(f, fieldnames=list(rows[0]))
        out.writeheader()
        out.writerows(rows)
    os.replace(tmp, path)


def git(*args):
    return subprocess.check_output(["git", *args], encoding="utf8").strip()


def cohorts():
    splits = {k: list(range(a, b + 1)) for k, (a, b) in read(PROTOCOL)["cohorts"].items()}
    if sum(map(len, splits.values())) != len(set().union(*splits.values())):
        raise ValueError("Cohorts overlap")
    return splits


def allocate(method, seed, root=ROOT):
    base = root / "runs"
    base.mkdir(parents=True, exist_ok=True)
    for i in range(1, 10000):
        loc = base / f"v02a-T3-{method}-K16-s{seed}-r{i:02d}"
        try:
            loc.mkdir()
            return loc
        except FileExistsError:
            continue
    raise RuntimeError("Attempt IDs exhausted")


def source(seed, k=16):
    if seed not in read(PROTOCOL)["source_seeds"]:
        raise ValueError("Source seed outside protocol")
    key = f"T1-K{k}-s{seed}"
    f = Path("checkpoints/synergy_budget") / key / "policy_1500.pt"
    pin = read(Path("results/synergy_budget/cells") / key / "points.json")[-1]
    if pin["iteration"] != 1500 or sha(f) != pin["sha256"]:
        raise ValueError(f"Parent policy checksum mismatch {key}")
    return {"uri":f.as_posix(),"sha256":pin["sha256"]}


def event(loc, stage, transitions=0, wall_s=None, **info):
    with (loc / "events.jsonl").open("a", encoding="utf8") as f:
        f.write(json.dumps(dict(stage=stage, transitions=int(transitions),wall_s=wall_s,
                                time_unix=time.time(),**info),allow_nan=False)+"\n")


@contextmanager
def attempt(method, seed, config=None):
    loc = allocate(method, seed)
    cfg = dict(method=method, seed=seed, **(config or {}))
    write(loc / "config.json", cfg)
    bank = Path("artifacts/v0_1r/banks/reach_only/K16/seed11/bank.npz")
    now = time.perf_counter()
    m = dict(schema_version="1.0",study_id="rsi_v02a",run_id=loc.name, method=method,
             train_seed=seed, status="RUNNING",git_commit=git("rev-parse","HEAD"),
             git_dirty=bool(git("status","--porcelain","--","experiments","myo_rsi","tests")),
             config_sha256=sha(loc/"config.json"),protocol_sha256=sha(PROTOCOL),
             cohorts=read(PROTOCOL)["cohorts"],source_checkpoint=None,synergy_bank=None,
             environment=None,final_checkpoint=None,failure_or_exclusion_reason=None)
    write(loc/"manifest.json",m)
    event(loc,"start")
    try:
        if m["git_dirty"]:
            raise ValueError("Uncommitted study code")
        m["source_checkpoint"]=source(seed)
        m["synergy_bank"]={"uri":bank.as_posix(),"sha256":sha(bank)}
        write(loc/"manifest.json",m)
        yield loc,m
    except BaseException as exc:
        m["status"]="INTERRUPTED" if isinstance(exc,KeyboardInterrupt) else "FAILED"
        m["failure_or_exclusion_reason"]=repr(exc)
        raise
    else:
        m["status"]="COMPLETED"
    finally:
        event(loc,"end",wall_s=time.perf_counter()-now,status=m["status"])
        events=[json.loads(s) for s in (loc/"events.jsonl").read_text().splitlines()]
        m["budget"]={k:sum(e["transitions"] for e in events if e["stage"]==k) for k in {e["stage"] for e in events} if k not in {"start","end"}}
        m["wall_s"]=time.perf_counter()-now
        write(loc/"manifest.json",m)


def completed(method,seed):
    paths=sorted((ROOT/"runs").glob(f"v02a-T3-{method}-K16-s{seed}-r*/manifest.json"))
    valid=[p.parent for p in paths if read(p)["status"]=="COMPLETED"]
    if not valid:
        raise FileNotFoundError(f"No successful {method} seed {seed}")
    return valid[-1]
