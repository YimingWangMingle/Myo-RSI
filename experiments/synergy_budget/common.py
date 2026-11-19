"""Shared files, immutable policy banks and experiment contract."""

from pathlib import Path

import numpy as np

from myo_rsi.bank import checksum
from myo_rsi.bank_study import BANKS, readcsv as readcsv
from myo_rsi.rerun_v01r import read, save as save, sha, table

HERE = Path(__file__).resolve().parent
LOCK = read(HERE / "protocol.json")
OUT = Path("results/synergy_budget")
ART = Path("artifacts/synergy_budget")
CKPT = Path("checkpoints/synergy_budget")
OLD = Path("results/v0_1r")
for folder in (OUT, ART, CKPT):
    folder.mkdir(exist_ok=True, parents=True)


def bank(k, seed):
    root = BANKS if k in (8, 16) else ART / "banks"
    p = root / f"K{k}" / f"seed{seed}" / "bank.npz"
    m = read(p.parent / "metrics.json")["reconstruction"]
    if sha(p.read_bytes()) != m["file_sha256"]:
        raise ValueError("Frozen bank file checksum mismatch")
    with np.load(p) as d:
        w, q, n = d["W"], d["q95"], d["actuator_names"].tolist()
    if checksum(w, n) != m["bank_hash"] or not np.isfinite(q).all() or min(q) <= 0:
        raise ValueError("Invalid bank coefficient scales or weights")
    return w, q, n, m


def seeds(group):
    first, last = LOCK[group]
    return list(range(first, last + 1))


def atomic_table(file, rows):
    tmp = file.with_suffix(".tmp")
    table(tmp, rows)
    tmp.replace(file)
