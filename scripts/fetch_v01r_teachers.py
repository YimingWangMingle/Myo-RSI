"""Fetch pinned MyoSuite release assets, verifying Git blobs and LFS identities."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = json.loads((ROOT / "results/v0_1r/run_lock.json").read_text())
REPO, REV = LOCK["hf_repo"], LOCK["hf_sha"]
BASE = "https://huggingface.co"
MODELS = {
    "myoHandReachRandom-v0": "model_1912.pt",
    "myoHandPoseRandom-v0": "model_24999.pt",
    "myoHandReachFixed-v0": "model_97.pt",
}


def download(url, target=None):
    args = [
        "curl.exe" if os.name == "nt" else "curl",
        "--fail",
        "--location",
        "--silent",
        "--show-error",
        "--retry",
        "4",
        "--retry-all-errors",
        "--connect-timeout",
        "25",
    ]
    if target is not None:
        args.extend(["--output", str(target)])
    data = subprocess.run([*args, url], capture_output=target is None, check=True)
    return None if target is not None else json.loads(data.stdout)


def fetch(env):
    folder = ROOT / "checkpoints/v0_1r/official" / env
    folder.mkdir(parents=True, exist_ok=True)
    tree = download(
        f"{BASE}/api/models/{REPO}/tree/{REV}/checkpoints/{env}?expand=true"
    )
    for name in ("manifest.json", MODELS[env]):
        entry = next(r for r in tree if r["path"] == f"checkpoints/{env}/{name}")
        n = entry["size"]

        def valid(file):
            if not file.is_file() or file.stat().st_size != n:
                return False
            data = file.read_bytes()
            if "lfs" in entry:
                return hashlib.sha256(data).hexdigest() == entry["lfs"]["oid"]
            return (
                hashlib.sha1(f"blob {n}\0".encode() + data).hexdigest() == entry["oid"]
            )

        dest = folder / name
        if valid(dest):
            continue
        part = dest.with_suffix(dest.suffix + ".part")
        download(f"{BASE}/{REPO}/resolve/{REV}/{entry['path']}", part)
        if not valid(part):
            raise ValueError(f"Source checksum mismatch for {name}")
        part.replace(dest)
        print(f"Verified {env}/{name}", flush=True)
    (folder / "tree.json").write_text(json.dumps(tree), encoding="utf8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("env", choices=list(MODELS), nargs="+")
    args = parser.parse_args()
    tag = download(f"{BASE}/api/models/{REPO}/revision/{LOCK['hf_tag']}")
    if tag["sha"] != REV:
        raise ValueError("Release tag resolution changed")
    for env in args.env:
        fetch(env)
