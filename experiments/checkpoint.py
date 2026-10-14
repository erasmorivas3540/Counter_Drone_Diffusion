"""Append-only checkpoints so long searches can resume after an interruption.

Each finished piece of work is written as one JSON line and flushed to disk at once, so
a crash, a dropped connection or a killed kernel loses at most the job in progress.
Rerunning the same script reads the file back and skips every key already present.

    ck = Checkpoint("q3_structured_scan")          # results/checkpoints/q3_structured_scan.jsonl
    if key not in ck:
        ck.add(key, value)                         # value: anything JSON can hold
    ck[key]                                        # read back

Keys are strings; use make_key(...) to build them from tuples of numbers and labels.
Delete the .jsonl file (or run the script with --fresh) to start over.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "results" / "checkpoints"


def make_key(*parts) -> str:
    out = []
    for p in parts:
        if isinstance(p, (list, tuple)):
            out.append("(" + ",".join(f"{float(x):.6f}" for x in p) + ")")
        elif isinstance(p, float):
            out.append(f"{p:.6f}")
        else:
            out.append(str(p))
    return "|".join(out)


class Checkpoint:
    def __init__(self, name: str, fresh: bool = False):
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        self.path = CHECKPOINT_DIR / f"{name}.jsonl"
        if fresh and self.path.exists():
            self.path.unlink()
        self.data: dict = {}
        if self.path.exists():
            with open(self.path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        break          # a half-written last line from a crash: ignore it
                    self.data[rec["key"]] = rec["value"]

    def __contains__(self, key) -> bool:
        return key in self.data

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self) -> int:
        return len(self.data)

    def get(self, key, default=None):
        return self.data.get(key, default)

    def add(self, key: str, value) -> None:
        self.data[key] = value
        with open(self.path, "a") as f:
            f.write(json.dumps({"key": key, "value": value}) + "\n")
            f.flush()
            os.fsync(f.fileno())
