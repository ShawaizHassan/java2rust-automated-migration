#!/usr/bin/env python3
"""
Filter a JSONL by test results from a CSV (idx-aligned).

Keeps only rows where:
  built == 1 AND tests_passed == tests_total   (i.e., 100% pass)

Assumes CSV has columns: idx,built,tests_passed,tests_total,pass_rate
and that `idx` is 0-based row index of the JSONL.

Usage:
  python keep_ut_passed.py \
    --jsonl /data/hassan/java2rust-automated-migration/data/processed/test_2048.jsonl \
    --csv   /data/hassan/java2rust-automated-migration/scripts/runtime_inouts.csv \
    --out   /data/hassan/java2rust-automated-migration/data/processed/ut_passed_test.jsonl
"""

import argparse, csv
from pathlib import Path
import json

DEFAULT_JSONL = "/data/hassan/java2rust-automated-migration/data/processed/test_2048.jsonl"
DEFAULT_CSV   = "/data/hassan/java2rust-automated-migration/scripts/out/runtime_inouts.csv"
DEFAULT_OUT   = "/data/hassan/java2rust-automated-migration/data/processed/final_test.jsonl"

def load_passing_indices(csv_path: Path) -> set[int]:
    keep = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                idx = int(row["idx"])
                built = int(row.get("built", "0"))
                tp = int(float(row.get("tests_passed", "0")))
                tt = int(float(row.get("tests_total", "0")))
            except Exception:
                continue
            if built == 1 and tt >= 0 and tp == tt:
                keep.add(idx)
    return keep

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default=DEFAULT_JSONL, help="Input JSONL aligned to CSV idx")
    ap.add_argument("--csv",   default=DEFAULT_CSV,   help="CSV with columns idx,built,tests_passed,tests_total")
    ap.add_argument("--out",   default=DEFAULT_OUT,   help="Output JSONL with only 100%%-passed rows")
    ap.add_argument("--idx-origin", type=int, default=0,
                    help="0 if CSV idx is zero-based (default), 1 if one-based")
    args = ap.parse_args()

    in_path  = Path(args.jsonl)
    csv_path = Path(args.csv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"JSONL not found: {in_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    keep_idxs = load_passing_indices(csv_path)
    if args.idx_origin == 1:
        # Convert one-based -> zero-based
        keep_idxs = {i - 1 for i in keep_idxs}

    kept = total = 0
    with open(in_path, "r", encoding="utf-8") as jin, \
         open(out_path, "w", encoding="utf-8") as jout:
        for i, line in enumerate(jin):
            total += 1
            if i in keep_idxs:
                jout.write(line)
                kept += 1

    print(f"✅ Done. Kept {kept}/{total} rows → {out_path}")

if __name__ == "__main__":
    main()