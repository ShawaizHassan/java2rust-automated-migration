#!/usr/bin/env python3
"""
Compile-time error logger for LLM-generated Rust.

Reads an input JSONL where each record contains:
  - "java"
  - "trans_rust"

Writes an output JSONL with:
  {"java": ..., "trans_rust": ..., "hidden_unit_tests": ..., "refine_unit_tests": ...,
   "compile_ok": 0/1, "error": "...compiler stderr (first 4 UNIQUE errors or fallback) ..."}

Notes:
- Extracts Rust from fenced blocks ```...``` if present.
- Tries rustc in BIN mode first, then LIB mode (same as your evaluator idea).
- Records compiler stderr (no categorization).
- Keeps first 4 unique `error[...]` blocks (dedupe by error code); if none found, falls back to truncation by max_err_chars.
"""

import os
import re
import jsonlines
import tempfile
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# ---------- fence extraction ----------
FENCE = re.compile(r"```(?:rust|rs|RUST|Rust)?\s*([\s\S]*?)```", re.MULTILINE)

def extract_raw_rust(text: str) -> str:
    if not text:
        return ""
    blocks = FENCE.findall(text)
    return blocks[-1].strip() if blocks else text.strip()


# ---------- helper: keep first N UNIQUE rustc error blocks ----------
# Start of an error block (covers both "error:" and "error[E0123]:")
ERR_BLOCK_START = re.compile(r"^error(\[[A-Za-z0-9_]+\])?:", re.MULTILINE)
# Capture error code if present: "error[E0425]:" -> "E0425"
ERR_CODE = re.compile(r"^error\[([A-Za-z0-9_]+)\]:", re.MULTILINE)

def keep_first_error_blocks(stderr: str, n: int = 4, max_chars: int | None = 10000) -> str:
    """
    Keep the first `n` UNIQUE compiler error blocks if present.
    Uniqueness is based on error code (e.g., E0425). If no code exists, a generic
    "NO_CODE" bucket is used.

    Fallback: truncate to `max_chars` if no blocks are found.
    """
    if not stderr:
        return ""

    starts = [m.start() for m in ERR_BLOCK_START.finditer(stderr)]
    if not starts:
        if max_chars is not None and len(stderr) > max_chars:
            return stderr[:max_chars] + "\n...[TRUNCATED]..."
        return stderr

    # Add sentinel end
    starts.append(len(stderr))

    kept = []
    seen_codes = set()

    for i in range(len(starts) - 1):
        block = stderr[starts[i]:starts[i + 1]].rstrip()
        if not block:
            continue

        m = ERR_CODE.search(block)
        code = m.group(1) if m else "NO_CODE"

        if code in seen_codes:
            continue

        seen_codes.add(code)
        kept.append(block)

        if len(kept) >= n:
            break

    out = "\n\n".join(kept).rstrip()

    # If we somehow kept nothing, fall back to truncation
    if not out:
        if max_chars is not None and len(stderr) > max_chars:
            return stderr[:max_chars] + "\n...[TRUNCATED]..."
        return stderr.rstrip()

    # Hard cap
    if max_chars is not None and len(out) > max_chars:
        out = out[:max_chars].rstrip() + "\n...[TRUNCATED]..."

    return out


# ---------- rustc compile check ----------
def rustc_check(code: str, timeout: int = 8):
    """
    Returns (ok: bool, stderr: str).
    Tries:
      1) rustc main.rs -o a.out   (binary)
      2) rustc --crate-type=lib main.rs (library)
    """
    if not code.strip():
        return False, "EMPTY_CODE"

    with tempfile.TemporaryDirectory() as tmpdir:
        src = os.path.join(tmpdir, "main.rs")
        with open(src, "w", encoding="utf-8") as f:
            f.write(code.rstrip() + "\n")

        # BIN mode
        try:
            p = subprocess.run(
                ["rustc", "main.rs", "-o", "a.out"],
                cwd=tmpdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )
            if p.returncode == 0:
                return True, p.stderr or ""
            bin_stderr = p.stderr or ""
        except subprocess.TimeoutExpired:
            return False, "TIMEOUT"

        # LIB mode (sometimes solutions are library-like)
        try:
            p2 = subprocess.run(
                ["rustc", "--crate-type=lib", "main.rs"],
                cwd=tmpdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )
            if p2.returncode == 0:
                return True, p2.stderr or ""
            return False, (p2.stderr or bin_stderr or "")
        except subprocess.TimeoutExpired:
            return False, "TIMEOUT"


def process_one(rec: dict, timeout: int, max_err_chars: int | None):
    java = rec.get("java", "")
    trans_rust = rec.get("trans_rust", "")
    hidden_unit_tests = rec.get("hidden_unit_tests", [])
    refine_unit_tests = rec.get("refine_unit_tests", [])

    code = extract_raw_rust(trans_rust)
    ok, err = rustc_check(code, timeout=timeout)

    # Keep first 4 UNIQUE error blocks; if none found -> fallback truncation by max_err_chars
    if not ok:
        err = keep_first_error_blocks(err or "", n=4, max_chars=max_err_chars)

    return {
        "java": java,
        "trans_rust": trans_rust,
        "hidden_unit_tests": hidden_unit_tests,
        "refine_unit_tests": refine_unit_tests,
        "compile_ok": 1 if ok else 0,
        "error": "" if ok else (err or "UNKNOWN_ERROR"),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/data/hassan/java2rust-automated-migration/experiments/lustre/no-refinement/ragcat-references-gptsummary/deepseek-coder-1.3b-instruct.jsonl", help="Input JSONL with keys: java, trans_rust")
    ap.add_argument("--out", default="/data/hassan/java2rust-automated-migration/experiments/lustre/error-logs/no-pass/ragcat-references-gptsummary/deepseek-coder-1.3b-instruct.jsonl", help="Output JSONL path")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=8, help="rustc timeout (seconds) per attempt")
    ap.add_argument("--max_err_chars", type=int, default=10000, help="truncate stderr to this many chars (use -1 to disable)")
    args = ap.parse_args()

    max_err_chars = None if args.max_err_chars == -1 else args.max_err_chars

    in_path = Path(args.input)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    with jsonlines.open(in_path, mode="r") as reader:
        for rec in reader:
            records.append(rec)

    results = [None] * len(records)

    with ThreadPoolExecutor(max_workers=args.workers) as exe:
        futures = {}
        for idx, rec in enumerate(records):
            fut = exe.submit(process_one, rec, args.timeout, max_err_chars)
            futures[fut] = idx

        for fut in as_completed(futures):
            idx = futures[fut]
            results[idx] = fut.result()

    # Write output
    with jsonlines.open(out_path, mode="w") as w:
        for r in results:
            w.write(r)

    # Print quick summary
    total = len(results)
    ok_count = sum(1 for r in results if r["compile_ok"] == 1)
    print("========== COMPILE SUMMARY ==========")
    print(f"Samples: {total}")
    print(f"Compile OK: {ok_count}/{total} = {ok_count/total*100:.2f}%")
    print("====================================")


if __name__ == "__main__":
    main()
