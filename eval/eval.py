#!/usr/bin/env python3
"""
Unified evaluator:
- CSR via rustc
- CA via cargo build + hidden tests
- Unsafe metrics (UR + URLOC)
- Short error categories (compile + runtime), deduplicated
"""

import os, re, time, shutil, tempfile, subprocess, jsonlines
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===================== IMPORT UNSAFE DETECTOR =====================
from unsafe_check import analyze_unsafe_java_to_rust


# SHORT ERROR CLASSIFIER
# ---------------------

ERR_PATTERNS = {
    # --- Java2Rust specific (compile) ---
    "borrow_lifetime": re.compile(
        r"(cannot borrow .* as (mutable|immutable) because it is also borrowed|"
        r"borrowed value does not live long enough|"
        r"lifetime may not live long enough|"
        r"explicit lifetime required|"
        r"use of moved value|"
        r"cannot move out of .* which is behind a (shared|mutable) reference)",
        re.I,
    ),

    # --- General compile (but very common in translation) ---
    "type_trait_generic": re.compile(
        r"(mismatched types|expected .* found|"
        r"the trait bound .* is not satisfied|"
        r"`.*` is not implemented for `.*`|"
        r"type annotations needed|cannot infer type)",
        re.I,
    ),
    "unresolved": re.compile(
        r"(cannot find (value|function|crate|module|type)|"
        r"unresolved import|"
        r"failed to resolve: use of undeclared (crate|type|module))",
        re.I,
    ),
    "syntax": re.compile(
        r"(unclosed delimiter|this file contains an unclosed delimiter|"
        r"unexpected token|expected .*? found|error: expected)",
        re.I,
    ),

    # --- Runtime / execution ---
    "panic": re.compile(
        r"(panicked at|thread 'main' panicked|panic!|index out of bounds|assertion failed)",
        re.I,
    ),
    "parse_input": re.compile(r"(ParseIntError|invalid digit found in string)", re.I),
    "timeout": re.compile(r"\bTIMEOUT\b", re.I),
}


def classify_error(stderr: str):
    """Return short error label OR None for no meaningful error."""
    if not stderr or not stderr.strip():
        return None  # FIX: do not treat empty stderr as error

    for name, pat in ERR_PATTERNS.items():
        if pat.search(stderr):
            return name

    return "unknown"


# --- NEW: extract unique Rust compiler error codes like E0382 ---
RE_ECODE = re.compile(r"error\[(E\d{4})\]")

def extract_error_codes(stderr: str):
    """Return unique Rust error codes (E####) in first-seen order."""
    if not stderr or not stderr.strip():
        return []
    codes = RE_ECODE.findall(stderr)
    return list(dict.fromkeys(codes))


# cleaner for fenced Rust
# -----------------------------

FENCE = re.compile(r"```(?:rust|rs|RUST|Rust)?\s*([\s\S]*?)```", re.MULTILINE)

def extract_raw_rust(text: str) -> str:
    if not text:
        return ""
    block = FENCE.findall(text)
    return block[-1].strip() if block else text.strip()


def clean_rust_for_ca(code: str) -> str:
    return code.strip() + "\n"


# rustc CSR with no leftover files
# -------------------------------

def rustc_check(code: str, timeout=8):
    with tempfile.TemporaryDirectory() as tmpdir:
        src = os.path.join(tmpdir, "main.rs")
        with open(src, "w") as f:
            f.write(code)

        # BIN MODE
        try:
            p = subprocess.run(
                ["rustc", "main.rs", "-o", "a.out"],
                cwd=tmpdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, timeout=timeout
            )
            if p.returncode == 0:
                return 0, p.stderr
        except subprocess.TimeoutExpired:
            return 124, "TIMEOUT"

        # LIB MODE
        try:
            p = subprocess.run(
                ["rustc", "--crate-type=lib", "main.rs"],
                cwd=tmpdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, timeout=timeout
            )
            return p.returncode, p.stderr
        except subprocess.TimeoutExpired:
            return 124, "TIMEOUT"


# Cargo for CA execution
# ---------------------

TOML_BIN = """[package]
name="proj"
version="0.1.0"
edition="2021"

[[bin]]
name="proj"
path="src/main.rs"
"""

def create_project(code: str):
    tmp = tempfile.mkdtemp()
    proj = os.path.join(tmp, "proj")
    os.makedirs(os.path.join(proj, "src"))
    with open(os.path.join(proj, "Cargo.toml"), "w") as f:
        f.write(TOML_BIN)
    with open(os.path.join(proj, "src", "main.rs"), "w") as f:
        f.write(code)
    return tmp, proj


def cargo_build(proj, timeout):
    try:
        p = subprocess.run(
            ["cargo", "build", "--release", "--quiet", "--offline"],
            cwd=proj, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=timeout
        )
        return p.returncode, p.stderr
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"


def run_binary(path, inp, timeout=3):
    try:
        p = subprocess.run(
            [path],
            input=inp.encode("utf-8", errors="replace"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout
        )
        # FIX: avoid UnicodeDecodeError from non-UTF8 output
        out = p.stdout.decode("utf-8", errors="replace")
        err = p.stderr.decode("utf-8", errors="replace")
        return out, err
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT"


# Worker evaluator (patched)
# -------------------------

def evaluate_one(idx, rec, timeout):

    result = {
        "id": idx,
        "csr": 0,
        "compiled": 0,
        "compile_errors": [],
        "compile_error_codes": [],  # NEW
        "ca": 0,
        "passed": 0,
        "total": 0,
        "runtime_errors": [],

        "unsafe_count": 0,
        "unsafe_loc_rate": 0.0,
    }

    raw = rec.get("trans_rust") or ""
    code_raw = extract_raw_rust(raw)

    if not code_raw:
        result["compile_errors"] = ["empty"]
        return result

    # === UNSAFE ===
    ur, urloc = analyze_unsafe_java_to_rust(code_raw)
    result["unsafe_count"] = ur          # 0.0 or 1.0
    result["unsafe_loc_rate"] = urloc    # fraction of unsafe lines


    # === CSR ===
    rc, err = rustc_check(code_raw)
    err_label = classify_error(err)

    # NEW: always extract codes from rustc stderr (even if repeated -> unique)
    result["compile_error_codes"] = extract_error_codes(err)
    
    if rc != 0:
        if err_label:
            result["compile_errors"] = [err_label]
    else:
        result["compile_errors"] = []  # FIX: no errors on success

    result["compiled"] = 1 if rc == 0 else 0
    result["csr"] = result["compiled"]

    if rc != 0:
        return result

    # === CA ===
    code_ca = clean_rust_for_ca(code_raw)
    tmp, proj = create_project(code_ca)

    rc_build, err_build = cargo_build(proj, timeout)
    if rc_build != 0:
        e = classify_error(err_build)
        if e:
            result["compile_errors"].append(e)

        # NEW: also collect error codes from cargo build diagnostics
        result["compile_error_codes"].extend(extract_error_codes(err_build))
        result["compile_error_codes"] = list(dict.fromkeys(result["compile_error_codes"]))

        # FIX: dedupe compile errors
        result["compile_errors"] = list(dict.fromkeys(result["compile_errors"]))
        shutil.rmtree(tmp, ignore_errors=True)
        return result

    bin_path = os.path.join(proj, "target", "release", "proj")

    tests = rec.get("hidden_unit_tests", [])
    result["total"] = len(tests)

    for t in tests:
        out, err2 = run_binary(bin_path, t["input"])

        if out is not None and out.strip() == t["output"][0].strip():
            result["passed"] += 1

        if err2:
            label = classify_error(err2)
            if label:
                result["runtime_errors"].append(label)

    # FIX: dedupe runtime errors
    result["runtime_errors"] = list(dict.fromkeys(result["runtime_errors"]))

    if result["passed"] == result["total"]:
        result["ca"] = 1

    shutil.rmtree(tmp, ignore_errors=True)
    return result



# MAIN
# ------

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/data/hassan/java2rust-automated-migration/experiments/baseline/no-refinement/Instruction/Qwen2.5-Coder-32B-Instruct.jsonl")
    ap.add_argument("--out", default="Qwen2.5-Coder-32B-Instruct.jsonl")
    ap.add_argument("--workers", default=8, type=int)
    ap.add_argument("--timeout", default=20, type=int)
    args = ap.parse_args()

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as exe, jsonlines.open(args.input) as reader:
        futures = [exe.submit(evaluate_one, idx, rec, args.timeout) for idx, rec in enumerate(reader)]
        for f in as_completed(futures):
            results.append(f.result())

    results.sort(key=lambda x: x["id"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(out_path, "w") as w:
        for r in results:
            w.write(r)

    csr = sum(r["csr"] for r in results)
    ca = sum(r["ca"] for r in results)
    total = len(results)

    unsafe_samples = sum(1 for r in results if r["unsafe_count"] > 0)
    avg_unsafe_loc = sum(r["unsafe_loc_rate"] for r in results) / total

    print("\n========== SUMMARY ==========")
    print(f"Samples: {total}")
    print(f"CSR (rustc): {csr}/{total} = {csr/total*100:.2f}%")
    print(f"CA (cargo tests): {ca}/{total} = {ca/total*100:.2f}%")
    print(f"Unsafe Rate: {unsafe_samples}/{total} = {unsafe_samples/total*100:.2f}%")
    print(f"Unsafe LOC Rate (avg): {avg_unsafe_loc*100:.2f}%")
    print("================================\n")


if __name__ == "__main__":
    main()