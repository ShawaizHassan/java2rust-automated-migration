#!/usr/bin/env python3
import argparse, json, os, re, shutil, subprocess, tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

RE_FENCE_BLOCK = re.compile(r"```(?:rust)?\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE)
RE_MISSING_MAIN = re.compile(r"\bmain function not found\b|\b`main` function not found\b", re.I)

_WORKDIR = None

def _init_worker(base_tmp: str):
    global _WORKDIR
    pid = os.getpid()
    _WORKDIR = os.path.join(base_tmp, f"w_{pid}")
    os.makedirs(_WORKDIR, exist_ok=True)

def refine_rust(code: str) -> str:
    if not code:
        return ""
    code = code.lstrip("\ufeff").strip()
    m = RE_FENCE_BLOCK.search(code)
    if m:
        code = m.group(1).strip()
    if "`" in code:
        code = code.replace("`", "")
    return code.strip()

def rustc_ok(code: str, timeout: int) -> bool:
    if not code.strip():
        return False

    src = os.path.join(_WORKDIR, "main.rs")
    out_bin = os.path.join(_WORKDIR, "out_bin")
    out_lib = os.path.join(_WORKDIR, "out_lib.rlib")

    for p in (out_bin, out_lib):
        try:
            if os.path.exists(p):
                os.remove(p)
        except OSError:
            pass

    with open(src, "w", encoding="utf-8", errors="ignore") as f:
        f.write(code)

    base = ["rustc", "--edition=2021", "-C", "opt-level=0", "-C", "debuginfo=0", src]

    try:
        p = subprocess.run(base + ["-o", out_bin], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           text=True, timeout=timeout)
        if p.returncode == 0:
            return True

        err = (p.stderr or "").strip()
        if RE_MISSING_MAIN.search(err):
            p2 = subprocess.run(base + ["--crate-type", "lib", "-o", out_lib],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, timeout=timeout)
            return p2.returncode == 0

        return False
    except subprocess.TimeoutExpired:
        return False

def _task(idx_line_timeout):
    idx, line, timeout = idx_line_timeout
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
    except Exception:
        return None

    rust_raw = obj.get("Rust", "")
    rust_ref = refine_rust(rust_raw)

    if rustc_ok(rust_ref, timeout):
        # overwrite with refined compilable Rust
        obj["Rust"] = rust_ref
        return json.dumps(obj, ensure_ascii=False)
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/data/hassan/java2rust-automated-migration/data/retrieval/processed/ragcorpus_512.jsonl")
    ap.add_argument("--output", default="/data/hassan/java2rust-automated-migration/data/retrieval/processed/ragcorpus.jsonl")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--timeout", type=int, default=8)
    ap.add_argument("--max_items", type=int, default=0)
    args = ap.parse_args()

    if shutil.which("rustc") is None:
        raise SystemExit("ERROR: rustc not found in PATH.")

    base_tmp = tempfile.mkdtemp(prefix="rustc_filter_")
    kept = total = 0

    with open(args.input, "r", encoding="utf-8", errors="ignore") as fin, \
         open(args.output, "w", encoding="utf-8") as fout, \
         ProcessPoolExecutor(max_workers=args.jobs, initializer=_init_worker, initargs=(base_tmp,)) as ex:

        futures = []
        for idx, line in enumerate(fin):
            if args.max_items and idx >= args.max_items:
                break
            total += 1
            futures.append(ex.submit(_task, (idx, line, args.timeout)))

        for fut in as_completed(futures):
            out_line = fut.result()
            if out_line is not None:
                fout.write(out_line + "\n")
                kept += 1

    shutil.rmtree(base_tmp, ignore_errors=True)
    print(f"Done. total={total} kept={kept} removed={total-kept} output={args.output}")

if __name__ == "__main__":
    main()
