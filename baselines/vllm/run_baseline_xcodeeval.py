#!/usr/bin/env python3
"""
Java -> Rust translator (parallel client) with collapsible JSONL parts:
- Java           (object; wraps string into {"bug_source_code": ...} if needed)
- Trans_Rust     (object: {"code": ...})
- GT_Rust        (object: {"code": ...})
- hidden_unit_tests (array)
"""

import argparse, time, re
from datetime import datetime
from pathlib import Path
import jsonlines
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def build_prompt(java_code: str):
    system_prompt = """You are an expert software engineer specializing in automated code migration from Java to Rust.
- Output ONLY Rust code unless asked to explain.
- Preserve logic and input/output behavior.
- Prefer idiomatic Rust: ownership, borrowing, Result/Option, error handling.
- Replace Java collections with Rust equivalents (Vec, HashMap, etc.).
- Avoid unsafe unless strictly necessary.
- Include use statements and a minimal main/test if needed."""
    user_prompt = (
        "Here is a Java code snippet. Translate it to Rust:\n"
        "```java\n"
        f"{java_code}\n"
        "```\n\n"
        "Return only Rust code:\n\n"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

def strip_md_fences(text: str) -> str:
    """Return code inside ```...``` if present; otherwise the trimmed text."""
    if not text:
        return text
    m = re.match(r"^\s*```(?:\w+)?\s*\n(.*)\n```[\s]*$", text, flags=re.DOTALL)
    return m.group(1).strip() if m else text.strip()

def as_java_object(java_obj_or_str):
    """Ensure Java is always an object with at least 'bug_source_code'."""
    if isinstance(java_obj_or_str, dict):
        # Make sure there's a bug_source_code key if dataset uses a different key name
        if "bug_source_code" in java_obj_or_str:
            return java_obj_or_str
        # fallback: put entire dict under 'meta' and keep best-effort code string if present
        out = {"meta": java_obj_or_str}
        # try common keys
        for k in ("code", "source", "src"):
            if k in java_obj_or_str and isinstance(java_obj_or_str[k], str):
                out["bug_source_code"] = java_obj_or_str[k]
                break
        return out
    else:
        return {"bug_source_code": str(java_obj_or_str) if java_obj_or_str is not None else ""}

def as_code_object(code_str):
    """Wrap code in an object so it’s collapsible in editors."""
    return {"code": code_str if code_str is not None else None}

def translate_one(client, model, temperature, record, ts, max_tokens=2048, timeout=60):
    # ---- Extract Java (normalize to object) ----
    java_raw = record.get("Java") or record.get("java")
    java_code = None
    if isinstance(java_raw, dict):
        java_code = java_raw.get("bug_source_code")
    else:
        java_code = java_raw

    if not java_code or not str(java_code).strip():
        return None
    java_obj = as_java_object(java_raw if java_raw is not None else java_code)

    # ---- Ground-truth Rust (normalize to object) ----
    rust_raw = record.get("Rust")
    if isinstance(rust_raw, dict):
        rust_gt_code = rust_raw.get("bug_source_code")
    else:
        rust_gt_code = rust_raw if isinstance(rust_raw, str) else None
    gt_rust_obj = as_code_object(rust_gt_code)

    # ---- Tests ----
    hidden_tests = record.get("hidden_unit_tests")

    # ---- Call model with simple retry ----
    last_err = None
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=build_prompt(java_code),
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            trans_rust_code = strip_md_fences(resp.choices[0].message.content)
            # Build ordered output for collapsible JSONL
            out = {}
            out["Java"] = java_obj
            out["Trans_Rust"] = as_code_object(trans_rust_code)
            out["GT_Rust"] = gt_rust_obj
            if hidden_tests is not None:
                out["hidden_unit_tests"] = hidden_tests
            return out
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))

    # On failure, still return collapsible structure (Trans_Rust contains error)
    out = {}
    out["Java"] = java_obj
    out["Trans_Rust"] = as_code_object(f"ERROR: {last_err}")
    out["GT_Rust"] = gt_rust_obj
    if hidden_tests is not None:
        out["hidden_unit_tests"] = hidden_tests
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/data/hassan/java2rust-automated-migration/data/raw/test.jsonl")
    ap.add_argument("--output", default="/data/hassan/java2rust-automated-migration/outputs")
    ap.add_argument("--model", default="Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default="hassan2024")
    ap.add_argument("--limit", type=int, default=0)  # 0 = all
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=6, help="Parallel request workers")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--timeout", type=int, default=60)
    args = ap.parse_args()

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    input_path = Path(args.input)
    output_path = Path(args.output) / f"translated_{args.model}_{ts}.jsonl"

    # Load up to --limit
    records = []
    with jsonlines.open(input_path) as reader:
        for i, rec in enumerate(reader):
            if args.limit and i >= args.limit:
                break
            records.append(rec)

    total = len(records)
    if total == 0:
        print("No records to process.")
        return

    futures = []
    with jsonlines.open(output_path, mode="w") as writer, \
         ThreadPoolExecutor(max_workers=args.workers) as ex, \
         tqdm(total=total, desc="Translating", unit="rec") as pbar:

        for rec in records:
            fut = ex.submit(
                translate_one,
                client, args.model, args.temperature, rec, ts,
                args.max_tokens, args.timeout
            )
            futures.append(fut)

        for fut in as_completed(futures):
            result = fut.result()
            if result is not None:
                writer.write(result)  # already in desired order/shape
            pbar.update(1)

    print(f"✅ Done! Translated {total} records → {output_path}")

if __name__ == "__main__":
    main()