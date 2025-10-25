#!/usr/bin/env python3
"""
Java -> Rust translator (parallel client) with:
- test cases preserved,
- original Java preserved (structured if available),
- ground-truth Rust ('rust_gt') preserved,
- translated Rust ('rust') stored as plain code (no ``` fences).
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

def translate_one(client, model, temperature, record, ts, max_tokens=2048, timeout=60):
    rid = record.get("id")

    # Prefer structured Java object if present
    java_obj = record.get("Java")
    java_block = record.get("java") or java_obj
    java_code = java_block.get("bug_source_code") if isinstance(java_block, dict) else java_block
    if not java_code or not str(java_code).strip():
        return None

    # Ground-truth Rust from dataset (if present)
    rust_obj = record.get("Rust")
    rust_gt = None
    if isinstance(rust_obj, dict):
        rust_gt = rust_obj.get("bug_source_code")
    elif isinstance(rust_obj, str):
        rust_gt = rust_obj  # if some datasets store it as plain string

    # Preserve hidden tests
    hidden_tests = record.get("hidden_unit_tests")

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
            rust_code = strip_md_fences(resp.choices[0].message.content)

            # Build ordered output: Java -> rust -> rust_gt -> metadata -> tests
            out = {}
            if java_obj is not None:
                out["Java"] = java_obj
            else:
                out["java"] = java_code

            out["rust"] = rust_code                     # translated
            if rust_gt is not None:
                out["rust_gt"] = rust_gt                # ground-truth from dataset

            out["id"] = rid
            out["model"] = model
            out["timestamp"] = ts

            if hidden_tests is not None:
                out["hidden_unit_tests"] = hidden_tests

            return out

        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))

    # Final failure: still return ordered fields with error string in 'rust'
    out = {}
    if java_obj is not None:
        out["Java"] = java_obj
    else:
        out["java"] = java_code

    out["rust"] = f"ERROR: {last_err}"
    if rust_gt is not None:
        out["rust_gt"] = rust_gt
    out["id"] = rid
    out["model"] = model
    out["timestamp"] = ts
    if hidden_tests is not None:
        out["hidden_unit_tests"] = hidden_tests
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/data/hassan/java2rust-automated-migration/data/processed/test_2048.jsonl")
    ap.add_argument("--output", default="/data/hassan/java2rust-automated-migration/outputs")
    ap.add_argument("--model", default="Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default="hassan2024")
    ap.add_argument("--limit", type=int, default=0)
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
                if not result.get("id"):
                    result["id"] = f"rec_{pbar.n + 1}"
                writer.write(result)
            pbar.update(1)

    print(f"✅ Done! Translated {total} records → {output_path}")

if __name__ == "__main__":
    main()