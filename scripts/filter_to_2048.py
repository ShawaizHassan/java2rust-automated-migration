#!/usr/bin/env python3
# Filter JSONL to keep only records with Java.bug_source_code <= MAX_TOKENS.
# Fast: threaded, char pre-check, optional tiktoken for accuracy.

import os, re, itertools, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import jsonlines
from tqdm.auto import tqdm

DEFAULT_IN  = "/data/hassan/java2rust-automated-migration/data/raw/test.jsonl"
DEFAULT_OUT = "/data/hassan/java2rust-automated-migration/data/processed/test_2048.jsonl"

def count_tokens_fast(text: str) -> int:
    # Very fast heuristic: ~1 token per 4 chars
    return max(1, len(text) // 4)

def make_token_counter(use_tiktoken: bool):
    if not use_tiktoken:
        return count_tokens_fast
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return lambda s: len(enc.encode(s))
    except Exception:
        # Fallback silently to fast estimate if tiktoken unavailable
        return count_tokens_fast

def process_record(rec, max_tokens: int, char_hint: int, count_tokens):
    j = rec.get("Java")
    if not isinstance(j, dict):
        return None, "invalid"
    src = j.get("bug_source_code")
    if not isinstance(src, str) or not src.strip():
        return None, "invalid"
    if len(src) > char_hint:                # quick reject before tokenization
        return None, "char"
    if count_tokens(src) > max_tokens:
        return None, "tokens"
    return rec, None

def batched(iterable, n):
    it = iter(iterable)
    while True:
        batch = list(itertools.islice(it, n))
        if not batch:
            break
        yield batch

def main():
    ap = argparse.ArgumentParser(description="Threaded JSONL filter for Java.bug_source_code token length")
    ap.add_argument("--input", default=DEFAULT_IN, help="Input JSONL path")
    ap.add_argument("--output", default=DEFAULT_OUT, help="Output JSONL path")
    ap.add_argument("--max-tokens", type=int, default=2048, help="Keep if tokens <= this")
    ap.add_argument("--use-tiktoken", action="store_true", help="Use tiktoken for accurate counting")
    ap.add_argument("--workers", type=int, default=max(4, (os.cpu_count() or 8)), help="Thread count")
    ap.add_argument("--batch-size", type=int, default=1000, help="Records per thread batch")
    args = ap.parse_args()

    in_path  = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    # Rough char limit to pre-filter (≈ 4 chars per token)
    char_hint = args.max_tokens * 4
    count_tokens = make_token_counter(args.use_tiktoken)

    total = kept = 0
    skipped_char = skipped_tokens = skipped_invalid = 0

    with jsonlines.open(in_path, "r") as reader, \
         jsonlines.open(out_path, "w") as writer, \
         ThreadPoolExecutor(max_workers=args.workers) as ex:

        for batch in tqdm(batched(reader, args.batch_size), desc="Filtering", unit="batch"):
            # Submit work
            futures = (ex.submit(process_record, rec, args.max_tokens, char_hint, count_tokens) for rec in batch)
            for fut in futures:
                rec, reason = fut.result()
                total += 1
                if rec is not None:
                    writer.write(rec)
                    kept += 1
                else:
                    if reason == "char": skipped_char += 1
                    elif reason == "tokens": skipped_tokens += 1
                    else: skipped_invalid += 1

    print(f"\nDone → {out_path}")
    print(f"Total:   {total}")
    print(f"Kept:    {kept}")
    print(f"Skipped: {skipped_char} (char) + {skipped_tokens} (tokens) + {skipped_invalid} (invalid) = {skipped_char+skipped_tokens+skipped_invalid}")

if __name__ == "__main__":
    main()