#!/usr/bin/env python3
"""
Generate PPT-ready CSVs for 100% stacked bar charts of *compile error buckets* (compile_errors).

You asked for:
  (A) RAG vs JavaRustTrans (1x Refinement)
  (B) JavaRustTrans (w/o Refinement) vs JavaRustTrans (1x Refinement)

Changes from your script:
  - Only TOP 5 categories (by total count within each comparison).
  - Buckets are renamed to readable labels (underscores -> spaces + Title Case, plus optional overrides).
  - Produces two sets of CSVs (wide_percent + wide_counts + long) for the two visualizations.

How to use in PowerPoint:
  - Insert -> Chart -> 100% Stacked Column
  - Edit Data -> paste contents of the *_wide_percent.csv
    (Rows=Categories, Columns=Methods)
"""

import json
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd

# -----------------------------
# INPUT FILES
# -----------------------------
RAG_FILES = [
    "/data/hassan/java2rust-automated-migration/results/baselines/rag/RAG/CodeLlama-7b-Instruct-hf.jsonl",
    "/data/hassan/java2rust-automated-migration/results/baselines/rag/RAG/CodeLlama-34b-Instruct-hf.jsonl",
    "/data/hassan/java2rust-automated-migration/results/baselines/rag/RAG/Deepseek-Coder-6.7b-Instruct.jsonl",
    "/data/hassan/java2rust-automated-migration/results/baselines/rag/RAG/Deepseek-Coder-33b-Instruct.jsonl",
    "/data/hassan/java2rust-automated-migration/results/baselines/rag/RAG/Qwen2.5-Coder-7B-Instruct.jsonl",
    "/data/hassan/java2rust-automated-migration/results/baselines/rag/RAG/Qwen2.5-Coder-14B-Instruct.jsonl",
    "/data/hassan/java2rust-automated-migration/results/baselines/rag/RAG/Qwen2.5-Coder-32B-Instruct.jsonl",
]

JRT_POST_FILES = [
    "/data/hassan/java2rust-automated-migration/results/JavaRustTrans/refined/ragcat-signals-gptsummary/CodeLlama-7b-Instruct-hf.jsonl",
    "/data/hassan/java2rust-automated-migration/results/JavaRustTrans/refined/ragcat-signals-gptsummary/CodeLlama-34b-Instruct-hf.jsonl",
    "/data/hassan/java2rust-automated-migration/results/JavaRustTrans/refined/ragcat-signals-gptsummary/Deepseek-Coder-6.7b-Instruct.jsonl",
    "/data/hassan/java2rust-automated-migration/results/JavaRustTrans/refined/ragcat-signals-gptsummary/Deepseek-Coder-33b-Instruct.jsonl",
    "/data/hassan/java2rust-automated-migration/results/JavaRustTrans/refined/ragcat-signals-gptsummary/Qwen2.5-Coder-7B-Instruct.jsonl",
    "/data/hassan/java2rust-automated-migration/results/JavaRustTrans/refined/ragcat-signals-gptsummary/Qwen2.5-Coder-14B-Instruct.jsonl",
    "/data/hassan/java2rust-automated-migration/results/JavaRustTrans/refined/ragcat-signals-gptsummary/Qwen2.5-Coder-32B-Instruct.jsonl",
]

JRT_PRE_FILES = [
    "/data/hassan/java2rust-automated-migration/results/JavaRustTrans/no-refinement/ragcat-signals-gptsummary/CodeLlama-7b-Instruct-hf.jsonl",
    "/data/hassan/java2rust-automated-migration/results/JavaRustTrans/no-refinement/ragcat-signals-gptsummary/CodeLlama-34b-Instruct-hf.jsonl",
    "/data/hassan/java2rust-automated-migration/results/JavaRustTrans/no-refinement/ragcat-signals-gptsummary/Deepseek-Coder-6.7b-Instruct.jsonl",
    "/data/hassan/java2rust-automated-migration/results/JavaRustTrans/no-refinement/ragcat-signals-gptsummary/Deepseek-Coder-33b-Instruct.jsonl",
    "/data/hassan/java2rust-automated-migration/results/JavaRustTrans/no-refinement/ragcat-signals-gptsummary/Qwen2.5-Coder-7B-Instruct.jsonl",
    "/data/hassan/java2rust-automated-migration/results/JavaRustTrans/no-refinement/ragcat-signals-gptsummary/Qwen2.5-Coder-14B-Instruct.jsonl",
    "/data/hassan/java2rust-automated-migration/results/JavaRustTrans/no-refinement/ragcat-signals-gptsummary/Qwen2.5-Coder-32B-Instruct.jsonl",
]

# -----------------------------
# CONFIG
# -----------------------------
TOP_K = 5
FAILURES_ONLY = True        # only compiled==0
DEDUP_PER_SAMPLE = True     # bucket counted once per sample


# Optional: override specific bucket IDs -> nicer names
# Add your real bucket strings here as you see them.
BUCKET_NAME_OVERRIDES = {
    "type_trait_generic": "Type/Trait/Generic",
    "borrow_checker": "Borrow/Ownership",
    "lifetime": "Lifetimes",
    "name_resolution": "Name Resolution",
    "mutability": "Mutability",
    "imports_modules": "Imports/Modules",
    "method_not_found": "Method Not Found",
    "type_mismatch": "Type Mismatch",
    # (If your buckets differ, overrides are optional—default formatting still works.)
}


def pretty_bucket_name(bucket_id: str) -> str:
    """Convert bucket id to a PPT-friendly label."""
    if bucket_id in BUCKET_NAME_OVERRIDES:
        return BUCKET_NAME_OVERRIDES[bucket_id]
    # default: snake_case -> Title Case
    s = bucket_id.replace("_", " ").strip()
    return " ".join(w[:1].upper() + w[1:] for w in s.split()) if s else bucket_id


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def aggregate_bucket_counts(files: List[str]) -> Tuple[Counter, int]:
    """
    Returns:
      counter: bucket_id -> count
      used: number of samples contributing (compile failures if FAILURES_ONLY)
    """
    c = Counter()
    used = 0
    for fp in files:
        if not Path(fp).exists():
            raise FileNotFoundError(f"Missing file: {fp}")
        for obj in read_jsonl(fp):
            compiled = int(obj.get("compiled", 0))
            if FAILURES_ONLY and compiled != 0:
                continue

            buckets = obj.get("compile_errors", []) or []
            if not isinstance(buckets, list) or len(buckets) == 0:
                continue

            if DEDUP_PER_SAMPLE:
                buckets = list(dict.fromkeys(buckets))

            used += 1
            for b in buckets:
                if isinstance(b, str) and b.strip():
                    c[b.strip()] += 1
    return c, used


def topk_with_other(counts_by_method: Dict[str, Counter], top_k: int) -> Tuple[List[str], Dict[str, Counter]]:
    """Pick top-K buckets by total count across compared methods; merge rest into 'Other'."""
    total = Counter()
    for c in counts_by_method.values():
        total.update(c)
    top = [b for b, _ in total.most_common(top_k)]

    merged: Dict[str, Counter] = {}
    for method, c in counts_by_method.items():
        nc = Counter()
        other_sum = 0
        for b, v in c.items():
            if b in top:
                nc[b] += v
            else:
                other_sum += v
        nc["Other"] += other_sum
        # ensure all top buckets exist for consistent ordering
        for b in top:
            nc.setdefault(b, 0)
        merged[method] = nc

    # ordering: top buckets first (by total), then Other last
    bucket_order = top + ["Other"]
    return bucket_order, merged


def build_outputs(tag: str, methods: Dict[str, List[str]]):
    # 1) aggregate raw counts
    counts_by_method: Dict[str, Counter] = {}
    used_by_method: Dict[str, int] = {}
    for name, flist in methods.items():
        c, used = aggregate_bucket_counts(flist)
        counts_by_method[name] = c
        used_by_method[name] = used

    # 2) top-k selection within this comparison
    bucket_ids, merged_counts = topk_with_other(counts_by_method, TOP_K)

    # 3) build wide counts
    wide_counts = pd.DataFrame(
        {m: [merged_counts[m].get(b, 0) for b in bucket_ids] for m in methods.keys()},
        index=[pretty_bucket_name(b) if b != "Other" else "Other" for b in bucket_ids]
    )
    wide_counts.index.name = "Category"

    # 4) build wide percent (100% stacked)
    wide_percent = wide_counts.copy()
    for col in wide_percent.columns:
        s = wide_percent[col].sum()
        wide_percent[col] = (wide_percent[col] / s * 100.0) if s > 0 else 0.0

    # 5) build long format (optional)
    long_rows = []
    for cat in wide_counts.index.tolist():
        for m in wide_counts.columns.tolist():
            long_rows.append({
                "Comparison": tag,
                "Method": m,
                "Category": cat,
                "Count": int(wide_counts.loc[cat, m]),
                "Percent": float(wide_percent.loc[cat, m]),
                "SamplesUsed": int(used_by_method[m]),
            })
    long_df = pd.DataFrame(long_rows)

    # 6) save CSVs
    wide_counts.to_csv(f"{tag}_wide_counts.csv")
    wide_percent.to_csv(f"{tag}_wide_percent.csv", float_format="%.2f")
    long_df.to_csv(f"{tag}_long.csv", index=False, float_format="%.2f")

    print(f"\n[{tag}] wrote:")
    print(f" - {tag}_wide_counts.csv")
    print(f" - {tag}_wide_percent.csv   (use this for 100% stacked bar)")
    print(f" - {tag}_long.csv")
    print("Top categories used (with 'Other'):", wide_counts.index.tolist())
    print("SamplesUsed per method (compile failures if FAILURES_ONLY):", used_by_method)


def main():
    # Visualization 1: RAG vs JavaRustTrans (1x Refinement)
    build_outputs(
        tag="viz1_rag_vs_jrt",
        methods={
            "RAG": RAG_FILES,
            "JavaRustTrans (1x Refinement)": JRT_POST_FILES,
        },
    )

    # Visualization 2: JavaRustTrans w/o Refinement vs with Refinement
    build_outputs(
        tag="viz2_jrt_pre_vs_post",
        methods={
            "JavaRustTrans (w/o Refinement)": JRT_PRE_FILES,
            "JavaRustTrans (1x Refinement)": JRT_POST_FILES,
        },
    )

    print("\nPowerPoint tip:")
    print(" - Insert -> Chart -> 100% Stacked Column")
    print(" - Paste from *_wide_percent.csv (rows=Category, cols=Methods)")
    print("\nCounts computed on:")
    print(f" - FAILURES_ONLY={FAILURES_ONLY} (compiled==0 only)")
    print(f" - DEDUP_PER_SAMPLE={DEDUP_PER_SAMPLE} (bucket counted once per sample)")


if __name__ == "__main__":
    main()
