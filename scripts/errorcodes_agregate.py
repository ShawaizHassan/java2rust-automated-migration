#!/usr/bin/env python3
"""
Compile error-code analysis for Java-to-Rust translation.

This single script generates TWO top-10 LaTeX tables ONLY (no CSV):

(1) Best baseline (RAG) vs JavaRustTrans (post-refinement)
    - ranks error codes by RAG frequency
    - output: top10_error_codes_rag_vs_jrt.tex

(2) JavaRustTrans pre- vs post-refinement
    - denominator fixed to programs that FAILED compilation pre-refinement
    - ranks error codes by PRE frequency
    - output: top10_error_codes_jrt_pre_vs_post.tex

Schema expected per line (JSONL):
{"id": ..., "compiled": 0/1, "compile_error_codes": ["E0308", ...], ...}

Counting:
- Program-level: each unique code in a program contributes +1.
"""

import os
import json
from collections import Counter
import pandas as pd


# -----------------------------
# 1) File lists
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
# 2) Helpers
# -----------------------------
def iter_jsonl(fp: str):
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def aggregate_error_codes(files):
    """
    Count frequency of each compile error code across files.
    Program-level: unique codes per program contribute +1.
    """
    code_ctr = Counter()
    total_programs = 0
    total_compile_fail = 0

    for fp in files:
        for rec in iter_jsonl(fp):
            total_programs += 1
            compiled = int(rec.get("compiled", 0))
            if compiled == 0:
                total_compile_fail += 1

            codes = rec.get("compile_error_codes", []) or []
            for c in set(codes):
                code_ctr[c] += 1

    return code_ctr, total_programs, total_compile_fail


def key_for_model(fp: str) -> str:
    return os.path.basename(fp)


def load_by_id(fp: str):
    d = {}
    for rec in iter_jsonl(fp):
        rid = rec.get("id")
        if rid is None:
            continue
        d[int(rid)] = rec
    return d


def write_tex_table(df: pd.DataFrame, out_tex: str):
    latex = df.to_latex(index=False, escape=True, column_format="r l r r r")
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"Saved: {out_tex}")


# -----------------------------
# 3) (1) RAG vs JavaRustTrans (post)
# -----------------------------
rag_ctr, rag_N, rag_fail = aggregate_error_codes(RAG_FILES)
jrt_ctr, jrt_N, jrt_fail = aggregate_error_codes(JRT_POST_FILES)

top_codes_rag = [c for c, _ in rag_ctr.most_common(10)]

rows = []
for rank, code in enumerate(top_codes_rag, start=1):
    rag_cnt = rag_ctr.get(code, 0)
    jrt_cnt = jrt_ctr.get(code, 0)
    rows.append(
        {
            "Rank": rank,
            "Error code": code,
            "RAG Cnt": rag_cnt,
            "JRT Cnt": jrt_cnt,
            "Δ Cnt": jrt_cnt - rag_cnt,  # negative = fewer errors in JRT
        }
    )

df_rag_vs_jrt = pd.DataFrame(rows)
write_tex_table(df_rag_vs_jrt, "top10_error_codes_rag_vs_jrt.tex")


# -----------------------------
# 4) (2) JavaRustTrans pre vs post refinement
#     Denominator fixed to PRE compile-failing programs
# -----------------------------
pre_map = {key_for_model(fp): load_by_id(fp) for fp in JRT_PRE_FILES}
post_map = {key_for_model(fp): load_by_id(fp) for fp in JRT_POST_FILES}

missing_post = sorted(set(pre_map.keys()) - set(post_map.keys()))
if missing_post:
    raise RuntimeError(f"Missing post-refinement files for: {missing_post}")

pre_ctr = Counter()
post_ctr = Counter()

for model_name, pre_by_id in pre_map.items():
    post_by_id = post_map[model_name]

    failing_ids = [i for i, r in pre_by_id.items() if int(r.get("compiled", 0)) == 0]

    for i in failing_ids:
        pre_rec = pre_by_id.get(i, {})
        post_rec = post_by_id.get(i, {})

        pre_codes = set(pre_rec.get("compile_error_codes", []) or [])
        post_codes = set(post_rec.get("compile_error_codes", []) or [])

        for c in pre_codes:
            pre_ctr[c] += 1
        for c in post_codes:
            post_ctr[c] += 1

top_codes_pre = [c for c, _ in pre_ctr.most_common(10)]

rows = []
for rank, code in enumerate(top_codes_pre, start=1):
    pre_cnt = pre_ctr.get(code, 0)
    post_cnt = post_ctr.get(code, 0)
    rows.append(
        {
            "Rank": rank,
            "Error code": code,
            "Pre Cnt": pre_cnt,
            "Post Cnt": post_cnt,
            "Δ Cnt": post_cnt - pre_cnt,
        }
    )

df_pre_vs_post = pd.DataFrame(rows)
write_tex_table(df_pre_vs_post, "top10_error_codes_jrt_pre_vs_post.tex")
