#!/usr/bin/env python3
"""
Refinement script (1-step repair) for Rust outputs.

Input JSONL records:
  - Compile stage (from compile error logger):
      {"java", "trans_rust", "hidden_unit_tests", "refine_unit_tests", "compile_ok", "error"}
  - Runtime stage (from runtime evaluator):
      {"java", "trans_rust", "hidden_unit_tests", "refine_unit_tests", "ca_ok", "error"}

This script:
- By default (compile refinement): repairs ONLY records with compile_ok == 0 (and non-empty error).
- If --unittest_refinement is set: repairs ONLY records with ca_ok == 0 (and non-empty error).
- Replaces `trans_rust` with the repaired Rust code (string).
- Keeps the SAME output format/keys as input (no extra keys).
- Leaves status/error fields as-is (you will re-run your evaluator to recompute them).

- Final output file contains schema:
  {"java":, "trans_rust":, "hidden_unit_tests":, "refine_unit_tests":}
"""

import os
import re
import jsonlines
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ----------- Rust fence extraction / postprocess -----------
FENCE = re.compile(r"```(?:rust|rs|RUST|Rust)?\s*([\s\S]*?)```", re.MULTILINE)

def extract_raw_rust(text: str) -> str:
    if not text:
        return ""
    blocks = FENCE.findall(text)
    return blocks[-1].strip() if blocks else text.strip()

# ----------- Prompt templates -----------
SYSTEM_CONTEXT = (
    "You are an expert Rust engineer. Produce correct, idiomatic, and safe Rust. "
    "Do not use unsafe. Output only the full Rust code."
)

COMPILE_USER_CONTEXT = (
    "Rust you generated:\n{rust_code}\n\n"
    "Compiler error:\n{error}\n\n"
    "Fix ONLY what is needed to compile. "
    "Do NOT change the algorithm, goal, or I/O format. "
    "Do NOT replace with a different solution. Prefer small edits; keep structure/names when possible. "
    "Output ONLY the full corrected Rust code."
)


RUNTIME_USER_CONTEXT = (
    "Rust you generated:\n{rust_code}\n\n"
    "Unit test failure logs:\n{error}\n\n"
    "Fix ONLY what is needed to pass the unit tests. "
    "Do NOT change the algorithm, goal, or I/O format. "
    "Do NOT replace with a different solution. Prefer small edits; keep structure/names when possible. "
    "Output ONLY the full corrected Rust code."
)


def build_compile_prompt(rust_code: str, error: str) -> str:
    return COMPILE_USER_CONTEXT.format(rust_code=rust_code, error=error)

def build_runtime_prompt(rust_code: str, error: str) -> str:
    return RUNTIME_USER_CONTEXT.format(rust_code=rust_code, error=error)

# ----------- vLLM inference -----------
def infer_vllm(prompts, model_path, tensor_parallel_size, max_model_len, batch_size, max_tokens):
    import torch
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    if tensor_parallel_size is None:
        tensor_parallel_size = torch.cuda.device_count() if torch.cuda.is_available() else 1

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        swap_space=20,
    )
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    def to_chat_prompt(user_content: str) -> str:
        msgs = [
            {"role": "system", "content": SYSTEM_CONTEXT},
            {"role": "user", "content": user_content},
        ]
        try:
            return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        except Exception:
            return SYSTEM_CONTEXT + "\n\n" + user_content + "\n"

    chat_prompts = [to_chat_prompt(p) for p in prompts]

    sampling = SamplingParams(
        temperature=0,
        top_p=1.0,
        max_tokens=max_tokens,
    )

    outputs = []
    for i in range(0, len(chat_prompts), batch_size):
        batch = chat_prompts[i:i + batch_size]
        res = llm.generate(batch, sampling)
        for r in res:
            text = r.outputs[0].text.strip() if r.outputs else ""
            outputs.append(text)

    return outputs

# ----------- OpenAI-compatible API inference -----------
def infer_openai(prompts, model_name, api_url, api_key, max_tokens, workers):
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=api_url)

    def call_one(p):
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_CONTEXT},
                {"role": "user", "content": p},
            ],
            temperature=0,
            top_p=1,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()

    outs = [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(call_one, p): idx for idx, p in enumerate(prompts)}
        for fut in as_completed(futs):
            idx = futs[fut]
            outs[idx] = fut.result()
    return outs

# ----------- Main -----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/data/hassan/java2rust-automated-migration/experiments/javarusttrans/error-logs/first-pass/ragcat-signals-gptsummary/Deepseek-Coder-33b-Instruct.jsonl", help="Input JSONL")
    ap.add_argument("--out", default="/data/hassan/java2rust-automated-migration/experiments/lustre/refinement/second-pass/ragcat-references-gptsummary/Deepseek-Coder-33b-Instruct", help="Output JSONL path (filtered schema)")
    ap.add_argument("--unittest_refinement", "-u", action="store_true",
                    help="If set, repair items where ca_ok==0 using runtime prompt")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--max_model_len", type=int, default=8112)

    # vLLM
    ap.add_argument("--model_path", default="/data/jfeng/models/Deepseek-Coder-33b-Instruct", help="Local HF/vLLM model path (required if not using api)")
    ap.add_argument("--tensor_parallel_size", type=int, default=2)

    # OpenAI-compatible API
    ap.add_argument("--api_url", default=None)
    ap.add_argument("--api_key", default=None)
    ap.add_argument("--api_model", default=None, help="Model name for API (e.g., gpt-4o-mini), used only if api_url+api_key set")
    ap.add_argument("--api_workers", type=int, default=16)

    args = ap.parse_args()

    use_api = bool(args.api_url and args.api_key and args.api_model)
    if not use_api and not args.model_path:
        raise SystemExit("ERROR: Provide either --model_path (vLLM) OR (--api_url --api_key --api_model).")

    in_path = Path(args.input)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load
    data = []
    with jsonlines.open(in_path, "r") as r:
        for rec in r:
            data.append(rec)

    # Build prompts for failures
    fail_indices = []
    prompts = []

    for idx, rec in enumerate(data):
        err = (rec.get("error") or "").strip()
        if not err:
            continue

        old_code = extract_raw_rust(rec.get("trans_rust", ""))

        if args.unittest_refinement:
            if int(rec.get("ca_ok", 1)) == 0:
                prompts.append(build_runtime_prompt(old_code, err))
                fail_indices.append(idx)
        else:
            if int(rec.get("compile_ok", 1)) == 0:
                prompts.append(build_compile_prompt(old_code, err))
                fail_indices.append(idx)

    # mark flags (default 0)
    for rec in data:
        rec["compile_repaired"] = 0
        rec["ca_refine"] = 0

    # ---- STREAM WRITE FIX: only rows NOT needing repair are "ready" initially ----
    needs_repair = set(fail_indices)

    # Pre-build final rows (content source of truth)
    rows = []
    for rec in data:
        rows.append({
            "java": rec.get("java", ""),
            "trans_rust": rec.get("trans_rust", ""),
            "hidden_unit_tests": rec.get("hidden_unit_tests", []),
            "refine_unit_tests": rec.get("refine_unit_tests", []),
            "compile_repaired": int(rec.get("compile_repaired", 0)),
            "ca_refine": int(rec.get("ca_refine", 0)),
        })

    # out_rows is the "ready-to-write" buffer in original order
    out_rows = [None] * len(rows)
    for i in range(len(rows)):
        if i not in needs_repair:
            out_rows[i] = rows[i]  # immediately writable

    next_to_write = 0

    def flush_ready(writer):
        nonlocal next_to_write
        while next_to_write < len(out_rows) and out_rows[next_to_write] is not None:
            writer.write(out_rows[next_to_write])
            out_rows[next_to_write] = None
            next_to_write += 1

    with jsonlines.open(out_path, "w") as w:
        if not prompts:
            # nothing to repair; all rows are ready
            flush_ready(w)
            print(f"Saved {len(rows)}/{len(rows)} items (no repairs) -> {out_path}")
            return

        # Inference
        if use_api:
            from openai import OpenAI
            client = OpenAI(api_key=args.api_key, base_url=args.api_url)

            def call_one(p):
                resp = client.chat.completions.create(
                    model=args.api_model,
                    messages=[
                        {"role": "system", "content": SYSTEM_CONTEXT},
                        {"role": "user", "content": p},
                    ],
                    temperature=0,
                    top_p=1,
                    max_tokens=args.max_tokens,
                )
                return resp.choices[0].message.content.strip()

            with ThreadPoolExecutor(max_workers=args.api_workers) as ex:
                futs = {ex.submit(call_one, p): local_i for local_i, p in enumerate(prompts)}
                for fut in as_completed(futs):
                    local_i = futs[fut]
                    idx = fail_indices[local_i]

                    raw = fut.result()
                    repaired = extract_raw_rust(raw or "")
                    if repaired.strip():
                        rows[idx]["trans_rust"] = repaired
                        if args.unittest_refinement:
                            rows[idx]["ca_refine"] = 1
                        else:
                            rows[idx]["compile_repaired"] = 1

                    # mark this index ready (even if model output empty -> keep original)
                    out_rows[idx] = rows[idx]
                    flush_ready(w)

        else:
            from vllm import LLM, SamplingParams
            from transformers import AutoTokenizer

            llm = LLM(
                model=args.model_path,
                trust_remote_code=True,
                tensor_parallel_size=args.tensor_parallel_size,
                max_model_len=args.max_model_len,
                gpu_memory_utilization=0.88, 
                max_num_seqs=8,  
                swap_space=20,
                enforce_eager=True
            )
            tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

            def to_chat_prompt(user_content: str) -> str:
                msgs = [
                    {"role": "system", "content": SYSTEM_CONTEXT},
                    {"role": "user", "content": user_content},
                ]
                try:
                    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
                except Exception:
                    return SYSTEM_CONTEXT + "\n\n" + user_content + "\n"

            chat_prompts = [to_chat_prompt(p) for p in prompts]
            sampling = SamplingParams(temperature=0, top_p=1.0, max_tokens=args.max_tokens)

            for start in range(0, len(chat_prompts), args.batch_size):
                batch = chat_prompts[start:start + args.batch_size]
                res = llm.generate(batch, sampling)

                for j, r in enumerate(res):
                    local_i = start + j
                    idx = fail_indices[local_i]

                    text = r.outputs[0].text.strip() if (r.outputs and len(r.outputs) > 0) else ""
                    repaired = extract_raw_rust(text or "")
                    if repaired.strip():
                        rows[idx]["trans_rust"] = repaired
                        if args.unittest_refinement:
                            rows[idx]["ca_refine"] = 1
                        else:
                            rows[idx]["compile_repaired"] = 1

                    # mark ready (even if empty -> original stays)
                    out_rows[idx] = rows[idx]

                flush_ready(w)

        # safety flush (should be no-op if everything became ready)
        flush_ready(w)
        # If anything is still None here, it means those indices were in needs_repair but never got outputs.
        # (Should not happen unless inference crashed mid-run.)

    print(f"Repaired {len(fail_indices)}/{len(data)} items and saved -> {out_path}")

if __name__ == "__main__":
    main()
