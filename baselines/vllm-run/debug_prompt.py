#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prompt debugger for BaseTranslator:
- imports BaseTranslator from your baseline file
- uses its own construct_prompt (+ optional tokenizer.apply_chat_template)
- SAVES each prompt to a separate .txt file
"""

import argparse
import os
import jsonlines

# change this import to your actual baseline filename (without .py)
from baseline import BaseTranslator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        type=str,
        default="/data/hassan/java2rust-automated-migration/data/processed/final_test.jsonl",
        help="final_test.jsonl path",
    )
    ap.add_argument(
        "--method",
        type=str,
        default="RAG",
        choices=["Instruction", "ICL", "RAG", "COT", "Vert"],
    )
    ap.add_argument("--model_name", type=str, default="CodeLlama-34b-Instruct-hf")
    ap.add_argument("--hf_model_path", type=str, default="/data/hassan/models/CodeLlama-34b-Instruct-hf")
    ap.add_argument("--rag_k", type=int, default=2)
    ap.add_argument("--icl_k", type=int, default=2)
    ap.add_argument("--n", type=int, default=522, help="how many prompts to dump")
    ap.add_argument("--idx", type=int, default=None, help="dump only this dataset index")
    ap.add_argument("--chat", action="store_true", help="also dump chat-formatted prompt (requires tokenizer)")
    ap.add_argument(
        "--out_dir",
        type=str,
        default="./debug_prompts",
        help="directory to store prompts",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    t = BaseTranslator(
        model_name=args.model_name,
        method=args.method,
        hf_model_path=args.hf_model_path,
        rag_k=args.rag_k,
        icl_k=args.icl_k,
        load_vllm=args.chat,  # only load tokenizer when you want --chat
    )

    # load java snippets
    items = []
    with jsonlines.open(args.dataset) as r:
        for it in r:
            java = (it.get("Java") or {}).get("source_code")
            if java and java.strip():
                items.append(java.strip())

    indices = [args.idx] if args.idx is not None else list(range(min(args.n, len(items))))

    for i in indices:
        prompt = t.construct_prompt(items[i])

        # one file per snippet
        base = os.path.join(args.out_dir, f"{args.method}_idx{i}")

        # raw system+user
        with open(base + ".txt", "w", encoding="utf-8") as f:
            f.write(f"INDEX: {i} | METHOD: {args.method}\n")
            f.write("=" * 80 + "\n")
            f.write("[SYSTEM]\n")
            f.write(prompt["system"] + "\n")
            f.write("-" * 80 + "\n")
            f.write("[USER]\n")
            f.write(prompt["user"] + "\n")

        # optional chat-formatted
        if args.chat:
            chat = t.tokenizer.apply_chat_template(
                [{"role": "system", "content": prompt["system"]},
                 {"role": "user", "content": prompt["user"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            with open(base + "_chat.txt", "w", encoding="utf-8") as f:
                f.write(chat)

        print(f"Saved: {base}.txt" + (f" and {base}_chat.txt" if args.chat else ""))

    print("Done.")


if __name__ == "__main__":
    main()
