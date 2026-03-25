#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Java -> Rust Translation Pipeline with BM25 RAG - Optimized for 4× A100 (vLLM 0.11.0)
"""

import os
os.environ["HF_HOME"] = "/data/hassan/huggingface"
os.environ["HF_DATASETS_CACHE"] = "/data/hassan/huggingface/datasets"
os.environ["TRANSFORMERS_CACHE"] = "/data/hassan/huggingface/models"
os.environ["HF_MODULES_CACHE"] = "/data/hassan/huggingface/modules"
os.makedirs(os.environ["HF_MODULES_CACHE"], exist_ok=True)

import sys
import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import jsonlines
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Path for BM25 retriever
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)
from retrieve.bm25.bm25 import setup_bm25_retriever


class BaseTranslator:
    def __init__(
        self,
        model_name: str,
        method: str = "RAG",
        hf_model_path: str = None,
        api_url: str = None,
        api_key: str = None,
        batch_size: int = 8,
        max_tokens: int = 1024,
        rag_k: int = 2,
        icl_k: int = 2,
        load_vllm = True
    ):
        
        self.model_name = model_name
        self.method = method
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.rag_k = rag_k
        self.icl_k = icl_k
        self.load_vllm = load_vllm

        valid_methods = ["Instruction", "ICL", "RAG", "COT", "Vert"]
        if method not in valid_methods:
            raise ValueError(f"Method must be one of: {valid_methods}")

        # Backend selection
        if api_url and api_key:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key, base_url=api_url)
            self.backend = "openai"

        else:
            self.backend = "vllm"
            if not hf_model_path:
                raise ValueError("HF model path required for local inference")

            LARGE_MODELS = [
                "Qwen2.5-Coder-32B-Instruct",
                "Deepseek-Coder-33b-Instruct",
                "CodeLlama-34b-Instruct-hf",
            ]
            if model_name in LARGE_MODELS:
                safe_max_model_len = 12000
            else:
                safe_max_model_len = 8192
                
            if load_vllm:
                self.llm = LLM(
                    model=hf_model_path,
                    trust_remote_code=True,
                    tensor_parallel_size=1,
                    dtype="float16",
                    gpu_memory_utilization=0.90,
                    max_model_len=8112,
                    max_num_seqs=32,
                    disable_log_stats=True,
                )

                self.tokenizer = AutoTokenizer.from_pretrained(hf_model_path)

        # ============================================================
        # RAG corpus path (used for:
        #  - BM25 retrieval when method == "RAG"
        #  - RANDOM sampling when method in ["ICL","COT"]
        # ============================================================
        self.rag_corpus_path = "/data/hassan/java2rust-automated-migration/data/retrieval/processed/ragcorpus_512.jsonl"

        # RAG (BM25 retriever only for method == "RAG")
        self.bm25_retriever = None
        if method == "RAG":
            self.bm25_retriever = setup_bm25_retriever(
                sample_path=self.rag_corpus_path,
                top_k=self.rag_k,
            )
            print(f" BM25 retriever initialized with top_k={self.rag_k}")

        # ============================================================
        # NEW: Random pool from ragcorpus_512 for ICL/COT (no cache)
        # ============================================================
        self.rag_random_pool = []
        if method in ["ICL", "COT"]:
            self.rag_random_pool = self._load_ragcorpus_pool()
            print(f" Loaded {len(self.rag_random_pool)} random examples from ragcorpus_512 for {method}")

        # system prompts
        self.system_prompt = (
            "You are an expert in both Java and Rust. Your task is to translate Java code into Rust."
        )
        self.cot_system_prompt = (
            "You are an expert in both Java and Rust. Your task is to translate Java code into Rust. "
            "First think briefly about tricky parts if any, then output the Rust code in a code block (```rust```), "
            "followed by a short 1-2 sentence rationale."
        )

        self._setup_prompt_templates()

    # ---------------------------
    def _setup_prompt_templates(self):
        self.prompt_templates = {
            "Instruction": {
                "system": self.system_prompt,
                "user": (
                    "Here is Java code. Translate it to Rust. Return ONLY the translated Safe Rust code (no extra text).\n\n"
                    "Preserve behavior exactly: replicate the same algorithm, control flow, edge cases, and I/O format—do not simplify, optimize, or refactor.\n\n"
                    "### BEGIN JAVA_TO_TRANSLATE\n{java_code}\n### END JAVA_TO_TRANSLATE"
                ),
            },
            "ICL": {
                "system": self.system_prompt,
                "user": (
                    "I will provide Java code and some examples.\nYour job is to translate ONLY the Java code into Safe Rust.\n\n"
                    "Preserve behavior exactly: replicate the same algorithm, control flow, edge cases, and I/O format—do not simplify, optimize, or refactor.\n\n"
                    "### Java code to translate\n"
                    "### BEGIN JAVA_TO_TRANSLATE\n{java_code}\n### END JAVA_TO_TRANSLATE\n\n"
                    "### Translation examples (DO NOT TRANSLATE)\n"
                    "### BEGIN EXAMPLES\n{few_shot}\n### END EXAMPLES\n\n"
                    "### Reminder\n"
                    "Output ONLY the translated Safe Rust code no explanations, comments, or extra text."
                ),
            },
            "RAG": {
                "system": self.system_prompt,
                "user": (
                    "I will provide Java code and some examples.\nYour job is to translate ONLY the Java code into Safe Rust.\n\n"
                    "Preserve behavior exactly: replicate the same algorithm, control flow, edge cases, and I/O format—do not simplify, optimize, or refactor.\n\n"
                    "### Java code to translate\n"
                    "### BEGIN JAVA_TO_TRANSLATE\n{java_code}\n### END JAVA_TO_TRANSLATE\n\n"
                    "### Translation examples (DO NOT TRANSLATE)\n"
                    "### BEGIN EXAMPLES\n{few_shot}\n### END EXAMPLES\n\n"
                    "### Reminder\n"
                    "Output ONLY the translated Safe Rust code no explanations, comments, or extra text."
                ),
            },
            "COT": {
                "system": self.cot_system_prompt,
                "user": (
                    "I will provide Java code and some examples.\nTranslate ONLY the Java code into Safe Rust.\n\n"
                    "Preserve behavior exactly: replicate the same algorithm, control flow, edge cases, and I/O format—do not simplify, optimize, or refactor.\n\n"
                    "### Java code to translate\n"
                    "### BEGIN JAVA_TO_TRANSLATE\n{java_code}\n### END JAVA_TO_TRANSLATE\n\n"
                    "### Translation examples (DO NOT TRANSLATE)\n"
                    "### BEGIN EXAMPLES\n{few_shot}\n### END EXAMPLES"
                ),
            },

        }


    # ---------------------------
    def load_dataset(self, dataset_path: str):
        data = []

        with jsonlines.open(dataset_path) as reader:
            for item in reader:
                java_block = item.get("Java", {})
                java_code = java_block.get("source_code")

                if not java_code or not java_code.strip():
                    continue

                rust_block = item.get("Rust", {})
                rust_code = rust_block.get("source_code", "")
                tests = item.get("hidden_unit_tests", [])

                data.append(
                    {
                        "java": java_code.strip(),
                        "rust": rust_code.strip(),
                        "tests": tests,
                    }
                )

        print(f" Loaded {len(data)} examples from {dataset_path}")
        return data

    # ---------------------------
    def _load_ragcorpus_pool(self):
        pool = []
        try:
            with jsonlines.open(self.rag_corpus_path) as reader:
                for item in reader:
                    # MINIMAL FIX: ragcorpus_512 uses "Java"/"Rust" keys (not java_code/rust_code)
                    if "Java" in item and "Rust" in item:
                        pool.append(
                            {
                                "java": item["Java"],
                                "rust": item["Rust"],
                            }
                        )
        except Exception:
            pass
        return pool

    # ---------------------------
    def _format_few_shot(self, examples):
        return "\n\n".join(
            [
                f"Java:\n```\n{ex['java']}\n```\nTranslated Rust:\n```\n{ex['rust']}\n```"
                for ex in examples
            ]
        )

    # ---------------------------
    def _retrieve_rag_examples(self, java_code):
        if not self.bm25_retriever:
            return []

        ranked = self.bm25_retriever(java_code) or []
        out = []
        for item in ranked:
            # MINIMAL FIX: retriever may return either raw (Java/Rust)
            # or normalized (java_code/rust_code) keys.
            j = item.get("java_code") or ""
            r = item.get("rust_code") or ""
            if j is None or r is None:
                continue
            out.append({"java": j, "rust": r})
        return out


    # ============================================================
    # NEW: Random ICL / COT examples from ragcorpus_512 per-query (no cache)
    # ============================================================
    def _sample_random_rag_examples(self, k):
        if not self.rag_random_pool:
            return []
        return random.sample(self.rag_random_pool, min(k, len(self.rag_random_pool)))

    # ---------------------------
    def _sample_icl_examples(self, java_code, k):
        # Now random from ragcorpus_512 every query (no cache)
        return self._sample_random_rag_examples(k)

    # ---------------------------
    def _sample_cot_examples(self, java_code, k):
        # Now random from ragcorpus_512 every query (no cache)
        return self._sample_random_rag_examples(k)

    # ---------------------------
    def construct_prompt(self, java_code):
        template = self.prompt_templates[self.method]

        if self.method == "RAG":
            examples = self._retrieve_rag_examples(java_code)

        elif self.method == "ICL":
            examples = self._sample_icl_examples(java_code, self.icl_k)

        elif self.method == "COT":
            examples = self._sample_cot_examples(java_code, self.icl_k)

        else:
            examples = []

        if examples:
            few_shot_text = self._format_few_shot(examples)
            user_prompt = template["user"].format(
                few_shot=few_shot_text, java_code=java_code
            )
        else:
            # MINIMAL FIX: avoid KeyError when template contains {few_shot}
            user_prompt = template["user"].format(few_shot="None", java_code=java_code)

        return {"system": template["system"], "user": user_prompt}

    # ---------------------------
    def generate_batch_vllm(self, prompts):
        formatted_prompts = []
        for prompt in prompts:
            try:
                formatted = self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": prompt["system"]},
                        {"role": "user", "content": prompt["user"]},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except:
                formatted = (
                    f"System: {prompt['system']}\n\nUser: {prompt['user']}"
                )
            formatted_prompts.append(formatted)

        # StarCoder2 fairness patch
        if "starcoder" in self.model_name.lower():
            SAFE_LIMIT = 8192
            truncated = []
            for p in formatted_prompts:
                ids = self.tokenizer(p, return_tensors="pt").input_ids[0]
                if len(ids) > SAFE_LIMIT:
                    ids = ids[-SAFE_LIMIT:]
                    p = self.tokenizer.decode(
                        ids, skip_special_tokens=True
                    )
                truncated.append(p)
            formatted_prompts = truncated

        effective_max_tokens = min(self.max_tokens, 2048)

        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=effective_max_tokens,
            skip_special_tokens=True,
        )

        try:
            outputs = self.llm.generate(formatted_prompts, sampling_params)
            return [o.outputs[0].text.strip() for o in outputs]
        except torch.cuda.OutOfMemoryError:
            return ["ERROR: CUDA OOM"] * len(prompts)
        except Exception as e:
            return [f"ERROR: {e}"] * len(prompts)

    # ---------------------------
    def run_inference(self, dataset_path, output_path):
        print(f" Starting {self.method} translation with {self.model_name}")

        data = self.load_dataset(dataset_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        total_batches = (len(data) + self.batch_size - 1) // self.batch_size
        progress = tqdm(total=total_batches)

        print(
            f" Resuming and appending to existing output file: {output_path}"
        )

        for i in range(0, len(data), self.batch_size):
            batch = data[i: i + self.batch_size]
            prompts = [
                self.construct_prompt(item["java"]) for item in batch
            ]

            translations = self.generate_batch_vllm(prompts)

            for item, pred in zip(batch, translations):
                record = {
                    "java": item["java"],
                    "trans_rust": pred,
                    "gt_rust": item["rust"],
                    "hidden_unit_tests": item["tests"],
                    "method": self.method,
                    "model": self.model_name,
                }

                with jsonlines.open(output_path, "a") as w:
                    w.write(record)

            progress.update(1)

        progress.close()
        print(f" Inference finished. Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="/data/hassan/java2rust-automated-migration/data/processed/final_test.jsonl",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/data/hassan/java2rust-automated-migration/experiments/baseline/no-refinement/ICL/deepseek-coder-1.3b-instruct.jsonl",
    )
    parser.add_argument("--model_name", type=str, default="deepseek-coder-1.3b-instruct")
    parser.add_argument("--method", type=str, default="ICL")
    parser.add_argument(
        "--hf_model_path",
        type=str,
        default="/data/hassan/models/deepseek-coder-1.3b-instruct",
    )
    parser.add_argument("--api_url", type=str)
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--rag_k", type=int, default=2)
    parser.add_argument("--icl_k", type=int, default=2)
    

    args = parser.parse_args()

    translator = BaseTranslator(
        model_name=args.model_name,
        method=args.method,
        hf_model_path=args.hf_model_path,
        api_url=args.api_url,
        api_key=args.api_key,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        rag_k=args.rag_k,
        icl_k=args.icl_k,
    )

    translator.run_inference(args.dataset, args.output)


if __name__ == "__main__":
    main()