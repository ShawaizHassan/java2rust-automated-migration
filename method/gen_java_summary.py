#!/usr/bin/env python3
"""
Agent 1 — Java Summary Generator
----------------------------------------------
Input  : Java source code
Output : Short semantic summary (input → processing → output)

Stored at:
middle_infer_result/summary/<MODEL_NAME>/infer_result.jsonl
"""

import os
import torch
import json
import jsonlines
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI
import argparse


# ============================================================
# Summary Prompt Templates
# ============================================================


SUMMARY_PROMPT = (
                "Your task is to generate a **summary** based on **Java code**.\n\n"
                "Write EXACTLY 3 lines in this format:\n"
                "Input: <what is read from stdin/files; data types/structure>\n"
                "Process: <main steps/algorithm; key conditions/loops/data structures>\n"
                "Output: <what is printed/returned; format>\n\n"
                "### Java Code\n"
                "```\n"
                "{java_code}\n"
                "```\n\n"
                "### Reminder\n"
                "**- Output ONLY those 3 lines. No extra text.**\n"
            )



# ============================================================
# Java Summary Class
# ============================================================

class InferJavaSummary:
    
    def __init__(self, model_name, api_url=None, api_key=None, batch_size=50):
        self.model_name = model_name
        self.batch_size = batch_size

        # If an API key is provided (or set in env), use OpenAI; otherwise use offline vLLM
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.api_url = api_url or os.getenv("OPENAI_BASE_URL")

        if self.api_key:
            self.use_openai = True
        else:
            self.use_openai = False
            self.tensor_parallel_size = 4
            self.load_vllm_model()

    # ------------------------------
    # Load vLLM Model
    # ------------------------------
    def load_vllm_model(self):
        model_path = "/data/jfeng/models/" + self.model_name

        self.model = LLM(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=8192,                 
            gpu_memory_utilization=0.90,        
            max_num_seqs=4,                     
            swap_space=32,
            enforce_eager=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    # ------------------------------
    # Load Java dataset
    # ------------------------------
    def load_data(self, dataset_path):
        data = []
        with jsonlines.open(dataset_path, "r") as reader:
            for item in reader:
                java_code = item["Java"]["source_code"]
                src_uid = item["Java"]["src_uid"] if "src_uid" in item["Java"] else item.get("src_uid")
                data.append({
                    "java_code": java_code,
                    "item": item,
                    "src_uid": src_uid,
                })
        self.data = data

    # ------------------------------
    # Construct prompt
    # ------------------------------
    def construct_prompt(self, java_code):
        return SUMMARY_PROMPT.format(java_code=java_code)

    # ------------------------------
    # Build prompts (batch input)
    # ------------------------------
    def build_prompts(self):
        prompts = []
        for d in self.data:
            user_prompt = self.construct_prompt(d["java_code"])

            if self.use_openai:
                # For OpenAI, send the raw user prompt
                prompt = user_prompt
            else:
                # For offline vLLM, apply the model's chat template
                prompt = self.tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": user_prompt}
                    ],
                    add_generation_prompt=True,
                    tokenize=False,
                )

            prompts.append(prompt)
        return prompts

    # ---------------------------------------------------------
    # Save output JSONL
    # ---------------------------------------------------------
    def save_result(self, results):
        out_dir = f"/data/hassan/java2rust-automated-migration/outputs/middle_infer_result/summary/{self.model_name}"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "infer_result.jsonl")

        with jsonlines.open(out_path, "w") as writer:
            for r in results:
                writer.write(r)

        print(f" Summary generation complete. Saved to: {out_path}")

    # ============================================================
    # Inference: vLLM
    # ============================================================
    def infer_vllm(self, prompts):
        out_dir = f"/data/hassan/java2rust-automated-migration/outputs/middle_infer_result/summary/{self.model_name}"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "infer_result.jsonl")

        batches = (len(prompts) + self.batch_size - 1) // self.batch_size

        # Open JSONL file in append mode (streaming)
        with jsonlines.open(out_path, "a") as writer:

            for start in tqdm(range(0, len(prompts), self.batch_size),
                            total=batches, desc="Running Summary vLLM"):

                batch_prompts = prompts[start:start+self.batch_size]

                sampling = SamplingParams(
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=512,
                )

                outputs = self.model.generate(batch_prompts, sampling)

                for i, out in enumerate(outputs):
                    summary = out.outputs[0].text.strip()
                    data_idx = start + i

                    record = {
                        "idx": data_idx,
                        "src_uid": self.data[data_idx]["src_uid"],
                        "summary": summary,
                        "java_source": self.data[data_idx]["java_code"],
                    }

                    writer.write(record)     


    # ============================================================
    # Inference: OpenAI Compatible
    # ============================================================
    def infer_openai(self, prompts):
        if self.api_url:
            client = OpenAI(api_key=self.api_key, base_url=self.api_url)
        else:
            client = OpenAI(api_key=self.api_key)

        out_dir = f"/data/hassan/java2rust-automated-migration/outputs/middle_infer_result/summary/{self.model_name}"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "infer_result.jsonl")

        with jsonlines.open(out_path, "a") as writer:
            def call_openai(idx, prompt):
                try:
                    resp = client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0,
                        max_tokens=512,
                    )
                    content = resp.choices[0].message.content.strip()

                    return {
                        "idx": idx,
                        "src_uid": self.data[idx]["src_uid"],
                        "summary": content,
                        "java_source": self.data[idx]["java_code"],
                    }
                except Exception as e:
                    return {"idx": idx, "error": str(e)}

            with ThreadPoolExecutor(max_workers=self.batch_size) as exe:
                futures = [exe.submit(call_openai, i, p) for i, p in enumerate(prompts)]
                for f in tqdm(as_completed(futures), total=len(futures), desc="Running Summary API"):
                    record = f.result()
                    writer.write(record)   

    # ============================================================
    # Main Pipeline
    # ============================================================
    def run(self, dataset_path):
        self.load_data(dataset_path)
        prompts = self.build_prompts()

        if self.use_openai:
            self.infer_openai(prompts)
        else:
            self.infer_vllm(prompts)


# ============================================================
# MAIN ENTRY
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="Qwen2.5-Coder-32B-Instruct")
    parser.add_argument("--input", type=str, default="/data/hassan/java2rust-automated-migration/data/processed/final_test.jsonl", help="Path to Java JSONL dataset")
    parser.add_argument("--api_url", default=None)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--batch_size", type=int, default=50)

    args = parser.parse_args()

    model = InferJavaSummary(
        model_name=args.model_name,
        api_url=args.api_url,
        api_key=args.api_key,
        batch_size=args.batch_size
    )

    model.run(args.input)
