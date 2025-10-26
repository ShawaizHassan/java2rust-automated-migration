#!/usr/bin/env python3
"""
Fast remote (server-mode) Java -> Rust translator.

Start your vLLM server once (example):
  CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --served-model-name qwen2.5-coder-7b-instruct \
    --gpu-memory-utilization 0.85 \
    --max-model-len 6144 \
    --max-num-seqs 128 \
    --dtype float16 \
    --api-key hassan2024 \
    --host 127.0.0.1 --port 8000

Run this client:
  python fast_remote_client.py \
    --base-url http://127.0.0.1:8000/v1 \
    --model qwen2.5-coder-7b-instruct \
    --api-key hassan2024 \
    --concurrency 64 --max-tokens 768 --temperature 0.0
"""

import argparse, asyncio, time, re, random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import jsonlines
import httpx
from tqdm import tqdm
from openai import AsyncOpenAI

# -------------------- Prompt & formatting helpers -------------------- #

SYSTEM_PROMPT = (
    "You are an expert in automated code migration from Java to Rust.\n"
    "- Output ONLY Rust code (no markdown fences).\n"
    "- Preserve behavior; prefer idiomatic Rust (ownership, Result/Option).\n"
    "- Avoid unsafe unless necessary; add minimal use statements as needed."
)

def build_messages(java_code: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content":
            "Translate this Java snippet to Rust. Output ONLY Rust code.\n\n"
            "```java\n" + java_code + "\n```"}
    ]

FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]+)?\s*\n(.*?)\n?```", re.DOTALL)

def strip_md_fences(text: str) -> str:
    if not text:
        return ""
    m = FENCE_RE.search(text)
    return (m.group(1).strip() if m else text.strip())

def as_java_object(java_obj_or_str: Any) -> Dict[str, Any]:
    if isinstance(java_obj_or_str, dict):
        if "bug_source_code" in java_obj_or_str:
            return java_obj_or_str
        out = {"meta": java_obj_or_str}
        for k in ("code", "source", "src"):
            if k in java_obj_or_str and isinstance(java_obj_or_str[k], str):
                out["bug_source_code"] = java_obj_or_str[k]
                break
        return out
    return {"bug_source_code": str(java_obj_or_str) if java_obj_or_str is not None else ""}

def as_code_object(code_str: Optional[str]) -> Dict[str, Optional[str]]:
    return {"code": code_str if code_str is not None else None}

# -------------------- Core async request worker -------------------- #

async def translate_one(
    client: AsyncOpenAI,
    model: str,
    temperature: float,
    record: Dict[str, Any],
    max_tokens: int,
    request_timeout: float,
    stop: Optional[List[str]],
    semaphore: asyncio.Semaphore,
    max_attempts: int = 4,
    seed: int = 0,
) -> Optional[Dict[str, Any]]:
    java_raw = record.get("Java") or record.get("java")
    java_code = java_raw.get("bug_source_code") if isinstance(java_raw, dict) else java_raw
    if not java_code or not str(java_code).strip():
        return None

    java_obj = as_java_object(java_raw if java_raw is not None else "")
    rust_raw = record.get("Rust")
    rust_gt_code = rust_raw.get("bug_source_code") if isinstance(rust_raw, dict) else (rust_raw if isinstance(rust_raw, str) else None)
    gt_rust_obj = as_code_object(rust_gt_code)
    hidden_tests = record.get("hidden_unit_tests")

    backoff = 0.75
    last_err = None

    for attempt in range(1, max_attempts + 1):
        try:
            async with semaphore:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=build_messages(java_code),
                    temperature=temperature,          # 0.0 is fastest & deterministic
                    max_tokens=max_tokens,            # keep tight for speed
                    timeout=request_timeout,
                    stop=stop,                        # helps cut tail
                    seed=seed,                        # determinism across retries
                )

            content = resp.choices[0].message.content or ""
            trans_rust_code = strip_md_fences(content)

            out: Dict[str, Any] = {
                "Java": java_obj,
                "Trans_Rust": as_code_object(trans_rust_code),
                "GT_Rust": gt_rust_obj,
            }
            if hidden_tests is not None:
                out["hidden_unit_tests"] = hidden_tests
            return out

        except Exception as e:
            last_err = e
            # exponential backoff with jitter
            await asyncio.sleep(backoff + random.random() * 0.3)
            backoff *= 2.0

    # Failed after retries: still return collapsible structure with error marker
    out: Dict[str, Any] = {
        "Java": java_obj,
        "Trans_Rust": as_code_object(f"ERROR: {last_err}"),
        "GT_Rust": gt_rust_obj,
    }
    if hidden_tests is not None:
        out["hidden_unit_tests"] = hidden_tests
    return out

# -------------------- Main async driver -------------------- #

async def preflight(client: AsyncOpenAI, expect_model: str):
    """Fail fast if the server is unreachable/unauthorized or model name mismatches."""
    models = await client.models.list()
    names = [m.id for m in models.data]
    if expect_model not in names:
        print(f"[warn] Served models: {names}. Your --model '{expect_model}' wasn't found. "
              f"Check --served-model-name on the server or update --model here.")

async def run(args):
    # HTTP/2 + connection pooling for lower overhead
    limits = httpx.Limits(
        max_connections=args.concurrency * 2,
        max_keepalive_connections=args.concurrency,
        keepalive_expiry=30.0,
    )
    http_client = httpx.AsyncClient(http2=True, limits=limits, timeout=None)
    client = AsyncOpenAI(
        api_key=args.api_key,                 # must match vLLM --api-key
        base_url=args.base_url,               # e.g., http://127.0.0.1:8000/v1
        http_client=http_client,
        max_retries=0,                        # we implement our own retry with jitter
    )

    # Preflight auth/model check (fail fast if 401 or wrong model name)
    try:
        await preflight(client, args.model)
    except Exception as e:
        print("❌ Cannot reach server or unauthorized. Check --base-url and --api-key.")
        raise

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    input_path = Path(args.input)
    output_path = Path(args.output) / f"translated_{args.model}_{ts}.jsonl"

    # Load up to --limit
    records: List[Dict[str, Any]] = []
    with jsonlines.open(input_path) as reader:
        for i, rec in enumerate(reader):
            if args.limit and i >= args.limit:
                break
            records.append(rec)

    total = len(records)
    if total == 0:
        print("No records to process.")
        await http_client.aclose()
        return

    sem = asyncio.Semaphore(args.concurrency)
    stop = ["```", "</s>"] if args.use_stop else None

    started = time.time()
    completed = 0

    with jsonlines.open(output_path, mode="w") as writer, tqdm(total=total, desc="Translating", unit="rec") as pbar:
        # kick off tasks lazily to avoid huge in-memory task lists on big corpora
        async def producer():
            for rec in records:
                yield asyncio.create_task(
                    translate_one(
                        client, args.model, args.temperature, rec,
                        args.max_tokens, args.timeout, stop, sem,
                        max_attempts=args.max_attempts, seed=args.seed,
                    )
                )

        tasks_iter = producer()
        pending: set = set()
        # prime the queue
        for _ in range(min(args.concurrency * 2, total)):
            try:
                pending.add(await tasks_iter.__anext__())
            except StopAsyncIteration:
                break

        # process as tasks complete; keep pipeline full
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                res = task.result()
                if res is not None:
                    writer.write(res)
                completed += 1
                pbar.update(1)

                try:
                    pending.add(await tasks_iter.__anext__())
                except StopAsyncIteration:
                    pass

    dur = time.time() - started
    rps = (completed / dur) if dur > 0 else 0.0
    print(f"✅ Done! Translated {completed} records → {output_path}")
    print(f"⏱  Time: {dur:.1f}s  |  Throughput: {rps:.2f} rec/s  |  Concurrency: {args.concurrency}")

    await http_client.aclose()

# -------------------- CLI -------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/data/hassan/java2rust-automated-migration/data/processed/test_2048.jsonl")
    ap.add_argument("--output", default="/data/hassan/java2rust-automated-migration/outputs")

    # Server (OpenAI-compatible) settings
    ap.add_argument("--model", default="Qwen2.5-Coder-7B-Instruct",
                    help="Must match --served-model-name used by vLLM server.")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--api-key", default="hassan2024")  # must match vLLM --api-key

    # Performance knobs
    ap.add_argument("--concurrency", type=int, default=64, help="Number of in-flight requests (drives server batching)")
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--max-attempts", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use-stop", action="store_true", help="Stop on ``` / </s> to cut tail if the model tends to add fences")

    # Dataset limit for dry runs
    ap.add_argument("--limit", type=int, default=0)  # 0 = all
    args = ap.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    asyncio.run(run(args))

if __name__ == "__main__":
    main()

















# #!/usr/bin/env python3
# """
# Java -> Rust translator (collapsible JSONL) with BUILT-IN vLLM batching
# and proper chat templating for instruct models.

# ENGINES
# - vllm  : load model inside this process (GPU), batched generation  [default]
# - openai: call an OpenAI-compatible server (e.g., vLLM HTTP API)
# - auto  : try vllm, fall back to openai

# JSONL per line:
# {
#   "Java": {"bug_source_code": "...", ...},
#   "Trans_Rust": {"code": "..."},
#   "GT_Rust": {"code": "..."},
#   "hidden_unit_tests": [...]
# }
# """

# import argparse, os, re, time, shutil, subprocess
# from datetime import datetime
# from pathlib import Path
# from typing import List, Dict, Any, Optional
# import jsonlines
# from tqdm import tqdm

# # Optional (HTTP mode)
# try:
#     from openai import OpenAI
# except Exception:
#     OpenAI = None

# # ============================== Utilities =============================== #

# def strip_md_fences(text: str) -> str:
#     """Extract first fenced block if present, else trimmed text."""
#     if not text:
#         return ""
#     m = re.search(r"```(?:[a-zA-Z0-9_+-]+)?\s*\n(.*?)\n?```", text, flags=re.DOTALL)
#     return m.group(1).strip() if m else text.strip()

# def as_java_object(java_obj_or_str):
#     """Normalize Java field to an object with at least 'bug_source_code'."""
#     if isinstance(java_obj_or_str, dict):
#         if "bug_source_code" in java_obj_or_str:
#             return java_obj_or_str
#         out = {"meta": java_obj_or_str}
#         for k in ("code", "source", "src"):
#             if k in java_obj_or_str and isinstance(java_obj_or_str[k], str):
#                 out["bug_source_code"] = java_obj_or_str[k]
#                 break
#         return out
#     return {"bug_source_code": str(java_obj_or_str) if java_obj_or_str is not None else ""}

# def as_code_object(code_str):
#     return {"code": code_str if code_str is not None else None}

# def load_records(path: Path, limit: int) -> List[Dict[str, Any]]:
#     recs = []
#     with jsonlines.open(path) as reader:
#         for i, rec in enumerate(reader):
#             if limit and i >= limit:
#                 break
#             recs.append(rec)
#     return recs

# # ============================== HTTP engine ============================= #

# def build_prompt_chat(java_code: str):
#     """OpenAI-compatible chat messages."""
#     return [
#         {"role": "system", "content":
#          "You are an expert software engineer specializing in automated code migration from Java to Rust.\n"
#          "- Output ONLY Rust code unless asked to explain.\n"
#          "- Preserve logic and input/output behavior.\n"
#          "- Prefer idiomatic Rust: ownership, borrowing, Result/Option, error handling.\n"
#          "- Replace Java collections with Rust equivalents (Vec, HashMap, etc.).\n"
#          "- Avoid unsafe unless strictly necessary.\n"
#          "- Include use statements and a minimal main/test if needed."},
#         {"role": "user", "content":
#          "Here is a Java code snippet. Translate it to Rust.\n"
#          "Return ONLY Rust code (no markdown fences).\n\n"
#          "```java\n" + java_code + "\n```"}
#     ]

# def translate_one_openai(client, model, temperature, record, max_tokens=2048, timeout=60):
#     java_raw = record.get("Java") or record.get("java")
#     java_code = java_raw.get("bug_source_code") if isinstance(java_raw, dict) else java_raw
#     if not java_code or not str(java_code).strip():
#         return None

#     java_obj = as_java_object(java_raw if java_raw is not None else "")
#     rust_raw = record.get("Rust")
#     rust_gt_code = rust_raw.get("bug_source_code") if isinstance(rust_raw, dict) else (rust_raw if isinstance(rust_raw, str) else None)
#     gt_rust_obj = as_code_object(rust_gt_code)
#     hidden_tests = record.get("hidden_unit_tests")

#     last_err = None
#     for attempt in range(3):
#         try:
#             resp = client.chat.completions.create(
#                 model=model,
#                 messages=build_prompt_chat(java_code),
#                 temperature=temperature,
#                 max_tokens=max_tokens,
#                 timeout=timeout,
#             )
#             trans_rust_code = strip_md_fences(resp.choices[0].message.content)
#             out = {"Java": java_obj, "Trans_Rust": as_code_object(trans_rust_code), "GT_Rust": gt_rust_obj}
#             if hidden_tests is not None:
#                 out["hidden_unit_tests"] = hidden_tests
#             return out
#         except Exception as e:
#             last_err = e
#             time.sleep(1.5 * (attempt + 1))

#     out = {"Java": java_obj, "Trans_Rust": as_code_object(f"ERROR: {last_err}"), "GT_Rust": gt_rust_obj}
#     if hidden_tests is not None:
#         out["hidden_unit_tests"] = hidden_tests
#     return out

# def run_openai(args):
#     if OpenAI is None:
#         raise RuntimeError("`openai` package not available. pip install openai")
#     client = OpenAI(api_key=args.api_key, base_url=args.base_url)

#     ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
#     out_path = Path(args.output) / f"translated_{args.model}_{ts}.jsonl"
#     records = load_records(Path(args.input), args.limit)
#     if not records:
#         print("No records to process."); return
#     total = len(records)

#     from concurrent.futures import ThreadPoolExecutor, as_completed
#     with jsonlines.open(out_path, "w") as writer, \
#          ThreadPoolExecutor(max_workers=args.workers) as ex, \
#          tqdm(total=total, desc="Translating (openai/http)", unit="rec") as pbar:

#         futures = [ex.submit(translate_one_openai, client, args.model, args.temperature, rec,
#                              args.max_tokens, args.timeout) for rec in records]
#         for fut in as_completed(futures):
#             res = fut.result()
#             if res is not None:
#                 writer.write(res)
#             pbar.update(1)

#     print(f"✅ Done! Translated {total} records → {out_path}")

# # ============================== vLLM engine ============================= #

# def _nvsmi() -> Optional[str]:
#     return shutil.which("nvidia-smi")

# def _free_total_ratio(device: str) -> Optional[float]:
#     nvsmi = _nvsmi()
#     if not nvsmi: return None
#     try:
#         out = subprocess.check_output(
#             [nvsmi, "--id="+device, "--query-gpu=memory.free,memory.total",
#              "--format=csv,noheader,nounits"], timeout=5
#         ).decode().strip()
#         free_mb, total_mb = [float(x.strip()) for x in out.split(",")]
#         return (free_mb / total_mb) if total_mb > 0 else None
#     except Exception:
#         return None

# def _force_device(device: Optional[str]):
#     if device is not None:
#         os.environ["CUDA_VISIBLE_DEVICES"] = device
#         print(f"[device] CUDA_VISIBLE_DEVICES={device}")

# def _safe_util(user_util: float, device: Optional[str]) -> float:
#     """Clamp util to <= free/total - 0.02 on selected device to avoid init error."""
#     dev = (device if device is not None else os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
#     r = _free_total_ratio(dev)
#     if r is None:  # no nvidia-smi; just return user value
#         return user_util
#     return max(0.50, min(user_util, max(0.0, r - 0.02)))

# def run_vllm(args):
#     from vllm import LLM, SamplingParams

#     _force_device(args.device)
#     util = _safe_util(args.gpu_memory_utilization, args.device)
#     if util < args.gpu_memory_utilization:
#         print(f"[vllm] gpu_memory_utilization clamped {args.gpu_memory_utilization:.2f} → {util:.2f}")

#     ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
#     out_path = Path(args.output) / f"translated_{args.model.replace('/','_')}_{ts}.jsonl"
#     raw_records = load_records(Path(args.input), args.limit)
#     if not raw_records:
#         print("No records to process."); return

#     # Init LLM (loads tokenizer too)
#     print(f"[vllm] init model={args.model}, tp={args.tensor_parallel_size}, dtype={args.dtype}, util={util:.2f}")
#     llm = LLM(
#         model=args.model,
#         tensor_parallel_size=args.tensor_parallel_size,
#         trust_remote_code=True,
#         dtype=args.dtype,
#         gpu_memory_utilization=util,
#         max_model_len=(args.max_model_len if args.max_model_len > 0 else None),
#         max_num_seqs=(args.max_num_seqs if args.max_num_seqs > 0 else None),
#         enforce_eager=args.enforce_eager,
#     )
#     tok = llm.get_tokenizer()

#     def to_chat_prompt(java_code: str) -> str:
#         """Use the model's chat template (crucial for instruct models)."""
#         messages = [
#             {
#                 "role": "system",
#                 "content":
#                     "You are an expert software engineer specializing in automated code migration from Java to Rust.\n"
#                     "- Output ONLY Rust code unless asked to explain.\n"
#                     "- Preserve logic and input/output behavior.\n"
#                     "- Prefer idiomatic Rust: ownership, borrowing, Result/Option, error handling.\n"
#                     "- Replace Java collections with Rust equivalents (Vec, HashMap, etc.).\n"
#                     "- Avoid unsafe unless strictly necessary.\n"
#                     "- Include use statements and a minimal main/test if needed."
#             },
#             {
#                 "role": "user",
#                 "content":
#                     "Here is a Java code snippet. Translate it to Rust.\n"
#                     "Return ONLY Rust code (no markdown fences).\n\n"
#                     "```java\n" + java_code + "\n```"
#             }
#         ]
#         return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

#     # Build filtered, aligned lists
#     kept_records: List[Dict[str, Any]] = []
#     prompts: List[str] = []
#     for rec in raw_records:
#         java_raw = rec.get("Java") or rec.get("java")
#         java_code = java_raw.get("bug_source_code") if isinstance(java_raw, dict) else java_raw
#         if java_code and str(java_code).strip():
#             kept_records.append(rec)
#             prompts.append(to_chat_prompt(java_code))

#     if not prompts:
#         print("No valid Java code found to translate."); return

#     sampling = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

#     # Warm up (ensures kernels compiled & GPU activity visible)
#     _ = llm.generate([to_chat_prompt("class A{public static void main(String[]a){System.out.println(1);}}")],
#                      SamplingParams(temperature=0.0, max_tokens=8))

#     # Batched generation (outputs aligned 1:1 with prompts)
#     batch = args.batch_size
#     print(f"[vllm] batched generation: batch_size={batch}")
#     empty_count = 0

#     with jsonlines.open(out_path, "w") as writer, \
#          tqdm(total=len(kept_records), desc="Translating (vLLM batched)", unit="rec") as pbar:

#         for s in range(0, len(prompts), batch):
#             sub_prompts = prompts[s:s+batch]
#             sub_recs    = kept_records[s:s+batch]

#             outs = llm.generate(sub_prompts, sampling)
#             for rec, out_obj in zip(sub_recs, outs):
#                 text = out_obj.outputs[0].text if out_obj.outputs else ""
#                 trans_rust = strip_md_fences(text)

#                 if not trans_rust:
#                     empty_count += 1
#                     trans_rust = "ERROR: empty generation"

#                 java_raw = rec.get("Java") or rec.get("java")
#                 java_obj = as_java_object(java_raw if java_raw is not None else "")
#                 rust_raw = rec.get("Rust")
#                 rust_gt  = rust_raw.get("bug_source_code") if isinstance(rust_raw, dict) else (rust_raw if isinstance(rust_raw, str) else None)

#                 out_line = {
#                     "Java": java_obj,
#                     "Trans_Rust": as_code_object(trans_rust),
#                     "GT_Rust": as_code_object(rust_gt)
#                 }
#                 if "hidden_unit_tests" in rec:
#                     out_line["hidden_unit_tests"] = rec["hidden_unit_tests"]
#                 writer.write(out_line)
#                 pbar.update(1)

#     if empty_count:
#         print(f"[warn] {empty_count} generations were empty. Consider lowering --temperature, increasing --max-tokens, "
#               f"or verifying the model ID/chat template.")
#     print(f"✅ Done! Translated {len(kept_records)} records → {out_path}")

# # =============================== CLI & Main ============================== #

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--engine", choices=["auto", "vllm", "openai"], default="vllm",
#                     help="vllm=load model in this process; openai=HTTP; auto=try vllm then HTTP")
#     ap.add_argument("--input", default="/data/hassan/java2rust-automated-migration/data/processed/test_2048.jsonl")
#     ap.add_argument("--output", default="/data/hassan/java2rust-automated-migration/outputs")
#     ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct",
#                     help="vllm: HF id or local path. openai: served name.")
#     ap.add_argument("--limit", type=int, default=0)
#     ap.add_argument("--temperature", type=float, default=0.2)
#     ap.add_argument("--max-tokens", type=int, default=2048)

#     # HTTP
#     ap.add_argument("--base-url", default="http://localhost:8000/v1")
#     ap.add_argument("--api-key", default="hassan2024")
#     ap.add_argument("--workers", type=int, default=6)
#     ap.add_argument("--timeout", type=int, default=60)

#     # vLLM
#     ap.add_argument("--device", type=str, default=None, help="GPU index to use (e.g., '3').")
#     ap.add_argument("--batch-size", type=int, default=48)
#     ap.add_argument("--tensor-parallel-size", type=int, default=1)
#     ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
#     ap.add_argument("--max-model-len", type=int, default=8192)
#     ap.add_argument("--max-num-seqs", type=int, default=0)
#     ap.add_argument("--dtype", default="float16", choices=["auto", "float16", "bfloat16", "float32"])
#     ap.add_argument("--enforce-eager", action="store_true")

#     args = ap.parse_args()
#     Path(args.output).mkdir(parents=True, exist_ok=True)

#     # Stable multiprocessing on some boxes
#     os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

#     if args.engine == "vllm":
#         run_vllm(args)
#     elif args.engine == "openai":
#         run_openai(args)
#     else:
#         try:
#             run_vllm(args)
#         except Exception as e:
#             print("[auto] vLLM failed, falling back to HTTP:", e)
#             run_openai(args)

# if __name__ == "__main__":
#     main()