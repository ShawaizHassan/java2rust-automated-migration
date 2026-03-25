import os
import sys
import json
import jsonlines
import argparse
from pathlib import Path
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

# Add project root to Python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm

from static_analyzer import JavaASTAnalyzer
from retrieve.bm25.bm25 import setup_bm25_retriever


class Java2RustInfer:
    # maping translation rule categories
    CATEGORIES = [
        "io_parsing",
        "arrays_indexing",
        "numeric_mod_arith",
        "datastructures",
        "mut_ownership",
    ]

    def __init__(self, model_path, rules_path, categories_dir,
                 batch_size=1, load_model=False):

        self.model_path = model_path
        self.translation_rules = json.load(open(rules_path, "r"))
        self.categories_dir = categories_dir
        self.batch_size = batch_size

        # --------------------------
        # BM25 CACHE (per-category)
        # --------------------------
        self.cache_dir = "/data/hassan/java2rust-automated-migration/retrieve/bm25/cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.retriever_cache = {}   # In-memory retriever cache

        # Optional debug switch
        self.rag_debug = os.environ.get("JavaRustTrans_RAG_DEBUG", "0") == "1"

        # --------------------------
        # Global RAG retriever (fallback when no category matched)
        # --------------------------
        self.rag_corpus_path = "/data/hassan/java2rust-automated-migration/data/retrieval/processed/ragcorpus.jsonl"
        self.bm25_retriever = None
        if os.path.exists(self.rag_corpus_path):
            self.bm25_retriever = setup_bm25_retriever(
                base_dir=self.cache_dir,
                sample_path=self.rag_corpus_path,
                cache_path="ragcorpus_bm25_cache.json",
                top_k=8  # candidates; we will pick a small number for few-shot below
            )
        else:
            if self.rag_debug:
                print(f"[RAG_DEBUG] RAG corpus not found: {self.rag_corpus_path}")

        # --------------------------
        # Model Loading
        # --------------------------
        if load_model:
            self.llm = LLM(
                model=model_path,
                tensor_parallel_size=2,
                max_model_len=8112,
                gpu_memory_utilization=0.90,
                max_num_seqs=32,
                enforce_eager=False
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            self.llm = None
            self.tokenizer = None

        # --------------------------
        # System Instruction
        # --------------------------
        self.system_prompt = (
            "You are an expert in both Java and Rust. "
            "Your task is to translate Java code into Rust."
        )

        # --------------------------
        # User Template
        # --------------------------
        self.USER_TEMPLATE = (
            "I will provide Java code and translation rule hints.\n"
            "Your job is to translate **only** the Java code into Safe Rust.\n\n"
            "Preserve behavior exactly: replicate the same algorithm, control flow, edge cases, and I/O format—do not simplify, optimize, or refactor.\n\n"
            "### Java Code to translate:\n"
            "```java\n"
            "{java_code}\n"
            "```\n"
            "### Translation examples (DO NOT TRANSLATE)\n"
            "{few_shot}\n"
            "### Summary (Reference Only)\n"
            "This section contains a summary about the Java code. Use it as **guidance**.\n"
            "{summary}\n"
            "### Translation Rule Hints\n"
            "This section contains Translation Rule Hints about the Java and Rust code. Use as guidance.\n"
            "{translation_rules}\n"
            "### Reminder\n"
            "1. Carefully read and understand the Java code, and Translation Rule Hints.\n"
            "2. Output **only** the Safe Rust code — no explanations, comments, or additional text.\n"
        )

    # =====================================================================
    # INTERNAL: build translation_rules as bullet points (instead of JSON)
    # =====================================================================
    def _translation_rules_to_bullets(self, translation_rules_for_prompt: dict) -> str:
        if not translation_rules_for_prompt:
            return "None"

        def _ref_num(k: str) -> int:
            # "rule_matches12" -> 12
            num = "".join(ch for ch in k if ch.isdigit())
            return int(num) if num else 999999

        lines_out = []
        for ref_name in sorted(translation_rules_for_prompt.keys(), key=_ref_num):
            lines_out.append(f"- **{ref_name}**")
            for it in (translation_rules_for_prompt.get(ref_name) or []):
                snip = (it.get("Code snippet") or "").replace("`", "'").strip()
                rt = (it.get("Suggested_rust_type") or "").strip()
                rs = (it.get("Reason") or "").strip()

                lns = it.get("Lines")
                if lns is None:
                    # fallback if old format still exists
                    lns = it.get("Line")
                if isinstance(lns, list):
                    ltxt = ", ".join(str(x) for x in lns if x is not None)
                elif lns is None:
                    ltxt = ""
                else:
                    ltxt = str(lns)

                bullet = "  - "
                if ltxt:
                    bullet += f"[lines {ltxt}] "
                if snip:
                    bullet += f"`{snip}` "
                if rt:
                    bullet += f"→ **{rt}**"
                if rs:
                    bullet += f" — {rs}"
                lines_out.append(bullet.rstrip())

        return "\n".join(lines_out)

    # =====================================================================
    # INTERNAL: build category-specific retrieval query (simple)
    # =====================================================================
    def _make_cat_query(self, full_java: str, cat_translation_rules: dict, cat: str) -> str:
        """
        Simple category query:
        query = original Java code + analyzer snippets for this category (if any)
        """
        snippets = []
        for item in (cat_translation_rules.get(cat) or []):
            s = item.get("Code snippet") or item.get("code") or ""
            if s:
                snippets.append(s)

        return full_java + ("\n\n" + "\n".join(snippets) if snippets else "")

    # =====================================================================
    # INTERNAL: Get retriever with PERSISTENT CACHE SUPPORT
    # =====================================================================
    def _get_cached_retriever(self, category: str, fp: str, top_k: int):
        """
        Returns a retriever from in-memory cache if available,
        otherwise builds one with a per-category CACHE FILE.
        """
        cache_key = (category, top_k)

        if cache_key in self.retriever_cache:
            return self.retriever_cache[cache_key]

        retriever = setup_bm25_retriever(
            base_dir=self.cache_dir,
            sample_path=fp,
            cache_path=f"{category}_bm25_cache.json",
            top_k=top_k
        )

        self.retriever_cache[cache_key] = retriever
        return retriever

    # =====================================================================
    # category-SPECIFIC RAG RETRIEVER (NO FILTERING; DEDUP ONLY)
    # =====================================================================
    def retrieve_examples(self, category: str, query_code: str, k: int = 1):
        fp = os.path.join(self.categories_dir, f"{category}.jsonl")
        if not os.path.exists(fp):
            if self.rag_debug:
                print(f"[RAG_DEBUG] missing corpus file for {category}: {fp}")
            return []

        retriever = self._get_cached_retriever(
            category=category,
            fp=fp,
            top_k=max(8, k + 6)
        )

        ranked = retriever(query_code)
        if not ranked:
            if self.rag_debug:
                print(f"[RAG_DEBUG] bm25 returned 0 results for {category}")
            return []

        # Dedup (within-category)
        seen = set()
        unique = []
        for ex in ranked:
            key = ex["java_code"][:150]
            if key not in seen:
                seen.add(key)
                unique.append(ex)

        return unique[:k]

    # =====================================================================
    # Global RAG retrieval (fallback)
    # =====================================================================
    def _retrieve_global_rag(self, java_code: str):
        if not self.bm25_retriever:
            return []
        return self.bm25_retriever(java_code) or []

    # =====================================================================
    # BUILD FINAL PROMPT
    # =====================================================================
    def build_prompt(self, java_code, summary_text):

        analyzer = JavaASTAnalyzer(
            language_path="/data/hassan/tree-sitter-langs/build/java-languages.so",
            java_code=java_code,
            translation_rules=self.translation_rules
        )

        # --------------------------
        # 1) Top-2 categories by score (ONLY)
        # --------------------------
        scores = analyzer.score_category() if hasattr(analyzer, "score_category") else {}
        scores = {c: int(scores.get(c, 0)) for c in self.CATEGORIES}

        detected_sorted = [c for c, sc in sorted(scores.items(), key=lambda kv: kv[1], reverse=True) if sc > 0]
        detected_sorted = detected_sorted[:2]  # enforce top-2

        # --------------------------
        # 2) Build translation_rules_for_prompt ONLY from top-2
        #    AND apply your analyzer merge/dedup (token-lean)
        # --------------------------
        gen_map = {
            "io_parsing": "gen_io_parsing_rule_matches",
            "arrays_indexing": "gen_arrays_indexing_rule_matches",
            "numeric_mod_arith": "gen_numeric_mod_arith_rule_matches",
            "datastructures": "gen_datastructures_rule_matches",
            "mut_ownership": "gen_mut_ownership_rule_matches",
        }

        translation_rules_for_prompt = {}
        ref = 1
        for cat in detected_sorted:
            fn = gen_map.get(cat)
            if not fn or not hasattr(analyzer, fn):
                continue
            try:
                block = getattr(analyzer, fn)() or []
            except Exception:
                block = []

            # IMPORTANT: keep merge/dedup behavior from static_analyzer
            if hasattr(analyzer, "_dedupe_merge_block"):
                try:
                    block = analyzer._dedupe_merge_block(block, cat)
                except Exception:
                    pass

            if block:
                translation_rules_for_prompt[f"translation_rules{ref}"] = block
                ref += 1

        # bullet-point format instead of JSON
        sig_str = self._translation_rules_to_bullets(translation_rules_for_prompt) if translation_rules_for_prompt else "None"

        # --------------------------
        # 3) Per-category translation_rules for BM25 query enrichment
        # --------------------------
        cat_translation_rules = {c: [] for c in self.CATEGORIES}
        for c, fn in gen_map.items():
            if hasattr(analyzer, fn):
                try:
                    cat_translation_rules[c] = getattr(analyzer, fn)() or []
                except Exception:
                    cat_translation_rules[c] = []

        # --------------------------
        # 4) Few-shot examples (category-first; fallback to global RAG if no category)
        # --------------------------
        few_shot = ""
        used_global = set()

        if len(detected_sorted) == 0:
            retrieved = self._retrieve_global_rag(java_code)

            count = 0
            for ex in retrieved:
                key = (ex.get("java_code") or "")[:150]
                if not key or key in used_global:
                    continue
                used_global.add(key)

                few_shot += (
                    "Java:\n```java\n"
                    + (ex.get("java_code") or "").strip()
                    + "\n```\nTranslated Rust:\n```rust\n"
                    + (ex.get("rust_code") or "").strip()
                    + "\n```\n\n"
                )
                count += 1
                if count == 2:
                    break

            if not few_shot:
                few_shot = "None"

        else:
            top_cat = detected_sorted[0]
            cat_query = self._make_cat_query(java_code, cat_translation_rules, top_cat)
            cat_retrieved = self.retrieve_examples(top_cat, cat_query, k=10)

            second_cat = detected_sorted[1] if len(detected_sorted) > 1 else None
            second_retrieved = []
            if second_cat:
                second_query = self._make_cat_query(java_code, cat_translation_rules, second_cat)
                second_retrieved = self.retrieve_examples(second_cat, second_query, k=10)

            cat_pick = None
            for ex in cat_retrieved:
                key = ex["java_code"][:150]
                if key in used_global:
                    continue
                used_global.add(key)
                cat_pick = ex
                break

            second_pick = None
            for ex in second_retrieved:
                key = ex["java_code"][:150]
                if key in used_global:
                    continue
                used_global.add(key)
                second_pick = ex
                break

            if cat_pick:
                few_shot += (
                    "Java:\n```java\n"
                    + cat_pick["java_code"].strip()
                    + "\n```\nTranslated Rust:\n```rust\n"
                    + cat_pick["rust_code"].strip()
                    + "\n```\n\n"
                )

            if second_pick:
                few_shot += (
                    "Java:\n```java\n"
                    + second_pick["java_code"].strip()
                    + "\n```\nTranslated Rust:\n```rust\n"
                    + second_pick["rust_code"].strip()
                    + "\n```\n\n"
                )

            if not few_shot:
                few_shot = "None"

        prompt = self.USER_TEMPLATE.format(
            java_code=java_code,
            few_shot=few_shot,
            summary=summary_text if summary_text else "None",
            translation_rules=sig_str
        )

        return prompt

    # =====================================================================
    # RUN full dataset inference (FASTER: sequential prompt building + batched vLLM generate)
    # =====================================================================
    def run_dataset(self, dataset_jsonl, summary_jsonl, output_path):

        if self.llm is None or self.tokenizer is None:
            raise RuntimeError("Model was not loaded (load_model=False).")

        # --------------------------
        # Load summaries (src_uid → summary)
        # --------------------------
        summary_map = {}
        for row in jsonlines.open(summary_jsonl):
            summary_map[row["src_uid"]] = row["summary"]

        data = list(jsonlines.open(dataset_jsonl))

        # --------------------------
        # Build prompts (SEQUENTIAL: Option A)
        # --------------------------
        prompts = []
        for item in tqdm(data, desc="Building prompts"):
            src_uid = item.get("src_uid") or item.get("Java", {}).get("src_uid") or item.get("Rust", {}).get("src_uid")
            summary_text = summary_map.get(src_uid, "None")
            prompts.append(self.build_prompt(item["Java"]["source_code"], summary_text))

        # --------------------------
        # Build chat prompts (can be parallel safely)
        # --------------------------
        def _to_chat(p: str) -> str:
            return self.tokenizer.apply_chat_template(
                [{"role": "system", "content": self.system_prompt},
                 {"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True
            )

        workers = min(32, max(1, cpu_count() * 2))
        with ThreadPool(processes=workers) as pool:
            chat_prompts = list(tqdm(pool.imap(_to_chat, prompts), total=len(prompts), desc="Formatting chat prompts"))

        sampling = SamplingParams(
            temperature=0,
            top_p=1,
            max_tokens=2048
        )
        model_name = os.path.basename(self.model_path.rstrip("/"))

        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, f"{model_name}.jsonl")

        # --------------------------
        # Batched vLLM generation (GPU throughput)
        # --------------------------
        bs = max(1, int(self.batch_size))
        total_batches = (len(chat_prompts) + bs - 1) // bs

        with jsonlines.open(output_file, "w") as writer:
            for start in tqdm(range(0, len(chat_prompts), bs), total=total_batches, desc="Running inference"):
                batch_cp = chat_prompts[start:start + bs]
                outs = self.llm.generate(batch_cp, sampling)

                for j, out in enumerate(outs):
                    i = start + j
                    answer = out.outputs[0].text.strip()

                    java_src = data[i]["Java"]["source_code"]
                    gt_rust_src = data[i]["Rust"]["source_code"]
                    hidden_tests = data[i].get("hidden_unit_tests", [])
                    refine_tests = data[i].get("refine_unit_tests", [])

                    result = {
                        "java": java_src,
                        "trans_rust": answer,
                        "gt_rust": gt_rust_src,
                        "hidden_unit_tests": hidden_tests,
                        "refine_unit_tests": refine_tests,
                        "method": "JavaRustTrans",
                        "model": model_name
                    }

                    writer.write(result)

                writer._fp.flush()

        print("Saved:", output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Java2Rust JavaRustTrans Inference")
    parser.add_argument("--model_path", type=str,
                        default="/data/jfeng/models/Qwen2.5-Coder-32B-Instruct",
                        help="Path to vLLM model directory")
    parser.add_argument("--rules_path", type=str, default="heuristic_guidance.json", help="Path to translation_rules.json")
    parser.add_argument("--summary_path", type=str,
                        default="/data/hassan/java2rust-automated-migration/experiments/middle_infer_result/summary/gpt-5.2/infer_result.jsonl",
                        help="Path to summary")
    parser.add_argument("--categories_dir", type=str,
                        default="/data/hassan/java2rust-automated-migration/data/retrieval/categories",
                        help="Directory containing category JSONL corpora")
    parser.add_argument("--dataset_jsonl", type=str,
                        default="/data/hassan/java2rust-automated-migration/data/processed/final_test.jsonl",
                        help="Path to input test JSONL dataset")
    parser.add_argument("--output_dir", type=str,
                        default="/data/hassan/java2rust-automated-migration/experiments/JavaRustTrans/no-refinement/rules",
                        help="Directory to write results.jsonl")
    parser.add_argument("--no_model", action="store_true", help="Debug mode: don't load model, only build prompts")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for vLLM generate (higher = more GPU throughput, limited by memory)")
    args = parser.parse_args()

    infer = Java2RustInfer(
        model_path=args.model_path,
        rules_path=args.rules_path,
        categories_dir=args.categories_dir,
        batch_size=args.batch_size,
        load_model=not args.no_model,
    )

    infer.run_dataset(
        dataset_jsonl=args.dataset_jsonl,
        summary_jsonl=args.summary_path,
        output_path=args.output_dir
    )
