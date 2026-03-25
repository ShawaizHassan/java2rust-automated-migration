import os
import json
import hashlib
from typing import List, Tuple, Dict, Any, Optional
import re
import jsonlines
from rank_bm25 import BM25Okapi


# -----------------------------
# Basic utils
# -----------------------------

def _hash_key(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def _preprocess(code: str) -> list[str]:
    code = code.lower()
    # remove comments
    code = re.sub(r"//.*|/\*[\s\S]*?\*/", "", code)
    # replace strings and numbers
    code = re.sub(r'".*?"', "<STR>", code)
    code = re.sub(r"\d+", "<NUM>", code)
    # split camelCase and snake_case
    code = re.sub(r"([a-z])([A-Z])", r"\1 \2", code)
    code = code.replace("_", " ")
    # split on punctuation / non-word characters
    tokens = re.findall(r"[a-zA-Z_]+|<STR>|<NUM>", code)
    return tokens

def _save_cache(cache_path: str, cache_data: Dict[str, Any]) -> None:
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, indent=2, ensure_ascii=False)

def _load_cache(cache_path: str) -> Dict[str, Any]:
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# -----------------------------
# Dataset loading (Java->Rust)
# -----------------------------
# My JSONL schema sample:
# {
#   "Java": {"lang": "Java 6", "source_code": "...", "code_uid": "...", ...},
#   "Rust": {"lang": "Rust",    "source_code": "...", "code_uid": "...", ...},
#   "hidden_unit_tests": [...]
# }

def _load_jsonl(fp: str) -> List[Dict[str, Any]]:
    with jsonlines.open(fp, "r") as r:
        return list(r)

def _pairs_from_java_rust_jsonl(fp: str) -> List[Dict[str, Any]]:
    raw = _load_jsonl(fp)
    pairs = []
    for rec in raw:
        try:
            java = rec.get("Java", {})
            rust = rec.get("Rust", {})
        except Exception:
            continue
        if isinstance(java, str) and java.strip():
            pairs.append({
                # Keep names for backward-compat with few-shot builder:
                "java_code": java,            # NOTE: holds Java code
                "rust_code": rust or "",
            })
    if not pairs:
        raise RuntimeError(f"No usable (Java, Rust) pairs in corpus {fp}")
    return pairs


# -----------------------------
# BM25 core
# -----------------------------

def _build_bm25_index(texts: List[str]) -> BM25Okapi:
    tokenized = [_preprocess(t) for t in texts]
    return BM25Okapi(tokenized)

def _bm25_retrieve(bm25: BM25Okapi, query: str, k: int = 10 ) -> List[Tuple[int, float]]:
    q_tokens = _preprocess(query)
    scores = bm25.get_scores(q_tokens)  # numpy array
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

    out: List[Tuple[int, float]] = []
    for idx, sc in ranked:
        out.append((idx, float(sc)))
        if len(out) >= k:
            break
    return out


# -----------------------------
# Public entry: setup_bm25_retriever
# -----------------------------

def setup_bm25_retriever(
    base_dir: Optional[str] = None,
    sample_path: str = "",
    cache_path: str = "bm25_cache_java2rust.json",
    top_k: int = 4
):
    """
    Create and return a retriever(query_code: str) -> List[Dict]
    - sample_path: JSONL with Java->Rust ground-truth pairs
    - cache_path:  on-disk JSON cache file (stored under base_dir if given)
    - top_k:       number of few-shot examples to return
    Returns entries shaped like:
      {"java_code": <JAVA>, "rust_code": <RUST>, "score": <float>}
    """

    if base_dir is None:
        base_dir = os.path.dirname(__file__)

    cache_path = os.path.join(base_dir, cache_path)

    # 1) Load dataset as pairs and build BM25 corpus
    data = _pairs_from_java_rust_jsonl(sample_path)
    texts = [d["java_code"] for d in data]
    bm25 = _build_bm25_index(texts)

    # 2) warm cache (lazy)
    cache = _load_cache(cache_path)

    def retriever(query_code: str) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most similar Java->Rust examples.
        - query_code: the input Java code string
        """
        key = _hash_key(query_code)
        if key in cache:
            return cache[key]

        ranked = _bm25_retrieve(bm25, query=query_code, k=top_k + 4)
        out: List[Dict[str, Any]] = []
        for idx, sc in ranked:
            ex = data[idx]
            out.append({"java_code": ex["java_code"], "rust_code": ex["rust_code"], "score": sc})
            if len(out) >= top_k:
                break

        cache[key] = out
        _save_cache(cache_path, cache)
        return out

    return retriever


# -----------------------------
# (Optional) quick self-test
# -----------------------------
if __name__ == "__main__":
    # Minimal smoke test (no printing huge stuff)
    rtr = setup_bm25_retriever(
        sample_path="",
        top_k=10
    )
    demo_query = """"""
    hits = rtr(demo_query)
    assert isinstance(hits, list) and hits and {"java_code","rust_code","score"}.issubset(hits[0].keys())
    print("[OK] BM25 retriever ready. Top-1 score:", hits[0]["score"])