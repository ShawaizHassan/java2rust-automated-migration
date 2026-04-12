# JavaRustTrans

**Category-Conditioned Retrieval and Grounded Hints for Reliable Java-to-Rust Translation**

*Shawaiz Hassan · Ahmad Hassan · Harbin Institute of Technology (Shenzhen)*

---

Direct prompting fails Java-to-Rust translation not randomly — it fails in predictable, category-structured ways: usize/i32 mismatches in array indexing, buffering gaps in I/O parsing, overflow in modular arithmetic, borrow conflicts in data structure manipulation. JavaRustTrans exploits this structure. Instead of prompting blindly, it detects the failure category first, retrieves category-aligned exemplars, injects grounded static hints, scaffolds the program's I/O contract, then runs one bounded compiler-repair pass. The result: **+16.64pp computational accuracy and +10.62pp compilation success** over the strongest RAG baseline, across 522 Java programs and 7 open-source LLMs — with unsafe Rust usage remaining uniformly low.

---

## Results

Evaluated on 522 Java programs curated from [xCodeEval](https://github.com/ntunlp/xCodeEval), across **7 open-source code LLMs**, measuring Computational Accuracy (CA), Compilation Success Rate (CSR), and Unsafe Rust ratio (UR).

### Average across all models (vs. strongest baseline: Global RAG)

| Method | CA (↑) | CSR (↑) | UR (↓) |
|---|---|---|---|
| Instruction (zero-shot) | baseline | baseline | — |
| ICL (global few-shot) | +moderate | +moderate | — |
| Global RAG | strong | strong | low |
| **JavaRustTrans** | **+16.64pp over RAG** | **+10.62pp over RAG** | **low (maintained)** |

### Per-model breakdown

| Model | CA — RAG | CA — JavaRustTrans | Δ CA | CSR — RAG | CSR — JavaRustTrans | Δ CSR |
|---|---|---|---|---|---|---|
| DeepSeek-Coder-6.7B | 21.46 | **42.34** | +20.88 | 74.90 | **77.97** | +3.07 |
| DeepSeek-Coder-33B | 1.53 | **10.34** | +8.81 | 5.17 | **18.97** | +13.80 |
| Qwen2.5-Coder-7B | 33.52 | **52.68** | +19.16 | 49.62 | **72.80** | +23.18 |
| Qwen2.5-Coder-14B | 52.49 | **70.69** | +18.20 | 64.94 | **84.48** | +19.54 |
| CodeLlama-32B | 55.17 | **73.75** | +18.58 | 66.48 | **84.48** | +18.00 |
| CodeLlama-7B | 3.07 | **14.94** | +11.87 | **76.44** | 61.49 | −14.95 ⚠ |
| CodeLlama-34B | 3.64 | **22.61** | +18.97 | 58.43 | **70.11** | +11.68 |
| **Average** | **24.41** | **41.05** | **+16.64** | **56.57** | **67.19** | **+10.62** |

> ⚠ **CodeLlama-7B CSR exception:** JavaRustTrans improves CA substantially (+11.87pp) but attains lower CSR than RAG on this model. As discussed in the paper, retrieval alone can sometimes satisfy compilation constraints on smaller models even when end-to-end functional correctness remains limited. CA improvement confirms the method still works; CSR trade-off is model-specific.

### Ablation: component contribution

| Configuration | CA | CSR |
|---|---|---|
| JavaRustTrans (full) | best | best |
| − CC-RAG (global RAG only) | largest drop | largest drop |
| − Translation Hints | moderate drop | moderate drop |
| − IPO Summarization | small drop | small drop |
| − Compiler Refinement | small drop | noticeable drop |

CC-RAG is the primary driver. Translation Hints, IPO, and compiler refinement provide complementary, additive gains.

---

## Why Translation Fails (And Why JavaRustTrans Fixes It)

LLM-based Java-to-Rust translation fails in structured, recurring patterns. Standard global RAG ignores this structure — it retrieves exemplars by surface similarity, not by failure mode. Two specific gaps motivated this work:

1. **Category-mismatched evidence selection.** Global retrieval injects exemplars from the wrong failure regime. An array-indexing program retrieves I/O examples; the LLM sees irrelevant context and still gets the usize cast wrong.

2. **No bounded use of compiler diagnostics.** The rustc compiler produces precise, line-level error messages. Existing pipelines ignore them. JavaRustTrans uses them for one deterministic repair pass.

### The five failure categories

| Category | What it captures |
|---|---|
| **I/O & Parsing** | Input tokenization, fast I/O, output buffering, numeric parsing defaults |
| **Arrays & Indexing** | Rust `usize` indexing, 1-indexed arrays, bounds/offset safety, 2D array mapping |
| **Numeric & Modular Arithmetic** | Overflow-safe multiplication, safe modular ops, bitmask typing, normalization |
| **Data Structures** | HashMap, BTreeMap, Vec, sorting, DS-specific ownership patterns |
| **Mutability & Ownership** | Borrow patterns, `mut` placement, reference lifetimes, ownership transfer |

---

## Architecture

JavaRustTrans is a three-stage pipeline: evidence selection, intent summarization, then translation with optional compiler repair.

```
Java Program (P)
       │
       ▼
┌──────────────────────────────────────────────┐
│         STAGE 1: Evidence Selection          │
│                                              │
│  Tree-sitter AST Analysis                   │
│    → infer heuristic category c*             │
│    → extract grounded Translation Hints (M)  │
│                                              │
│  CC-RAG (BM25 per-category index)           │
│    → retrieve category-aligned exemplars (E) │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│       STAGE 2: IPO Summarization             │
│                                              │
│  Input:   what the program reads             │
│  Process: core computation                   │
│  Output:  what the program prints            │
│                                              │
│  → intent summary (S)                        │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│    STAGE 3: Translation + Compile-Refine     │
│                                              │
│  Prompt: (P, E, S, M) → R⁰                  │
│  rustc compile R⁰                            │
│    ✓ passes → done                           │
│    ✗ fails  → one repair pass with           │
│               diagnostics → R¹ (final)       │
└──────────────────────────────────────────────┘
```

### Stage 1A: Category-Conditioned Retrieval (CC-RAG)

The Tree-sitter analyzer scores the input Java program against five heuristic categories by inspecting AST node types, API calls, and type usage. The top-scoring category determines which BM25 index to query. Up to two category-aligned Java–Rust exemplar pairs are retrieved as few-shot context. If no category is detected, global BM25 is used as fallback.

This directly addresses the regime-mismatch problem in global RAG.

### Stage 1B: Grounded Translation Hints (HGF)

The Heuristic Guidance Framework (HGF) organizes Java-to-Rust rules by category: hints, idiom mappings, and micro-examples. The analyzer extracts line-anchored match candidates tied to specific AST nodes, then consolidates them by deduplicating on (line, snippet) pairs and keeping the top-priority keys per category. The resulting hint set M is rendered as concise bullets in the prompt.

Hints are grounded — tied to actual lines in the input program, not generic advice.

### Stage 2: IPO-Structured Summarization

A fixed three-line summary exposes the program's I/O contract:

```
Input:   reads N integers from stdin
Process: computes prime factorization and sieve
Output:  prints count of prime divisors per element
```

This is included as reference-only context. It stabilizes behavioral preservation across the compile-refine step.

### Stage 3: Compile-Refine Translation

The LLM generates R⁰ from the full structured prompt (P, E, S, M). If rustc reports errors, a single repair prompt includes R⁰ and the full diagnostic log, requesting minimal edits that preserve algorithm and I/O behavior. Decoding is deterministic (temperature = 0) throughout. Refinement is strictly bounded to one step.

---

## Concrete Example

**Input Java (Array/Indexing category):**

```java
int n = sc.nextInt();
int[] plainNumber = new int[n];
for (int i = 0; i < n; i++) plainNumber[i] = i + 1;
int a = getNumberOfPrimeDivisors(i + 1);
```

**Direct prompting output** — fails with E0277, E0308:

```rust
let n: usize = input.trim().parse().unwrap();
let plain_number = (1..=n).collect::<Vec<i32>>();  // E0277: usize → i32
let a = get_number_of_prime_divisors(i + 1);        // E0308: expected i32, found usize
```

**JavaRustTrans output** — compiles and passes:

```rust
let n: usize = it.next().unwrap().parse().unwrap();
let plain_number = (1..=n as i32).collect::<Vec<i32>>();
let a = get_number_of_prime_divisors((i + 1) as i32);
```

The category detector identified `Array/Indexing (usize ↔ i32)`. CC-RAG retrieved an exemplar showing the cast pattern. The Translation Hints flagged the specific line. The model resolved both errors correctly without needing the compiler repair step.

---

## Repository Structure

```
java2rust-automated-migration/
├── baselines/          # Instruction, ICL, CoT, and Global RAG baselines
├── method/             # JavaRustTrans full pipeline (main.py entry point)
├── retrieve/           # BM25 indexing and CC-RAG retrieval
├── eval/               # CSR, CA, UR/UR-LOC evaluation scripts
├── prompts/            # All prompt templates (baseline + method + IPO + refinement)
├── scripts/            # Setup, formatting, linting, analysis utilities
├── dockerfile          # Reproducible container environment
├── requirements.txt
└── requirements-dev.txt
```

**Language breakdown note:** The pipeline is implemented in Python. Rust appears at 0.3% because the framework *generates* Rust code via LLMs and compiles it with rustc — it does not contain manually-written Rust. The `eval/compile_test.py` script invokes rustc programmatically to measure CSR.

---

## Quickstart

### Option 1: Docker (recommended for full reproducibility)

```bash
git clone https://github.com/ShawaizHassan/java2rust-automated-migration.git
cd java2rust-automated-migration
docker build -t javarusttrans .
docker run --gpus all -it javarusttrans
```

### Option 2: Local environment

```bash
git clone https://github.com/ShawaizHassan/java2rust-automated-migration.git
cd java2rust-automated-migration

# Create environment
bash scripts/setup_env.sh

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt   # optional: linting and dev tools
```

### Run baselines

```bash
python baselines/vllm-run/baseline.py --method instruction
python baselines/vllm-run/baseline.py --method rag
```

### Run JavaRustTrans

```bash
python method/main.py
```

### Evaluate

```bash
python eval/compile_test.py --help    # CSR evaluation (requires rustc)
python eval/metrics.py --help         # CA and UR evaluation
```

**Requirements:** rustc must be installed and on PATH for CSR evaluation. All LLM inference uses vLLM with greedy decoding (temperature = 0).

---

## Benchmark & Dataset

JavaRustTrans is evaluated on **522 Java programs** curated from [xCodeEval](https://github.com/ntunlp/xCodeEval) (ACL 2024), covering competitive programming problems with known I/O contracts and correctness criteria.

Programs were selected and categorized into the five heuristic failure categories. Ground-truth Rust translations are verified for compilation and functional correctness.

> The full dataset reconstruction script is available in `scripts/`. The raw xCodeEval source is publicly available under its original license.

---

## Evaluation Metrics

| Metric | Definition |
|---|---|
| **CA** (Computational Accuracy) | Fraction of programs passing all functional test cases |
| **CSR** (Compilation Success Rate) | Fraction of generated Rust programs that compile without errors |
| **UR** (Unsafe Ratio) | Fraction of translated programs containing `unsafe` blocks |
| **UR-LOC** | Lines of unsafe code as a fraction of total generated lines |

All metrics are computed per-program and averaged across the benchmark. Compiler verification uses rustc directly; functional correctness uses input/output test cases from xCodeEval.

---

## Models Evaluated

| Model | Family | Size |
|---|---|---|
| DeepSeek-Coder-6.7B-Instruct | DeepSeek | 6.7B |
| DeepSeek-Coder-7B-Instruct-v1.5 | DeepSeek | 7B |
| Qwen2.5-Coder-7B-Instruct | Qwen | 7B |
| Qwen2.5-Coder-14B-Instruct | Qwen | 14B |
| CodeLlama-7B-Instruct | CodeLlama | 7B |
| CodeLlama-13B-Instruct | CodeLlama | 13B |
| CodeLlama-34B-Instruct | CodeLlama | 34B |

All models run with greedy decoding (temperature = 0) via vLLM. JavaRustTrans shows consistent gains across all seven model families and sizes.

---

## Limitations

- **Single-step refinement.** Compiler repair is bounded to one iteration. Programs with multiple layered errors may not fully resolve after one pass.
- **Five fixed categories.** The HGF covers the dominant Java-to-Rust failure modes observed in xCodeEval competitive programming. Real-world codebases may surface additional categories (e.g., generics, trait bounds, async patterns) not yet covered.
- **Program-level scope.** JavaRustTrans is designed for self-contained program translation. Repository-level or multi-file migration is future work.
- **Deterministic decoding.** All experiments use temperature = 0. Sampling-based approaches may yield different trade-offs.

---

## Citation

```bibtex
@article{hassan2026javarusttrans,
  title     = {Category-Conditioned Retrieval and Grounded Hints for Reliable Java-to-Rust Translation},
  author    = {Hassan, Shawaiz and Hassan, Ahmad},
  institution = {Harbin Institute of Technology (Shenzhen)},
  year      = {2026}
}
```

---

## License

See [LICENSE](LICENSE) for details.
