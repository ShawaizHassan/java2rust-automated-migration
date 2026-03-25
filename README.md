# 🚀 JavaRustTrans: LLM-Based Java → Rust Translation with Static Analysis and Retrieval

A structured framework for generating **compilable, behavior-preserving Rust code** from Java using LLMs, combining **retrieval, static analysis, semantic scaffolding, and compiler-guided refinement**.

📄 Paper: *Integrating Static Analysis and Retrieval for LLM-Based Java-to-Rust Translation*  
---

## 🔍 Overview

Java-to-Rust translation is challenging due to:
- strict ownership & type system in Rust
- subtle I/O and behavioral differences
- frequent compilation failures in LLM outputs

**JavaRustTrans** addresses these challenges by transforming translation into a **structured pipeline** instead of naive prompting.

> Instead of “just prompt the model”, we guide it with:
> **relevant examples + rules + intent + compiler feedback**

📊 Results (from paper):
- **+16.64 CA (Computational Accuracy)**
- **+10.62 CSR (Compilation Success Rate)**  
compared to strong baselines :contentReference[oaicite:0]{index=0}

---

## 🧠 Key Idea

JavaRustTrans decomposes translation into:


Evidence (retrieval)

Rules (static analysis)
Intent (IPO summary)
Repair (compiler feedback)

This leads to:
- ✔ higher compilation success  
- ✔ better functional correctness  
- ✔ low unsafe Rust usage  

---

## 🏗️ Architecture

The framework consists of **4 core components**:

### 1. 🔎 Category-Constrained Retrieval (CC-RAG)
- Detects translation category via static analysis  
- Retrieves **relevant Java–Rust examples** (not random/global)

---

### 2. 📚 Translation Rules Library (TRL)
Encodes common Java → Rust pitfalls:
- ownership & borrowing  
- indexing (`usize`)  
- I/O parsing  
- data structures  

Provides **grounded, actionable hints**

---

### 3. 🧾 IPO-Structured Summarization
Extracts program intent:


Input → Process → Output


Helps preserve:
- behavior  
- I/O contract  
- algorithm logic  

---

### 4. 🔧 Compiler-Driven Refinement
- Runs `rustc` on generated code  
- Uses error messages to fix issues  
- Performs **one-step minimal repair**

---

## ⚙️ Pipeline


Java Code
↓
Static Analysis → Rule Category
↓
CC-RAG Retrieval + TRL Rules
↓
IPO Summary
↓
LLM Translation
↓
(Optional) Compile & Fix
↓
Final Rust Code


---

## 📊 Results

JavaRustTrans consistently outperforms baselines across **7 open-source LLMs**:

| Method | CA ↑ | CSR ↑ |
|------|------|------|
| Instruction | Low | Low |
| ICL | Medium | Medium |
| RAG | Strong | Strong |
| **JavaRustTrans** | **Best** | **Best** |

✔ Improvements are consistent across model families  
✔ Unsafe Rust usage remains low :contentReference[oaicite:1]{index=1}  

---

## 📁 Project Structure


.
├── baselines/ # Instruction, ICL, CoT, RAG
├── method/ # JavaRustTrans pipeline
├── retrieve/ # BM25 + CC-RAG
├── eval/ # Evaluation scripts
├── scripts/ # Utilities & analysis
├── prompts/ # All prompt templates
├── Dockerfile # Reproducible environment


---

## 🚀 Getting Started

### 1. Clone repository

```bash
git clone https://github.com/yourusername/java2rust-automated-migration.git
cd java2rust-automated-migration
2. Run with Docker (recommended)
docker build -t javarusttrans .
docker run --gpus all -it javarusttrans
3. Run experiments
python baselines/vllm-run/baseline.py
python method/main.py
🧪 Evaluation

We evaluate using:

CSR (Compilation Success Rate) → does it compile?
CA (Computational Accuracy) → does it pass tests?
UR / UR-LOC → unsafe Rust usage

✔ Rust code compiled using rustc
✔ Correctness verified via unit tests

🧩 Prompts

All prompts are modular and stored in:

prompts/

Includes:

baseline prompts (Instruction, ICL, CoT, RAG)
method prompts (JavaRustTrans)
summary prompts (IPO)
refinement prompts (compiler fix)
🔬 Contributions
✅ Category-aware retrieval (CC-RAG)
✅ Translation Rules Library (TRL)
✅ IPO-based semantic scaffolding
✅ Compiler-driven refinement
✅ End-to-end structured translation pipeline
⚠️ Notes
Dataset is not included (can be reconstructed)
Outputs are omitted for size
Deterministic decoding (temperature = 0)
📌 Future Work
Better handling of type/trait mismatches
Improved retrieval filtering
Multi-step refinement
Support for larger real-world codebases
📜 Citation
@article{javarusttrans2026,
  title={Integrating Static Analysis and Retrieval for LLM-Based Java-to-Rust Translation},
  author={Anonymous},
  year={2026}
}