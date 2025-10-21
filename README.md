# JAVA2RUST: Toward Automated Migration of Java Programs to Rust

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)
![CI](https://img.shields.io/github/actions/workflow/status/OWNER/java2rust-automated-migration/ci.yml?branch=main)
![Python](https://img.shields.io/badge/Python-3.10%20|%203.11-3776AB)
![CUDA](https://img.shields.io/badge/CUDA-optional-informational)

Automated **Java → Rust** migration with **LLM baselines** (vLLM, DeepSeek), **JSONL parallel pairs**, prompts, and evaluation (BLEU, CodeBLEU, compile/test pass).  
This repo aims to be a clean, reproducible starting point for research and practical migration pipelines.

> **SEO keywords:** Java to Rust, automated migration, code translation, LLM, vLLM, DeepSeek, JSONL dataset, CodeBLEU, compile pass.

---

## TL;DR Quickstart

```bash
# 1) Clone
git clone https://github.com/OWNER/java2rust-automated-migration.git
cd java2rust-automated-migration

# 2) Environment (choose one)
bash scripts/setup_env.sh        # creates conda env or venv

# 3) Install
pip install -r requirements.txt
pip install -r requirements-dev.txt   # optional: dev tools

# 4) Run format/lint (optional but recommended)
bash scripts/format.sh
bash scripts/lint.sh

# 5) Try evaluation stubs (will print usage)
python eval/metrics.py --help
python eval/compile_test.py --help
```
