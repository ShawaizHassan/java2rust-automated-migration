#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Evaluator Script
# ============================================================

SCRIPT="/data/hassan/java2rust-automated-migration/eval/eval.py"
OUTROOT="/data/hassan/java2rust-automated-migration/results/JavaRustTrans"
SUMMARY_FILE="${OUTROOT}/summary.txt"

mkdir -p "$OUTROOT"
: > "$SUMMARY_FILE"


FILES=(

/data/hassan/java2rust-automated-migration/experiments/JavaRustTrans/no-refinement/instruction/Qwen2.5-Coder-32B-Instruct.jsonl
/data/hassan/java2rust-automated-migration/experiments/JavaRustTrans/no-refinement/rules/Qwen2.5-Coder-32B-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/third-pass/ragcat-rules-gptsummary/CodeLlama-7b-Instruct-hf.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/third-pass/ragcat-rules-gptsummary/CodeLlama-34b-Instruct-hf.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/third-pass/ragcat-rules-gptsummary/Deepseek-Coder-6.7b-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/third-pass/ragcat-rules-gptsummary/Deepseek-Coder-33b-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/third-pass/ragcat-rules-gptsummary/Qwen2.5-Coder-7B-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/third-pass/ragcat-rules-gptsummary/Qwen2.5-Coder-14B-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/third-pass/ragcat-rules-gptsummary/Qwen2.5-Coder-32B-Instruct.jsonl

# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/fourth-pass/ragcat-rules-gptsummary/CodeLlama-7b-Instruct-hf.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/fourth-pass/ragcat-rules-gptsummary/CodeLlama-34b-Instruct-hf.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/fourth-pass/ragcat-rules-gptsummary/Deepseek-Coder-6.7b-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/fourth-pass/ragcat-rules-gptsummary/Deepseek-Coder-33b-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/fourth-pass/ragcat-rules-gptsummary/Qwen2.5-Coder-7B-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/fourth-pass/ragcat-rules-gptsummary/Qwen2.5-Coder-14B-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/fourth-pass/ragcat-rules-gptsummary/Qwen2.5-Coder-32B-Instruct.jsonl

# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/second-pass/ragcat-rules-gptsummary/CodeLlama-7b-Instruct-hf.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/second-pass/ragcat-rules-gptsummary/CodeLlama-34b-Instruct-hf.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/second-pass/ragcat-rules-gptsummary/Deepseek-Coder-6.7b-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/second-pass/ragcat-rules-gptsummary/Deepseek-Coder-33b-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/second-pass/ragcat-rules-gptsummary/Qwen2.5-Coder-7B-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/second-pass/ragcat-rules-gptsummary/Qwen2.5-Coder-14B-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/second-pass/ragcat-rules-gptsummary/Qwen2.5-Coder-32B-Instruct.jsonl

# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/fifth-pass/ragcat-rules-gptsummary/CodeLlama-7b-Instruct-hf.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/fifth-pass/ragcat-rules-gptsummary/CodeLlama-34b-Instruct-hf.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/fifth-pass/ragcat-rules-gptsummary/Deepseek-Coder-6.7b-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/fifth-pass/ragcat-rules-gptsummary/Deepseek-Coder-33b-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/fifth-pass/ragcat-rules-gptsummary/Qwen2.5-Coder-7B-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/fifth-pass/ragcat-rules-gptsummary/Qwen2.5-Coder-14B-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/fifth-pass/ragcat-rules-gptsummary/Qwen2.5-Coder-32B-Instruct.jsonl

# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/first-pass/ragcat-rules-gptsummary/CodeLlama-7b-Instruct-hf.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/first-pass/ragcat-rules-gptsummary/CodeLlama-34b-Instruct-hf.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/first-pass/ragcat-rules-gptsummary/Deepseek-Coder-6.7b-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/first-pass/ragcat-rules-gptsummary/Deepseek-Coder-33b-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/first-pass/ragcat-rules-gptsummary/Qwen2.5-Coder-7B-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/first-pass/ragcat-rules-gptsummary/Qwen2.5-Coder-14B-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/first-pass/ragcat-rules-gptsummary/Qwen2.5-Coder-32B-Instruct.jsonl





### Baselines ###
# /data/hassan/java2rust-automated-migration/experiments/baselines/COT/CodeLlama-7b-Instruct-hf.jsonl
# /data/hassan/java2rust-automated-migration/experiments/baselines/COT/CodeLlama-34b-Instruct-hf.jsonl
# /data/hassan/java2rust-automated-migration/experiments/baselines/COT/deepseek-coder-1.3b-instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/baselines/COT/Deepseek-Coder-6.7b-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/baselines/COT/Deepseek-Coder-33b-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/baselines/COT/Qwen2.5-Coder-7B-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/baselines/COT/Qwen2.5-Coder-14B-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/baselines/COT/Qwen2.5-Coder-32B-Instruct.jsonl

# /data/hassan/java2rust-automated-migration/experiments/baselines/ICL/CodeLlama-7b-Instruct-hf.jsonl
# /data/hassan/java2rust-automated-migration/experiments/baselines/ICL/CodeLlama-34b-Instruct-hf.jsonl
# /data/hassan/java2rust-automated-migration/experiments/baselines/ICL/deepseek-coder-1.3b-instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/baselines/ICL/Deepseek-Coder-6.7b-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/baselines/ICL/Deepseek-Coder-33b-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/baselines/ICL/Qwen2.5-Coder-7B-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/baselines/ICL/Qwen2.5-Coder-14B-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/baselines/ICL/Qwen2.5-Coder-32B-Instruct.jsonl

# /data/hassan/java2rust-automated-migration/experiments/baselines/Instruction/CodeLlama-7b-Instruct-hf.jsonl
# /data/hassan/java2rust-automated-migration/experiments/baselines/Instruction/CodeLlama-34b-Instruct-hf.jsonl
# /data/hassan/java2rust-automated-migration/experiments/baselines/Instruction/deepseek-coder-1.3b-instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/baselines/Instruction/Deepseek-Coder-6.7b-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/baselines/Instruction/Deepseek-Coder-33b-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/baselines/Instruction/Qwen2.5-Coder-7B-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/baselines/Instruction/Qwen2.5-Coder-14B-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/baselines/Instruction/Qwen2.5-Coder-32B-Instruct.jsonl

# /data/hassan/java2rust-automated-migration/experiments/baselines/RAG/CodeLlama-7b-Instruct-hf.jsonl
# /data/hassan/java2rust-automated-migration/experiments/baselines/RAG/CodeLlama-34b-Instruct-hf.jsonl
# /data/hassan/java2rust-automated-migration/experiments/baselines/RAG/deepseek-coder-1.3b-instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/baselines/RAG/Deepseek-Coder-6.7b-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/baselines/RAG/Deepseek-Coder-33b-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/baselines/RAG/Qwen2.5-Coder-7B-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/baselines/RAG/Qwen2.5-Coder-14B-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/baselines/RAG/Qwen2.5-Coder-32B-Instruct.jsonl

### Ablations ####
# # ragcat
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/ablation/ragcat/CodeLlama-7b-Instruct-hf.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/ablation/ragcat/CodeLlama-34b-Instruct-hf.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/ablation/ragcat/deepseek-coder-1.3b-instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/ablation/ragcat/Deepseek-Coder-6.7b-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/ablation/ragcat/Deepseek-Coder-33b-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/ablation/ragcat/Qwen2.5-Coder-7B-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/ablation/ragcat/Qwen2.5-Coder-14B-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/ablation/ragcat/Qwen2.5-Coder-32B-Instruct.jsonl

# # w/o signals (ragcat gptsummary)
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/ablation/ragcat-gptsummary/CodeLlama-7b-Instruct-hf.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/ablation/ragcat-gptsummary/CodeLlama-34b-Instruct-hf.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/ablation/ragcat-gptsummary/deepseek-coder-1.3b-instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/ablation/ragcat-gptsummary/Deepseek-Coder-6.7b-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/ablation/ragcat-gptsummary/Deepseek-Coder-33b-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/ablation/ragcat-gptsummary/Qwen2.5-Coder-7B-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/ablation/ragcat-gptsummary/Qwen2.5-Coder-14B-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/ablation/ragcat-gptsummary/Qwen2.5-Coder-32B-Instruct.jsonl

# # w/o summary (ragcat-signals)
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/ablation/ragcat-signals/CodeLlama-7b-Instruct-hf.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/ablation/ragcat-signals/CodeLlama-34b-Instruct-hf.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/ablation/ragcat/deepseek-coder-1.3b-instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/ablation/ragcat-signals/Deepseek-Coder-6.7b-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/ablation/ragcat-signals/Deepseek-Coder-33b-Instruct0.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/ablation/ragcat-signals/Qwen2.5-Coder-7B-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/ablation/ragcat-signals/Qwen2.5-Coder-14B-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/ablation/ragcat-signals/Qwen2.5-Coder-32B-Instruct.jsonl

# # w/o ccrag (refernces-gptsummary)
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/ablation/signals-gptsummary/CodeLlama-7b-Instruct-hf.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/ablation/ragcat-gptsummary/CodeLlama-34b-Instruct-hf.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/ablation/ragcat-gptsummary/deepseek-coder-1.3b-instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/ablation/signals-gptsummary/Deepseek-Coder-6.7b-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/ablation/signals-gptsummary/Deepseek-Coder-33b-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/ablation/signals-gptsummary/Qwen2.5-Coder-7B-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/ablation/signals-gptsummary/Qwen2.5-Coder-14B-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/ablation/signals-gptsummary/Qwen2.5-Coder-32B-Instruct.jsonl


### JavaRustTrans ###

# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/no-refinement/ragcat-signals-gptsummary/CodeLlama-7b-Instruct-hf.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/no-refinement/ragcat-signals-gptsummary/CodeLlama-34b-Instruct-hf.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/no-refinement/ragcat-signals-gptsummary/deepseek-coder-1.3b-instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/no-refinement/ragcat-signals-gptsummary/Deepseek-Coder-6.7b-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/no-refinement/ragcat-signals-gptsummary/Deepseek-Coder-33b-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/no-refinement/ragcat-signals-gptsummary/Qwen2.5-Coder-7B-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/no-refinement/ragcat-signals-gptsummary/Qwen2.5-Coder-14B-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/no-refinement/ragcat-signals-gptsummary/Qwen2.5-Coder-32B-Instruct.jsonl

### Refined JavaRustTrans ###

# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/ragcat-signals-gptsummary/CodeLlama-7b-Instruct-hf.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/ragcat-signals-gptsummary/CodeLlama-34b-Instruct-hf.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/ragcat-signals-gptsummary/deepseek-coder-1.3b-instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/ragcat-signals-gptsummary/Deepseek-Coder-6.7b-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/ragcat-signals-gptsummary/Deepseek-Coder-33b-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/ragcat-signals-gptsummary/Qwen2.5-Coder-7B-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/ragcat-signals-gptsummary/Qwen2.5-Coder-14B-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/ragcat-signals-gptsummary/Qwen2.5-Coder-32B-Instruct.jsonl
)

echo "Running evaluator on ${#FILES[@]} files..."
echo ""

# ============================================================
# MAIN LOOP
# ============================================================

for INPUT in "${FILES[@]}"; do
  if [[ ! -f "$INPUT" ]]; then
    echo "[SKIP] Missing file: $INPUT" | tee -a "$SUMMARY_FILE"
    continue
  fi;

  # METHOD = parent directory name (e.g., rag-contexts-summary)
  METHOD="$(basename "$(dirname "$INPUT")")"
  MODEL="$(basename "$INPUT" .jsonl)"

  OUTDIR="${OUTROOT}/${METHOD}"
  OUTFILE="${OUTDIR}/${MODEL}.jsonl"
  mkdir -p "$OUTDIR"

  echo "--------------------------------------------------------"
  echo "Running evaluator for:"
  echo "Method : $METHOD"
  echo "Model  : $MODEL"
  echo "Input  : $INPUT"
  echo "Output : $OUTFILE"
  echo "--------------------------------------------------------"

  python "$SCRIPT" --input "$INPUT" --out "$OUTFILE" > tmp_summary.txt

  {
    echo ""
    echo "================ Eval Summary — ${METHOD} ${MODEL} ================"
    cat tmp_summary.txt
    echo "============================================================================"
  } >> "$SUMMARY_FILE"

done

rm -f tmp_summary.txt

echo ""
echo "DONE! summary written to:"
echo "$SUMMARY_FILE"
