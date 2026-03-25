import os
import re
import json
import glob
from pathlib import Path
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Config
# ----------------------------
BASE = Path("/data/hassan/java2rust-automated-migration/results/javarusttrans/iterations")

PASS_TO_ITER = {
    "first-pass": 0,
    "second-pass": 1,
    "third-pass": 2,
    "fourth-pass": 3,
    "fifth-pass": 4,
}

SUBDIR = "ragcat-rules-gptsummary"

# Choose how to compute CA:
#   - "binary": mean(ca) * 100   (if ca is 0/1)
#   - "testpass": sum(passed) / sum(total) * 100  (percent unit tests passed)
CA_MODE = "binary"   # change to "testpass" if you want passed/total

OUT_CSV = "iter_results.csv"
OUT_PNG = "iter_effect.png"
OUT_PDF = "iter_effect.pdf"


# ----------------------------
# Helpers
# ----------------------------
def parse_family_and_size(filename: str):
    """
    Examples:
      CodeLlama-7b-Instruct-hf.jsonl          -> (CodeLlama, 7B)
      Deepseek-Coder-33b-Instruct.jsonl       -> (Deepseek-Coder, 33B)
      Qwen2.5-Coder-14B-Instruct.jsonl        -> (Qwen2.5-Coder, 14B)
    """
    name = Path(filename).stem

    if name.startswith("CodeLlama-"):
        fam = "CodeLlama"
        m = re.search(r"CodeLlama-(\d+)\s*b", name, flags=re.I)
        size = f"{m.group(1)}B" if m else "UNK"
    elif name.startswith("Deepseek-Coder-"):
        fam = "Deepseek-Coder"
        m = re.search(r"Deepseek-Coder-([\d.]+)\s*b", name, flags=re.I)
        size = f"{m.group(1)}B" if m else "UNK"
    elif name.startswith("Qwen2.5-Coder-"):
        fam = "Qwen2.5-Coder"
        m = re.search(r"Qwen2\.5-Coder-(\d+)\s*B", name, flags=re.I)
        size = f"{m.group(1)}B" if m else "UNK"
    else:
        fam, size = "Other", "UNK"

    # normalize (7b -> 7B)
    size = size.replace("b", "B")
    return fam, size


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def aggregate_jsonl(path: Path):
    """
    Returns:
      csr_pct: mean(csr)*100
      ca_pct: based on CA_MODE
    """
    n = 0
    csr_sum = 0.0
    ca_sum = 0.0
    passed_sum = 0.0
    total_sum = 0.0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            csr_sum += safe_float(obj.get("csr", 0))
            ca_sum += safe_float(obj.get("ca", 0))
            passed_sum += safe_float(obj.get("passed", 0))
            total_sum += safe_float(obj.get("total", 0))
            n += 1

    if n == 0:
        return None

    csr_pct = (csr_sum / n) * 100.0

    if CA_MODE == "binary":
        ca_pct = (ca_sum / n) * 100.0
    elif CA_MODE == "testpass":
        ca_pct = (passed_sum / total_sum) * 100.0 if total_sum > 0 else 0.0
    else:
        raise ValueError("CA_MODE must be 'binary' or 'testpass'")

    return csr_pct, ca_pct


def size_sort_key(size_str: str):
    # "6.7B" -> 6.7, "34B" -> 34.0
    m = re.match(r"([\d.]+)B", size_str)
    return float(m.group(1)) if m else 9999.0


# ----------------------------
# Build tidy table
# ----------------------------
rows = []

for pass_name, it in PASS_TO_ITER.items():
    folder = BASE / pass_name / SUBDIR
    if not folder.exists():
        print(f"[WARN] missing folder: {folder}")
        continue

    for fp in sorted(folder.glob("*.jsonl")):
        fam, size = parse_family_and_size(fp.name)
        agg = aggregate_jsonl(fp)
        if agg is None:
            print(f"[WARN] empty file: {fp}")
            continue
        csr_pct, ca_pct = agg
        rows.append({
            "family": fam,
            "size": size,
            "iter": it,
            "CSR": csr_pct,
            "CA": ca_pct,
            "file": str(fp),
        })

df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit("No data found. Check BASE path and folders.")

df.to_csv(OUT_CSV, index=False)
print(f"[OK] wrote {OUT_CSV} with {len(df)} rows")


# ----------------------------
# Plot Irene-style 2x3 grid
# ----------------------------
families = ["Deepseek-Coder", "Qwen2.5-Coder", "CodeLlama"]
metric_rows = [("CA", "CA (%)"), ("CSR", "CSR (%)")]
iters = sorted(df["iter"].unique())

fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)

for col, fam in enumerate(families):
    dff = df[df["family"] == fam].copy()

    if dff.empty:
        # no data for this family
        for row in range(2):
            axes[row, col].set_title(fam)
            axes[row, col].text(0.5, 0.5, "No data", ha="center", va="center")
            axes[row, col].set_axis_off()
        continue

    sizes = sorted(dff["size"].unique(), key=size_sort_key)

    for row, (metric, ylabel) in enumerate(metric_rows):
        ax = axes[row, col]
        for size in sizes:
            tmp = dff[dff["size"] == size].sort_values("iter")
            ax.plot(tmp["iter"], tmp[metric], marker="o", linewidth=2, label=size)

        ax.set_title(fam)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        ax.set_xticks(iters)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, frameon=True)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
plt.savefig(OUT_PDF, bbox_inches="tight")
print(f"[OK] saved {OUT_PNG} and {OUT_PDF}")
plt.show()
