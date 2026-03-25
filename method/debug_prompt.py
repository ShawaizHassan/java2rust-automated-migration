import os
import json
import jsonlines
import argparse
import tempfile
from javarusttrans import Java2RustInfer


# --------------------------
# Dummy objects to bypass vLLM/tokenizer (so we can call infer.run_dataset())
# --------------------------
class _DummyTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        sys_msg = messages[0]["content"] if messages and messages[0].get("role") == "system" else ""
        user_msg = messages[1]["content"] if len(messages) > 1 else ""
        return f"### SYSTEM\n{sys_msg}\n\n### USER\n{user_msg}\n"


class _DummyText:
    def __init__(self, text):
        self.text = text


class _DummyOutput:
    def __init__(self, text):
        self.outputs = [_DummyText(text)]


class _DummyLLM:
    def generate(self, prompts, sampling_params):
        # echo each prompt back so batching works
        return [_DummyOutput(p) for p in prompts]


def main():
    parser = argparse.ArgumentParser(
        description="Debug prompts for first K items using Java2RustInfer.run_dataset()"
    )

    # ---- required inputs ----
    parser.add_argument(
        "--dataset_jsonl",
        type=str,
        default="/data/hassan/java2rust-automated-migration/data/processed/final_test.jsonl",
        help="Dataset JSONL path"
    )
    parser.add_argument(
        "--summary_jsonl",
        type=str,
        default="/data/hassan/java2rust-automated-migration/experiments/middle_infer_result/summary/gpt-5.2/infer_result.jsonl",
        help="Summary JSONL path (src_uid -> summary)"
    )
    parser.add_argument(
        "--rules_path",
        type=str,
        default="translation_rules.json",
        help="translation_rules.json path"
    )
    parser.add_argument(
        "--categories_dir",
        type=str,
        default="/data/hassan/java2rust-automated-migration/data/retrieval/categories",
        help="categories dir"
    )

    # ---- debug controls ----
    parser.add_argument(
        "--k",
        type=int,
        default=522,
        help="First K dataset items to dump prompts for"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/hassan/java2rust-automated-migration/my-algo/prompts",
        help="Output dir for prompt_*.txt"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="dummy-debug",
        help="Only used to name the intermediate JSONL (model_name.jsonl)"
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Write FIRST K dataset items to a temp JSONL (so run_dataset only processes K)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as tmp:
        tmp_path = tmp.name
        kept = 0
        for item in jsonlines.open(args.dataset_jsonl):
            tmp.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1
            if kept >= args.k:
                break

    # Init infer but DO NOT load a real model
    infer = Java2RustInfer(
        model_path=args.model_path,
        rules_path=args.rules_path,
        categories_dir=args.categories_dir,
        batch_size=1,      # keep simple for debugging
        load_model=False
    )

    # Inject dummy model/tokenizer so run_dataset() will run and load summaries internally
    infer.llm = _DummyLLM()
    infer.tokenizer = _DummyTokenizer()

    infer.run_dataset(
        dataset_jsonl=tmp_path,
        summary_jsonl=args.summary_jsonl,
        output_path=args.output_dir
    )

    # Extract echoed prompts from produced JSONL and save as prompt_*.txt
    model_name = os.path.basename(args.model_path.rstrip("/"))
    out_jsonl = os.path.join(args.output_dir, f"{model_name}.jsonl")

    count = 0
    for i, row in enumerate(jsonlines.open(out_jsonl)):
        prompt_text = row["trans_rust"]  # echoed chat prompt that includes summary
        out_path = os.path.join(args.output_dir, f"prompt_{i:03d}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(prompt_text)
        count += 1

    print(f"Saved {count} prompts to: {args.output_dir}")
    print(f"Intermediate JSONL: {out_jsonl}")
    print(f"Temp dataset: {tmp_path}")


if __name__ == "__main__":
    main()
