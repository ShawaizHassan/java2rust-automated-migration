#!/usr/bin/env python3
import argparse, os, json, pathlib
from datetime import datetime
from typing import Dict, Any, List
from openai import OpenAI
from openai.types.chat import ChatCompletion

Path = pathlib.Path


def load_text(path: Path) -> str:
    """Read a UTF-8 text file."""
    return path.read_text(encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    """Read a JSON file safely."""
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dir(p: Path):
    """Create directory if it doesn’t exist."""
    p.mkdir(parents=True, exist_ok=True)


def main():
    script_dir = Path(__file__).resolve().parent
    default_config = script_dir.parent / "configs" / "vllm" / "qwen2.5-coder-7b.json"

    # -------------------------------
    # CLI arguments
    # -------------------------------
    ap = argparse.ArgumentParser(description="Config-driven baseline runner for Java→Rust translation")
    ap.add_argument("--config", default=str(default_config), help="Path to model config JSON")
    ap.add_argument("--java-file", help="Path to Java file to translate (optional)")
    ap.add_argument("--outdir", default="runs", help="Directory for outputs (default: runs/)")
    ap.add_argument("--n", type=int, help="Override number of completions (n)")
    ap.add_argument("--temperature", type=float, help="Override temperature")
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"❌ Config not found: {config_path}")

    cfg = load_json(config_path)

    # -------------------------------
    # Config parsing
    # -------------------------------
    base_url = cfg.get("base_url", "http://localhost:8000/v1")
    model = cfg.get("model")
    if not model:
        raise ValueError("❌ Config missing required field: 'model'")

    api_key = os.environ.get("OPENAI_API_KEY", cfg.get("api_key", "hassan2024"))
    use_chat = bool(cfg.get("use_chat", False))
    max_tokens = int(cfg.get("max_tokens", 512))
    temperature = args.temperature or float(cfg.get("temperature", 0.2))
    top_p = float(cfg.get("top_p", 0.9))
    n_choices = args.n or int(cfg.get("n", 5))

    # System prompt
    shared_prompt_default = (script_dir.parent / "prompts" / "java2rust_system.txt").resolve()
    system_prompt_path = Path(cfg.get("system_prompt_path", shared_prompt_default)).resolve()
    if not system_prompt_path.exists():
        raise FileNotFoundError(f"❌ system_prompt_path not found: {system_prompt_path}")
    system_content = load_text(system_prompt_path)

    # Java input
    if args.java_file:
        java_file = Path(args.java_file).resolve()
        if not java_file.exists():
            raise FileNotFoundError(f"❌ Java file not found: {java_file}")
        java_code = load_text(java_file)
    else:
        java_code = 'public class HelloWorld { public static void main(String[] a) { System.out.println("Hello"); }}'

    # -------------------------------
    # Initialize OpenAI-compatible client
    # -------------------------------
    client = OpenAI(api_key=api_key, base_url=base_url)

    # -------------------------------
    # Generate completions
    # -------------------------------
    prompt = f"Translate the following Java code to Rust:\n\n{java_code}\n\nRust code:"

    if use_chat:
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt}
        ]
        resp: ChatCompletion = client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=n_choices
        )
        outputs = [(choice.message.content or "").strip() for choice in resp.choices]
    else:
        resp = client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=n_choices
        )
        outputs = [(choice.text or "").strip() for choice in resp.choices]

    # -------------------------------
    # Save results
    # -------------------------------
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_tag = cfg.get("name", model).replace("/", "_")
    out_dir = (Path(args.outdir) / model_tag).resolve()
    ensure_dir(out_dir)
    out_path = out_dir / f"outputs_{ts}.jsonl"

    with out_path.open("w", encoding="utf-8") as f:
        for i, text in enumerate(outputs, 1):
            rec = {
                "idx": i,
                "model": model,
                "base_url": base_url,
                "use_chat": use_chat,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "prompt_style": "chat" if use_chat else "completion",
                "system_prompt_path": str(system_prompt_path),
                "java_code_len": len(java_code),
                "output": text,
                "timestamp": ts,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"✅ Saved {len(outputs)} completions → {out_path}")


if __name__ == "__main__":
    main()