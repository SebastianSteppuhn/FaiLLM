#!/usr/bin/env python3
import argparse
import requests

def ollama_chat(prompt: str, base_url: str, model: str, temperature: float = 0.2, timeout: int = 120) -> str:
    """Send a single prompt to Ollama and return raw text."""
    base = base_url.rstrip("/")
    payload = {
        "model": model,
        "stream": False,
        "options": {"temperature": temperature},
        "messages": [{"role": "user", "content": prompt}],
    }
    r = requests.post(f"{base}/api/chat", json=payload, timeout=timeout)
    if r.status_code == 404:
        r = requests.post(
            f"{base}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": temperature}},
            timeout=timeout,
        )
    r.raise_for_status()
    data = r.json()
    return data.get("message", {}).get("content") or data.get("response", "") or ""

def main():
    ap = argparse.ArgumentParser(description="Send a prompt to Ollama and print raw output.")
    ap.add_argument("--base-url", default="http://localhost:11434", help="Ollama base URL")
    ap.add_argument("--model", default="llama3.1:8b", help="Ollama model name")
    ap.add_argument("--prompt-file", help="Path to prompt file")
    ap.add_argument("--prompt-text", help="Prompt string")
    ap.add_argument("--temperature", type=float, default=0.2, help="Temperature")
    ap.add_argument("--timeout", type=int, default=180, help="HTTP timeout seconds")
    args = ap.parse_args()

    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
    elif args.prompt_text:
        prompt = args.prompt_text.strip()
    else:
        print("Error: Provide --prompt-file or --prompt-text")
        return

    output = ollama_chat(prompt, args.base_url, args.model, args.temperature, args.timeout)
    print("=== RAW MODEL OUTPUT START ===")
    print(output)
    print("=== RAW MODEL OUTPUT END ===")

if __name__ == "__main__":
    main()
