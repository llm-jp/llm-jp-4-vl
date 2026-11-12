"""Minimal client for the SGLang server started by launch.sh (OpenAI-compatible /v1/chat/completions).

    python inference/sglang/client.py --image assets/Shiba_inu.jpg --prompt "この画像を説明してください。"
    python inference/sglang/client.py --model llm-jp/llm-jp-4.1-vl-9b-v2-mix-30k --date 2026-10-01 --effort high

Standard library only. The model answers in the Harmony format: an optional analysis channel (reasoning), then the
final channel. The request keeps the special tokens (skip_special_tokens=false) so the two can be split here.
"""

import argparse
import base64
import json
import mimetypes
import re
import urllib.request

FINAL = re.compile(r"<\|channel\|>\s*final\s*<\|message\|>")
ANALYSIS = re.compile(r"<\|channel\|>\s*analysis\s*<\|message\|>(.*?)<\|end\|>", re.S)


def image_part(path: str) -> dict:
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


def split_harmony(text: str) -> tuple[str, str]:
    """(final answer, reasoning) of a completion decoded with its special tokens."""
    reasoning = "\n".join(m.strip() for m in ANALYSIS.findall(text))
    m = list(FINAL.finditer(text))
    final = text[m[-1].end():] if m else text
    return re.sub(r"<\|[^|]*\|>", "", final).strip(), reasoning


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:30000/v1")
    ap.add_argument("--model", default="llm-jp/llm-jp-4-vl-9b")
    ap.add_argument("--image", action="append", default=[], help="image file (repeatable)")
    ap.add_argument("--prompt", default="この画像を説明してください。")
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high"], help="Harmony reasoning effort")
    ap.add_argument("--date", help="Harmony 'Current date' line (llm-jp-4.1 templates; default: today)")
    ap.add_argument("--max-tokens", type=int, default=8192)
    args = ap.parse_args()

    content = [image_part(p) for p in args.image] + [{"type": "text", "text": args.prompt}]
    kwargs = {"reasoning_effort": args.effort}
    if args.date:
        kwargs["conversation_start_date"] = args.date
    body = {"model": args.model, "messages": [{"role": "user", "content": content}], "temperature": 0.0,
            "max_tokens": args.max_tokens, "skip_special_tokens": False, "chat_template_kwargs": kwargs}
    req = urllib.request.Request(f"{args.base_url}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        out = json.loads(r.read())
    final, reasoning = split_harmony(out["choices"][0]["message"]["content"] or "")
    if reasoning:
        print(f"[reasoning]\n{reasoning}\n")
    print(f"[answer]\n{final}")
    print(f"\n[usage] {out.get('usage')}")


if __name__ == "__main__":
    main()
