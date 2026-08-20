import re

import torch
from transformers import AutoProcessor, AutoModel

# Works with both models:
#   - "llm-jp/llm-jp-4-vl-9b"       (reasoning model; may emit an analysis
#                                     channel before the final answer)
#   - "llm-jp/llm-jp-4-vl-9B-beta"  (non-reasoning; final answer only)
model_id = "llm-jp/llm-jp-4-vl-9b"
is_reasoning = "beta" not in model_id  # the beta model has no reasoning channel

# load model
model = (
    AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_flash_attn=True,
    )
    .eval()
    .cuda()
)

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# The final answer follows this marker; a reasoning model may first emit
# "<|channel|>analysis<|message|>...<|end|>". decode() can put spaces around the
# marker tokens, so match them space-tolerantly.
_FINAL_RE = re.compile(r"<\|channel\|>\s*final\s*<\|message\|>")


def generate(messages, max_new_tokens=1024, temperature=0.0, reasoning_effort="medium"):
    # reasoning_effort ("low" -> direct answer / "medium" / "high") is only
    # accepted by the reasoning model's chat template.
    template_kwargs = {"reasoning_effort": reasoning_effort} if is_reasoning else {}
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        **template_kwargs,
    ).to(model.device)

    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model.dtype)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
    )

    text = processor.decode(outputs[0], skip_special_tokens=False)
    # Keep only the text after the LAST final-channel marker, so any analysis
    # (chain-of-thought) is dropped. Works for the non-reasoning model too,
    # whose output is just "<|channel|>final<|message|>{answer}<|return|>".
    matches = list(_FINAL_RE.finditer(text))
    if matches:
        text = text[matches[-1].end():]
    text = re.sub(r"<\|[^|]*\|>", "", text).replace(processor.tokenizer.eos_token, "")
    return text.strip()


# -----------------------
# 1. Text-only
# -----------------------
messages = [
    {
        "role": "user",
        "content": [{"type": "text", "text": "富士山について簡潔に説明してください。"}],
    }
]
print(generate(messages))
# 富士山は、日本最高峰の山で、標高3,776メートルです。静岡県と山梨県にまたがっており、世界遺産にも登録されています。

# -----------------------
# 2. Single image
# -----------------------
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "assets/kaonashi.jpg"},
            {"type": "text", "text": "このキャラクターの名前は何ですか？"},
        ],
    }
]
print(generate(messages))
# カオナシ

# -----------------------
# 3. Multi-image
# -----------------------
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "assets/Shiba_inu.jpg"},
            {"type": "image", "image": "assets/yesoensis.jpg"},
            {"type": "text", "text": "それぞれの動物の名前を教えてください。"},
        ],
    }
]
print(generate(messages))
# 柴犬と鹿です。

# -----------------------
# 4. Multi-turn example
# -----------------------
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "assets/kaonashi.jpg"},
            {
                "type": "text",
                "text": "このキャラクターが登場する映画のタイトルは何ですか？",
            },
        ],
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "千と千尋の神隠し"}],
    },
    {
        "role": "user",
        "content": [{"type": "text", "text": "監督は誰ですか？"}],
    },
]

print(generate(messages))
# 宮崎駿
