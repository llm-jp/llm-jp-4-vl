import torch
from transformers import AutoProcessor, AutoModel

model_id = "llm-jp/llm-jp-4-vl-9B-beta"

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


def generate(messages, max_new_tokens=256, temperature=0.0):
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
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
    text = text.replace("<|channel|>final<|message|>", "")
    text = text.replace("<|return|>", "")
    text = text.replace(processor.tokenizer.eos_token, "")
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
