# Finetune LLM-jp-4-VL

Simple finetuning framework for [llm-jp-4-vl](https://huggingface.co/llm-jp/llm-jp-4-vl-9B-beta) based on HuggingFace Trainer.

## Requirements

Install dependencies from the project root:

```bash
uv sync
```

The following packages are required (already included in `pyproject.toml`):

- `torch==2.8.0`
- `transformers==4.57.0`
- `flash-attn==2.8.3`
- `pillow==11.3.0`

## Dataset Format

Training data should be a JSONL file where each line is a JSON object. Three formats are supported:

### Single Image

```json
{
    "image": "path/to/image.jpg",
    "conversations": [
        {"from": "human", "value": "<image>\nこの画像を説明してください。"},
        {"from": "gpt", "value": "これは..."}
    ]
}
```

### Multi-Image

```json
{
    "image": ["path/to/image1.jpg", "path/to/image2.jpg"],
    "conversations": [
        {"from": "human", "value": "<image><image>\nそれぞれの画像の違いを教えてください。"},
        {"from": "gpt", "value": "1枚目は...、2枚目は..."}
    ]
}
```

### Text-Only

```json
{
    "conversations": [
        {"from": "human", "value": "富士山について説明してください。"},
        {"from": "gpt", "value": "富士山は日本最高峰の山で..."}
    ]
}
```

### Multi-Turn

```json
{
    "image": "path/to/image.jpg",
    "conversations": [
        {"from": "human", "value": "<image>\nこれは何ですか？"},
        {"from": "gpt", "value": "柴犬です。"},
        {"from": "human", "value": "この犬種の特徴は？"},
        {"from": "gpt", "value": "柴犬は日本原産の犬種で..."}
    ]
}
```

### Notes

- Image paths can be absolute or relative to the JSONL file's directory.
- Use `<image>` tag in the human message to indicate where the image should be placed. If omitted, `<image>` is automatically prepended.
- One `<image>` tag corresponds to one image in the `image` field.
- An optional system message can be added as the first conversation turn with `{"from": "system", "value": "..."}`.

## Usage

### Quick Start (Demo)

Run the demo script with the provided sample data:

```bash
bash finetune/demo.sh
```

### Single GPU

```bash
uv run python finetune/train.py \
    --data_path path/to/data.jsonl \
    --output_dir output/my_finetune \
    --bf16 true \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_strategy epoch \
    --gradient_checkpointing true \
    --dataloader_num_workers 4
```

### Multi-GPU (torchrun)

```bash
uv run torchrun --nproc_per_node=4 finetune/train.py \
    --data_path path/to/data.jsonl \
    --output_dir output/my_finetune \
    --bf16 true \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_strategy epoch \
    --gradient_checkpointing true \
    --dataloader_num_workers 4
```

## Key Parameters

### Model

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_id` | `llm-jp/llm-jp-4-vl-9B-beta` | HuggingFace model ID or local path |
| `--freeze_vision` | `true` | Freeze the vision backbone (SiglipV2) |
| `--freeze_projector` | `false` | Freeze the MLP projector |
| `--max_dynamic_patch` | `12` | Max number of dynamic patches per image |

### Training (HuggingFace TrainingArguments)

| Parameter | Suggested | Description |
|-----------|-----------|-------------|
| `--learning_rate` | `1e-5` ~ `2e-5` | Learning rate |
| `--num_train_epochs` | `1` ~ `3` | Number of training epochs |
| `--per_device_train_batch_size` | `1` | Batch size per GPU |
| `--gradient_accumulation_steps` | `8` | Gradient accumulation steps |
| `--gradient_checkpointing` | `true` | Reduce memory usage |
| `--bf16` | `true` | Use bfloat16 mixed precision |
| `--warmup_ratio` | `0.03` | Warmup ratio |
| `--lr_scheduler_type` | `cosine` | Learning rate scheduler |
| `--save_strategy` | `epoch` | Checkpoint saving strategy |
| `--deepspeed` | - | Path to DeepSpeed config for ZeRO |

## File Structure

```
finetune/
├── README.md          # This file
├── train.py           # Main training script
├── dataset.py         # Dataset and data collator
├── demo.sh            # Demo launch script
└── demo_data.jsonl    # Sample data using assets/ images
```
