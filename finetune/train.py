"""
Simple finetuning script for llm-jp-4-vl.

Usage:
    uv run python finetune/train.py \
        --data_path finetune/example_data.jsonl \
        --output_dir output/finetune \
        --num_train_epochs 3 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --learning_rate 2e-5

Multi-GPU:
    uv run torchrun --nproc_per_node=4 finetune/train.py \
        --data_path finetune/example_data.jsonl \
        --output_dir output/finetune
"""

from dataclasses import dataclass, field

import torch
from transformers import AutoModel, AutoProcessor, Trainer, TrainingArguments, HfArgumentParser

from dataset import VLFinetuneDataset, collate_fn


@dataclass
class ModelArguments:
    model_id: str = field(default="llm-jp/llm-jp-4-vl-9B-beta")
    freeze_vision: bool = field(default=True, metadata={"help": "Freeze the vision backbone"})
    freeze_projector: bool = field(default=False, metadata={"help": "Freeze the MLP projector"})
    max_dynamic_patch: int = field(default=12, metadata={"help": "Max dynamic patches per image"})


@dataclass
class DataArguments:
    data_path: str = field(metadata={"help": "Path to JSONL training data"})


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Load model
    model = AutoModel.from_pretrained(
        model_args.model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_flash_attn=True,
    )
    model.train()

    # Freeze components as requested
    if model_args.freeze_vision:
        for param in model.vision_backbone.parameters():
            param.requires_grad = False
    if model_args.freeze_projector:
        for param in model.mlp1.parameters():
            param.requires_grad = False

    # Enable gradient checkpointing to save memory
    if training_args.gradient_checkpointing:
        model.language_model.gradient_checkpointing_enable()

    # Load processor
    processor = AutoProcessor.from_pretrained(model_args.model_id, trust_remote_code=True)
    tokenizer = processor.tokenizer

    # Build dataset
    dataset = VLFinetuneDataset(
        data_path=data_args.data_path,
        tokenizer=tokenizer,
        processor=processor,
        max_dynamic_patch=model_args.max_dynamic_patch,
    )

    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
    )
    trainer.train()
    trainer.save_model()
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
