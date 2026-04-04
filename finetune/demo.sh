#!/bin/bash
# Demo: finetune llm-jp-4-vl on a small dataset

uv run torchrun --nproc_per_node=8 finetune/train.py \
    --data_path finetune/demo_data.jsonl \
    --output_dir output/demo \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --save_strategy epoch \
    --gradient_checkpointing \
    --dataloader_num_workers 4
