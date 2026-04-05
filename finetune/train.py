"""
Finetune llm-jp-4-vl with PyTorch FSDP2.

Usage (single node, 8 GPUs):
    uv run torchrun --nproc_per_node=8 finetune/train.py \
        --data_path finetune/demo_data.jsonl \
        --output_dir output/finetune
"""

import argparse
import math
import os

import torch
import torch.distributed as dist
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModel, AutoProcessor

from dataset import VLFinetuneDataset, collate_fn


def parse_args():
    p = argparse.ArgumentParser()
    # Model
    p.add_argument("--model_id", default="llm-jp/llm-jp-4-vl-9B-beta")
    p.add_argument("--freeze_vision", action="store_true", default=True)
    p.add_argument("--no_freeze_vision", dest="freeze_vision", action="store_false")
    p.add_argument("--freeze_projector", action="store_true", default=False)
    p.add_argument("--max_dynamic_patch", type=int, default=12)
    # Data
    p.add_argument("--data_path", required=True)
    p.add_argument("--dataloader_num_workers", type=int, default=4)
    # Training
    p.add_argument("--output_dir", default="output/finetune")
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--save_strategy", default="epoch", choices=["epoch", "steps"])
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--cpu_offload", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def setup_distributed():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, local_rank


def apply_fsdp2(model, cpu_offload=False):
    """Apply FSDP2 fully_shard per encoder/decoder layer, then to the root model.

    Follows the same pattern as torchtitan's apply_fsdp: shard each transformer
    layer individually, then shard the root module.
    """
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    fsdp_config = {"mp_policy": mp_policy, "reshard_after_forward": True}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    # Shard each vision encoder layer
    for layer in model.vision_backbone.vision_model.encoder.layers:
        fully_shard(layer, **fsdp_config)

    # Shard each LLM decoder layer (GptOss, Llama, Qwen2, Qwen3, etc.)
    for layer in model.language_model.model.layers:
        fully_shard(layer, **fsdp_config)

    # Shard the projector
    fully_shard(model.mlp1, **fsdp_config)

    # Shard the root model
    fully_shard(model, **fsdp_config)
    return model


def save_checkpoint(model, output_dir, rank):
    """Save FSDP2 checkpoint using DCP (distributed, no full gather)."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_model_state_dict

    state_dict = get_model_state_dict(model)
    dcp.save(state_dict, checkpoint_id=output_dir)
    if rank == 0:
        print(f"Saved DCP checkpoint to {output_dir}")
    dist.barrier()


def forward_backward_step(model, batch, gradient_accumulation_steps):
    """Forward pass + loss scaling + backward. Follows torchtitan's pattern."""
    pred = model(
        input_ids=batch["input_ids"],
        pixel_values=batch["pixel_values"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        image_flags=batch["image_flags"],
    )
    loss = pred.loss / gradient_accumulation_steps
    del pred
    loss.backward()
    return loss


def main():
    args = parse_args()
    rank, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print("[rank 0] Loading model...")

    # Load model
    model = AutoModel.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_flash_attn=True,
    )

    if rank == 0:
        print("[rank 0] Model loaded. Freezing components...")

    # Freeze components
    if args.freeze_vision:
        for param in model.vision_backbone.parameters():
            param.requires_grad = False
    if args.freeze_projector:
        for param in model.mlp1.parameters():
            param.requires_grad = False

    # Activate gradient checkpointing
    if args.gradient_checkpointing:
        model.language_model.gradient_checkpointing_enable()
        model.vision_backbone.gradient_checkpointing_enable()
        model.config.use_cache = False

    if rank == 0:
        print("[rank 0] Applying FSDP2...")

    # Apply FSDP2
    model = apply_fsdp2(model, cpu_offload=args.cpu_offload)
    model.train()

    if rank == 0:
        print("[rank 0] FSDP2 applied. Loading processor and dataset...")

    # Processor and dataset
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer = processor.tokenizer

    dataset = VLFinetuneDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        processor=processor,
        max_dynamic_patch=args.max_dynamic_patch,
    )

    if rank == 0:
        print(f"[rank 0] Dataset loaded: {len(dataset)} samples. Building dataloader...")

    sampler = DistributedSampler(dataset, shuffle=True, seed=args.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_train_batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )

    # Optimizer
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # LR scheduler: linear warmup + cosine decay
    steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    total_steps = steps_per_epoch * args.num_train_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    warmup_scheduler = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=max(warmup_steps, 1))
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup_steps, 1))
    scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

    if rank == 0:
        print(
            f"Training: {len(dataset)} samples, {total_steps} steps "
            f"(warmup {warmup_steps}), grad accum {args.gradient_accumulation_steps}"
        )

    # Training loop (follows torchtitan's train_step pattern)
    global_step = 0
    for epoch in range(args.num_train_epochs):
        sampler.set_epoch(epoch)
        data_iterator = iter(dataloader)
        microbatch_idx = 0

        for batch in data_iterator:
            if microbatch_idx == 0:
                optimizer.zero_grad()

            # Move batch to device
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            batch["pixel_values"] = batch["pixel_values"].to(dtype=torch.bfloat16)

            # Forward + backward (loss is scaled by grad accum steps)
            loss = forward_backward_step(model, batch, args.gradient_accumulation_steps)
            microbatch_idx += 1

            # Accumulate gradients, then step
            if microbatch_idx == args.gradient_accumulation_steps:
                if args.max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()

                global_step += 1
                microbatch_idx = 0

                if global_step % args.logging_steps == 0 and rank == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    # loss was already divided by grad_accum_steps, multiply back for logging
                    print(
                        f"epoch={epoch+1}/{args.num_train_epochs} "
                        f"step={global_step}/{total_steps} "
                        f"loss={loss.item() * args.gradient_accumulation_steps:.4f} "
                        f"lr={lr:.2e}"
                    )

                if args.save_strategy == "steps" and global_step % args.save_steps == 0:
                    save_checkpoint(model, f"{args.output_dir}/step-{global_step}", rank)

        # Handle remaining microbatches at end of epoch
        if microbatch_idx > 0:
            if args.max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            global_step += 1
            microbatch_idx = 0

        if args.save_strategy == "epoch":
            save_checkpoint(model, f"{args.output_dir}/epoch-{epoch+1}", rank)

    # Save final model
    save_checkpoint(model, f"{args.output_dir}/final", rank)

    if rank == 0:
        print("Training completed")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
