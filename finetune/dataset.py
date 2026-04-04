import json
import logging
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile, PngImagePlugin
from torch.utils.data import Dataset
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
PngImagePlugin.MAX_TEXT_CHUNK = 1024 * 2**20

logger = logging.getLogger(__name__)

# Special tokens (must match the model's vocabulary)
IMG_CONTEXT_TOKEN = "<|image_pad|>"
IMAGE_START = "<|image_start|>"
IMAGE_END = "<|image_end|>"
DEFAULT_SYSTEM_MESSAGE = "You are LLM-jp-VL, a Multimodal LLM trained by LLM-jp."
HARMONY_END = "<|end|>"
HARMONY_RETURN = "<|return|>"


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=512, use_thumbnail=False):
    """Split an image into patches based on aspect ratio."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    best_ratio = (1, 1)
    best_diff = float("inf")
    area = orig_width * orig_height
    for ratio in target_ratios:
        diff = abs(aspect_ratio - ratio[0] / ratio[1])
        if diff < best_diff:
            best_diff = diff
            best_ratio = ratio
        elif diff == best_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best_ratio = ratio

    target_w = image_size * best_ratio[0]
    target_h = image_size * best_ratio[1]
    resized = image.resize((target_w, target_h))

    patches = []
    for i in range(best_ratio[0] * best_ratio[1]):
        x = (i % (target_w // image_size)) * image_size
        y = (i // (target_w // image_size)) * image_size
        patches.append(resized.crop((x, y, x + image_size, y + image_size)))

    if use_thumbnail and len(patches) != 1:
        patches.append(image.resize((image_size, image_size)))
    return patches


def preprocess_conversation(tokenizer, conversations, num_image_token_list, text_only=False):
    """Tokenize conversations and create labels with proper masking."""
    if conversations[0]["from"] == "system":
        system_prompt = conversations[0]["value"]
        conversations = conversations[1:]
    else:
        system_prompt = DEFAULT_SYSTEM_MESSAGE

    # Replace <image> placeholders with image tokens
    if not text_only:
        current_image_idx = 0
        new_conversations = []
        for conv in conversations:
            if conv["from"] == "human":
                image_cnt = conv["value"].count("<image>")
                for _ in range(image_cnt):
                    if current_image_idx >= len(num_image_token_list):
                        break
                    image_tokens = f"{IMAGE_START}{IMG_CONTEXT_TOKEN * num_image_token_list[current_image_idx]}{IMAGE_END}"
                    conv["value"] = conv["value"].replace("<image>", image_tokens, 1)
                    current_image_idx += 1
            new_conversations.append(conv)
        conversations = new_conversations

    # Build formatted text for each turn
    batches, roles = [], []
    if system_prompt is not None:
        batches.append(f"<|start|>system<|message|>{system_prompt}<|end|>")
        roles.append("system")
    for i, conv in enumerate(conversations):
        if conv["from"] == "human":
            batches.append(f"<|start|>user<|message|>{conv['value']}<|end|>")
            roles.append("human")
        elif conv["from"] == "gpt":
            is_last = i == len(conversations) - 1
            eos = "<|return|>" if is_last else "<|end|>"
            batches.append(f"<|start|>assistant<|channel|>final<|message|>{conv['value']}{eos}")
            roles.append("gpt" if is_last else "gpt_intermediate")

    # Tokenize each turn
    input_ids_list = tokenizer(
        batches, return_tensors="np", padding=False,
        max_length=tokenizer.model_max_length, truncation=False, add_special_tokens=False,
    ).input_ids

    ignore_ids = tokenizer("<|start|>assistant", return_tensors="np", add_special_tokens=False).input_ids[0]
    ignore_len = ignore_ids.shape[0]
    end_token_id = tokenizer.convert_tokens_to_ids(HARMONY_END)
    return_token_id = tokenizer.convert_tokens_to_ids(HARMONY_RETURN)

    final_input_ids, final_targets = [], []
    for role, ids in zip(roles, input_ids_list):
        final_input_ids.append(ids)
        if role in ("system", "human"):
            final_targets.append(np.full(ids.shape, IGNORE_TOKEN_ID))
        elif role in ("gpt", "gpt_intermediate"):
            target = ids.copy()
            target[:ignore_len] = IGNORE_TOKEN_ID
            if role == "gpt_intermediate" and ids[-1] == end_token_id:
                target[-1] = return_token_id
            final_targets.append(target)

    input_ids = torch.tensor(np.concatenate(final_input_ids))[:tokenizer.model_max_length]
    labels = torch.tensor(np.concatenate(final_targets))[:tokenizer.model_max_length]
    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


class VLFinetuneDataset(Dataset):
    """Dataset for vision-language finetuning.

    Expected JSONL format:
    {"image": "path/to/image.jpg", "conversations": [{"from": "human", "value": "Describe <image>"}, {"from": "gpt", "value": "..."}]}

    For multi-image:
    {"image": ["path1.jpg", "path2.jpg"], "conversations": [...]}

    For text-only:
    {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
    """

    def __init__(self, data_path, tokenizer, processor, max_dynamic_patch=6, image_size=512, patch_size=16):
        self.data = []
        self.data_dir = Path(data_path).parent
        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(json.loads(line))

        self.tokenizer = tokenizer
        self.processor = processor  # the model's AutoProcessor (for vision)
        self.image_size = image_size
        self.patch_size = patch_size
        self.max_dynamic_patch = max_dynamic_patch
        self.num_image_token = int((image_size // patch_size) ** 2 * (0.5**2))

    def __len__(self):
        return len(self.data)

    def _resolve_image_path(self, path):
        p = Path(path)
        if p.is_absolute():
            return str(p)
        return str(self.data_dir / p)

    def _load_and_patch_image(self, image_path, max_num=None):
        if max_num is None:
            max_num = self.max_dynamic_patch
        image = Image.open(self._resolve_image_path(image_path)).convert("RGB")
        patches = dynamic_preprocess(
            image, min_num=1, max_num=max_num,
            image_size=self.image_size, use_thumbnail=True,
        )
        pixel_values = self.processor.image_processor(images=patches, return_tensors="pt").pixel_values
        return pixel_values

    def __getitem__(self, idx):
        item = deepcopy(self.data[idx])
        image_paths = item.get("image", None)

        # Determine if this is text-only, single-image, or multi-image
        if image_paths is None or image_paths == []:
            return self._process_text_only(item)
        if isinstance(image_paths, str):
            return self._process_single_image(item, image_paths)
        return self._process_multi_image(item, image_paths)

    def _process_text_only(self, item):
        ret = preprocess_conversation(self.tokenizer, item["conversations"], [], text_only=True)
        position_ids = ret["attention_mask"].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret["attention_mask"] == 0, 1)
        return {
            "input_ids": ret["input_ids"],
            "labels": ret["labels"],
            "attention_mask": ret["attention_mask"],
            "position_ids": position_ids,
            "pixel_values": torch.zeros(0, 3, self.image_size, self.image_size),
            "image_flags": torch.tensor([], dtype=torch.long),
        }

    def _count_text_tokens(self, conversations):
        """Count text tokens excluding <image> placeholders."""
        convs = list(conversations)
        if convs[0]["from"] == "system":
            system_prompt = convs[0]["value"]
            convs = convs[1:]
        else:
            system_prompt = DEFAULT_SYSTEM_MESSAGE

        parts = [f"<|start|>system<|message|>{system_prompt}<|end|>"]
        for i, conv in enumerate(convs):
            value = conv["value"].replace("<image>", "")
            if conv["from"] == "human":
                parts.append(f"<|start|>user<|message|>{value}<|end|>")
            elif conv["from"] == "gpt":
                is_last = i == len(convs) - 1
                eos = "<|return|>" if is_last else "<|end|>"
                parts.append(f"<|start|>assistant<|channel|>final<|message|>{value}{eos}")
        return len(self.tokenizer.encode("".join(parts), add_special_tokens=False))

    def _compute_max_patches(self, conversations, num_images=1):
        """Compute max patches per image so the sequence fits in model_max_length."""
        text_tokens = self._count_text_tokens(conversations)
        image_budget = self.tokenizer.model_max_length - text_tokens
        # Each image uses (num_patches) * num_image_token + 2 tokens (image_start + image_end)
        # With thumbnail: num_patches = grid_patches + 1, so max_grid = budget / num_images / num_image_token - 1 - 2/num_image_token
        max_num = (image_budget // num_images - 2) // self.num_image_token - 1
        return max(1, min(self.max_dynamic_patch, max_num))

    def _process_single_image(self, item, image_path):
        conversations = item["conversations"]
        first_idx = 1 if conversations[0]["from"] == "system" else 0
        if "<image>" not in conversations[first_idx]["value"]:
            conversations[first_idx]["value"] = "<image>" + conversations[first_idx]["value"]

        max_patches = self._compute_max_patches(conversations)
        pixel_values = self._load_and_patch_image(image_path, max_num=max_patches)
        num_patches = pixel_values.size(0)

        ret = preprocess_conversation(
            self.tokenizer, conversations,
            [self.num_image_token * num_patches],
        )
        position_ids = ret["attention_mask"].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret["attention_mask"] == 0, 1)
        return {
            "input_ids": ret["input_ids"],
            "labels": ret["labels"],
            "attention_mask": ret["attention_mask"],
            "position_ids": position_ids,
            "pixel_values": pixel_values,
            "image_flags": torch.tensor([1] * num_patches, dtype=torch.long),
        }

    def _process_multi_image(self, item, image_paths):
        conversations = item["conversations"]
        first_idx = 1 if conversations[0]["from"] == "system" else 0
        if "<image>" not in conversations[first_idx]["value"]:
            conversations[first_idx]["value"] = "<image>" * len(image_paths) + conversations[first_idx]["value"]

        max_patches = self._compute_max_patches(conversations, num_images=len(image_paths))

        all_pixel_values = []
        num_tiles = []
        for path in image_paths:
            pv = self._load_and_patch_image(path, max_num=max_patches)
            all_pixel_values.append(pv)
            num_tiles.append(pv.size(0))

        pixel_values = torch.cat(all_pixel_values, dim=0)
        num_image_tokens = [self.num_image_token * nt for nt in num_tiles]

        ret = preprocess_conversation(self.tokenizer, conversations, num_image_tokens)
        position_ids = ret["attention_mask"].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret["attention_mask"] == 0, 1)
        return {
            "input_ids": ret["input_ids"],
            "labels": ret["labels"],
            "attention_mask": ret["attention_mask"],
            "position_ids": position_ids,
            "pixel_values": pixel_values,
            "image_flags": torch.tensor([1] * pixel_values.size(0), dtype=torch.long),
        }


def collate_fn(batch):
    """Pad sequences to the same length within a batch."""
    max_len = max(item["input_ids"].size(0) for item in batch)

    input_ids = []
    labels = []
    attention_mask = []
    position_ids = []
    pixel_values = []
    image_flags = []

    for item in batch:
        seq_len = item["input_ids"].size(0)
        pad_len = max_len - seq_len

        input_ids.append(torch.cat([item["input_ids"], torch.zeros(pad_len, dtype=torch.long)]))
        labels.append(torch.cat([item["labels"], torch.full((pad_len,), IGNORE_TOKEN_ID, dtype=torch.long)]))
        attention_mask.append(torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.bool)]))
        position_ids.append(torch.cat([item["position_ids"], torch.zeros(pad_len, dtype=torch.long)]))
        pixel_values.append(item["pixel_values"])
        image_flags.append(item["image_flags"])

    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask),
        "position_ids": torch.stack(position_ids),
        "pixel_values": torch.cat(pixel_values, dim=0),
        "image_flags": torch.cat(image_flags, dim=0).unsqueeze(-1),
    }
