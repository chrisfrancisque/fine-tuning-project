import math
import os
import numpy as np
from typing import Tuple
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, Subset, DistributedSampler
from transformers import AutoTokenizer

def set_all_seeds(base_seed: int, rank: int = 0):
    import random
    random.seed(base_seed + rank)
    np.random.seed(base_seed + rank)
    torch.manual_seed(base_seed + rank)
    os.environ["PYTHONHASHSEED"] = str(base_seed + rank)

def load_sst2_tokenized(model_name_or_path: str, max_len: int):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    raw = load_dataset("glue", "sst2")

    def _tok(batch):
        out = tokenizer(batch["sentence"], truncation=True, padding="max_length", max_length=max_len)
        out["labels"] = batch["label"]
        return out

    tokenized = raw.map(_tok, batched=True, remove_columns=["sentence", "label", "idx"])
    tokenized.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])
    return tokenized, tokenizer

def make_train_eval_loaders(
    tokenized_ds,
    train_samples_per_epoch: int,
    per_core_train_bs: int,
    per_core_eval_bs: int,
    seed: int,
    epoch: int,
    rank: int,
    world_size: int,
) -> Tuple[DataLoader, DataLoader, int, int]:
    """
    Returns train_loader, eval_loader, global_batch_size, steps_per_epoch.
    Uses a deterministic subset of the train set (size=train_samples_per_epoch) and a DistributedSampler.
    drop_last=True to keep shapes stable across TPU cores.
    """
    train_full = tokenized_ds["train"]
    eval_full  = tokenized_ds["validation"]

    # deterministic subset indices (same across epochs; shuffle happens in the sampler)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(train_full))
    subset_indices = perm[:train_samples_per_epoch]
    train_subset = Subset(train_full, subset_indices.tolist())

    # distributed samplers
    train_sampler = DistributedSampler(
        train_subset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed + epoch, drop_last=True
    )
    eval_sampler = DistributedSampler(
        eval_full, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
    )

    # DataLoader generator for deterministic shuffling worker-side
    g = torch.Generator()
    g.manual_seed(seed + rank + epoch * 31)

    train_loader = DataLoader(
        train_subset,
        batch_size=per_core_train_bs,
        sampler=train_sampler,
        num_workers=0,
        drop_last=True,
        pin_memory=False,
        generator=g,
    )
    eval_loader = DataLoader(
        eval_full,
        batch_size=per_core_eval_bs,
        sampler=eval_sampler,
        num_workers=0,
        drop_last=False,
        pin_memory=False,
    )

    global_bs = per_core_train_bs * world_size
    # effective examples per epoch after drop_last
    eff_examples = (len(train_subset) // global_bs) * global_bs
    steps_per_epoch = eff_examples // global_bs
    return train_loader, eval_loader, global_bs, steps_per_epoch
