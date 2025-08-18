import argparse
import json
import os
from datetime import datetime

import torch
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from torch_xla.distributed.parallel_loader import MpDeviceLoader

from config import get_config
from data_utils import set_all_seeds, load_sst2_tokenized, make_train_eval_loaders
from load_baseline_fixed import load_warmed_baseline

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", type=str, default="v3_8_gradual")
    ap.add_argument("--baseline_path", type=str, default="baseline_model_seed42/pytorch_model.bin")
    ap.add_argument("--output_root", type=str, default="checkpoints_gradual")
    return ap.parse_args()

def mesh_mean(name, value):
    return xm.mesh_reduce(name, value, lambda xs: sum(xs) / len(xs))

def train_worker(rank, flags):
    cfg = flags["cfg"]
    baseline_path = flags["baseline_path"]
    run_root = flags["run_root"]

    # Derive world size robustly under PJRT
    world_size = len(xm.get_xla_supported_devices())

    # Repro
    set_all_seeds(cfg.seed, rank)

    device = xm.xla_device()
    xm.master_print(f"[rank {rank}] Device: {device} | world_size={world_size}")

    # Data & tokenizer
    tokenized, tokenizer = load_sst2_tokenized(cfg.model_name_or_path, cfg.max_seq_length)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name_or_path, num_labels=2
    )
    # Load warmed baseline
    model, missing, unexpected = load_warmed_baseline(model, baseline_path)
    xm.master_print(f"[baseline] missing={len(missing)} unexpected={len(unexpected)}")

    model.to(device)

    # Optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    # Build loaders for epoch 0 to discover steps_per_epoch
    train_loader, eval_loader, global_bs, steps_per_epoch = make_train_eval_loaders(
        tokenized, cfg.train_samples_per_epoch, cfg.per_core_train_batch_size,
        cfg.per_core_eval_batch_size, cfg.seed, epoch=0, rank=rank, world_size=world_size
    )
    total_training_steps = steps_per_epoch * cfg.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=total_training_steps
    )

    # Parallel loaders
    pl_train = MpDeviceLoader(train_loader, device)
    pl_eval  = MpDeviceLoader(eval_loader, device)

    # Save tokenizer once (master)
    if cfg.save_tokenizer_once and xm.is_master_ordinal():
        os.makedirs(run_root, exist_ok=True)
        tokenizer.save_pretrained(run_root)

    # Training loop
    for epoch in range(cfg.num_epochs):
        # Rebuild train loader each epoch to advance sampler seed
        train_loader, eval_loader, _, _ = make_train_eval_loaders(
            tokenized, cfg.train_samples_per_epoch, cfg.per_core_train_batch_size,
            cfg.per_core_eval_batch_size, cfg.seed, epoch=epoch, rank=rank, world_size=world_size
        )
        pl_train = MpDeviceLoader(train_loader, device)
        pl_eval  = MpDeviceLoader(eval_loader, device)

        model.train()
        running_loss = 0.0
        steps_done = 0

        for step, batch in enumerate(pl_train, start=1):
            inputs = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "token_type_ids": batch.get("token_type_ids", None),
                "labels": batch["labels"],
            }
            if inputs["token_type_ids"] is None:
                inputs.pop("token_type_ids")

            outputs = model(**inputs)
            loss = outputs.loss

            loss.backward()
            xm.optimizer_step(optimizer)        # XLA-safe optimizer step
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            xm.mark_step()                       # Execute the compiled graph

            running_loss += loss.item()
            steps_done += 1

            if (step % 50 == 0) and xm.is_master_ordinal():
                xm.master_print(f"Epoch {epoch} | Step {step}/{steps_per_epoch} | loss={running_loss/steps_done:.4f}")

        # End-of-epoch eval (single sync via mesh_reduce)
        model.eval()
        correct_local = 0
        total_local = 0
        with torch.no_grad():
            for batch in pl_eval:
                inputs = {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"],
                    "token_type_ids": batch.get("token_type_ids", None),
                }
                if inputs["token_type_ids"] is None:
                    inputs.pop("token_type_ids")
                labels = batch["labels"]

                logits = model(**inputs).logits
                preds = logits.argmax(dim=-1)
                correct_local += (preds == labels).sum().item()
                total_local   += preds.size(0)

        acc = mesh_mean("eval_acc", correct_local / max(1, total_local))
        train_loss_mean = mesh_mean("train_loss", running_loss / max(1, steps_done))

        if xm.is_master_ordinal():
            xm.master_print(f"[Epoch {epoch}] acc={acc:.4f} | train_loss={train_loss_mean:.4f}")

            ckpt_dir = os.path.join(run_root, f"checkpoint_epoch_{epoch}")
            os.makedirs(ckpt_dir, exist_ok=True)
            xm.save(model.state_dict(), os.path.join(ckpt_dir, "checkpoint.pt"))

            info = {
                "epoch": epoch,
                "train_loss": float(train_loss_mean),
                "eval_accuracy": float(acc),
                "global_batch_size": cfg.per_core_train_batch_size * world_size,
                "per_core_batch_size": cfg.per_core_train_batch_size,
                "steps_per_epoch": steps_per_epoch,
                "learning_rate": cfg.learning_rate,
                "warmup_steps": cfg.warmup_steps,
                "weight_decay": cfg.weight_decay,
                "max_seq_length": cfg.max_seq_length,
                "seed": cfg.seed,
                "baseline_path": baseline_path,
            }
            with open(os.path.join(ckpt_dir, "checkpoint_info.json"), "w") as f:
                json.dump(info, f, indent=2)

    if xm.is_master_ordinal():
        xm.master_print("Training complete.")

def main():
    args = parse_args()
    cfg = get_config(args.profile)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = f"{args.output_root}_{ts}"
    flags = {"cfg": cfg, "baseline_path": args.baseline_path, "run_root": run_root}

    # v3-8 has 8 cores; torch_xla 2.7 (PJRT) no longer provides xm.xrt_world_size()
    xmp.spawn(train_worker, args=(flags,), nprocs=8)

if __name__ == "__main__":
    main()
