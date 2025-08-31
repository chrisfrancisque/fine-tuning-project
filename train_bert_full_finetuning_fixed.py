"""
BERT Full Fine-tuning with Fixed TPU Issues
"""

import os
import torch
import json
import time
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup
)
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix

from config import config
from data_utils import load_and_prepare_dataset, create_dataloaders

# Import TPU libraries
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

def load_warmed_baseline(baseline_path='warmed_baseline_60pct'):
    """Load the pre-created warmed baseline model"""
    
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f"Baseline not found at {baseline_path}")
    
    print(f"Loading warmed baseline from {baseline_path}...")
    
    model = AutoModelForSequenceClassification.from_pretrained(baseline_path)
    tokenizer = AutoTokenizer.from_pretrained(baseline_path)
    
    with open(os.path.join(baseline_path, 'baseline_info.json'), 'r') as f:
        baseline_info = json.load(f)
    
    print(f"✓ Loaded baseline with {baseline_info['accuracy']:.2%} accuracy")
    
    for param in model.parameters():
        param.requires_grad = True
    
    return model, tokenizer, baseline_info['accuracy']

def train_bert_with_checkpoints(index=None):
    """Main training function with checkpoint saving every 100 steps"""
    
    # Setup device
    device = xm.xla_device()
    is_master = xm.is_master_ordinal()
    
    # Create output directory
    if is_master:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results_100step_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        print(f"Results will be saved to: {output_dir}")
    else:
        output_dir = None
    
    # Synchronize output_dir
    output_dir = xm.mesh_reduce('broadcast', output_dir, lambda x: x[0] if x else None)
    
    # Load baseline
    if is_master:
        print("\n" + "="*60)
        print("STARTING BERT FULL FINE-TUNING ON TPU")
        print("="*60)
    
    model, tokenizer, baseline_accuracy = load_warmed_baseline('warmed_baseline_60pct')
    model.to(device)
    
    # Load data
    train_dataset, eval_dataset, _ = load_and_prepare_dataset(config)
    train_dataloader, eval_dataloader = create_dataloaders(train_dataset, eval_dataset, config)
    
    # Wrap for TPU
    train_dataloader = pl.MpDeviceLoader(train_dataloader, device)
    
    # FIXED CHECKPOINT INTERVAL
    steps_per_checkpoint = 100  # Every 100 steps as requested
    total_steps = len(train_dataloader) * config.num_train_epochs
    
    if is_master:
        print(f"\nTraining Configuration:")
        print(f"  Baseline accuracy: {baseline_accuracy:.2%}")
        print(f"  Total training steps: {total_steps}")
        print(f"  Checkpoint interval: every {steps_per_checkpoint} steps")
        print(f"  Expected checkpoints: ~{total_steps // steps_per_checkpoint}")
        print(f"  Batch size: {config.total_train_batch_size}")
        print(f"  Learning rate: {config.learning_rate}")
        print("="*60)
    
    # Setup optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training tracking
    global_step = 0
    checkpoint_count = 0
    all_losses = []
    checkpoint_info = []
    
    # Training loop
    model.train()
    
    for epoch in range(config.num_train_epochs):
        epoch_start = time.time()
        epoch_loss = 0
        num_batches = 0
        
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            
            loss.backward()
            xm.optimizer_step(optimizer)
            scheduler.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            all_losses.append(loss.item())
            num_batches += 1
            global_step += 1
            
            # Log progress
            if step % 50 == 0 and is_master:
                xm.master_print(f"Epoch {epoch}, Step {step}/{len(train_dataloader)}, "
                               f"Loss: {loss.item():.4f}, Global Step: {global_step}")
            
            # Save checkpoint at intervals (WITHOUT EVALUATION to avoid crash)
            if global_step % steps_per_checkpoint == 0 and is_master:
                print(f"\nSaving checkpoint at step {global_step}...")
                
                checkpoint_name = f'checkpoint_{checkpoint_count:02d}_step_{global_step}'
                checkpoint_dir = os.path.join(output_dir, checkpoint_name)
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                # Save model to CPU to avoid TPU issues
                model_cpu = model.cpu()
                model_cpu.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
                model.to(device)
                
                # Save basic metrics without evaluation
                metrics = {
                    'step': global_step,
                    'epoch': epoch,
                    'checkpoint': checkpoint_count,
                    'train_loss': float(epoch_loss / max(1, num_batches)),
                    'baseline_accuracy': float(baseline_accuracy)
                }
                
                with open(os.path.join(checkpoint_dir, 'checkpoint_info.json'), 'w') as f:
                    json.dump(metrics, f, indent=2)
                
                checkpoint_info.append(metrics)
                checkpoint_count += 1
                
                print(f"✓ Checkpoint {checkpoint_count} saved at step {global_step}")
                
                # Continue training
                model.train()
        
        # End of epoch
        epoch_time = time.time() - epoch_start
        avg_epoch_loss = epoch_loss / num_batches
        
        if is_master:
            print(f"\nEpoch {epoch} completed in {epoch_time:.1f}s")
            print(f"  Average loss: {avg_epoch_loss:.4f}")
    
    # Final evaluation ONLY on master after training completes
    if is_master:
        print("\n" + "="*60)
        print("TRAINING COMPLETE - Evaluating final model...")
        print("="*60)
        
        # Move to CPU for safe evaluation
        model_cpu = model.cpu()
        eval_device = torch.device('cpu')
        
        # Create CPU dataloader for evaluation
        _, eval_dataset_cpu, _ = load_and_prepare_dataset(config)
        eval_loader_cpu = DataLoader(
            eval_dataset_cpu,
            batch_size=32,
            shuffle=False
        )
        
        model_cpu.eval()
        final_preds = []
        final_labels = []
        
        with torch.no_grad():
            for batch in eval_loader_cpu:
                batch = {k: v.to(eval_device) for k, v in batch.items()}
                outputs = model_cpu(**batch)
                preds = torch.argmax(outputs.logits, dim=-1)
                final_preds.extend(preds.numpy())
                final_labels.extend(batch['labels'].numpy())
        
        final_accuracy = accuracy_score(final_labels, final_preds)
        
        # Save final model
        final_dir = os.path.join(output_dir, f'final_model_acc_{final_accuracy:.4f}')
        os.makedirs(final_dir, exist_ok=True)
        model_cpu.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        
        # Save summary
        summary = {
            'baseline_accuracy': float(baseline_accuracy),
            'final_accuracy': float(final_accuracy),
            'improvement': float(final_accuracy - baseline_accuracy),
            'total_steps': global_step,
            'checkpoints_saved': checkpoint_count,
            'checkpoints': checkpoint_info
        }
        
        with open(os.path.join(output_dir, 'training_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Baseline: {baseline_accuracy:.4f}")
        print(f"Final: {final_accuracy:.4f}")
        print(f"Improvement: +{(final_accuracy - baseline_accuracy):.4f}")
        print(f"Results saved to: {output_dir}/")

def main():
    """Entry point"""
    xmp.spawn(train_bert_with_checkpoints, args=())

if __name__ == "__main__":
    main()
