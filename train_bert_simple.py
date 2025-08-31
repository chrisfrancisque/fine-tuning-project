"""
Simplified BERT training - saves checkpoints without device transfers
"""

import os
import torch
import json
import time
import numpy as np
from datetime import datetime
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score
from config import config
from data_utils import load_and_prepare_dataset, create_dataloaders

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

def train_bert(index=None):
    """Simple training without problematic checkpoint saves"""
    
    device = xm.xla_device()
    
    # Only master process prints
    if xm.is_master_ordinal():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving to: {output_dir}")
        
        # Load baseline info
        with open('warmed_baseline_60pct/baseline_info.json', 'r') as f:
            baseline_info = json.load(f)
        print(f"Baseline accuracy: {baseline_info['accuracy']:.2%}")
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained('warmed_baseline_60pct')
    model.to(device)
    
    # Load data
    train_dataset, eval_dataset, tokenizer = load_and_prepare_dataset(config)
    train_loader, eval_loader = create_dataloaders(train_dataset, eval_dataset, config)
    train_loader = pl.MpDeviceLoader(train_loader, device)
    
    # Setup training
    total_steps = len(train_loader) * config.num_train_epochs
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    model.train()
    global_step = 0
    losses = []
    
    for epoch in range(config.num_train_epochs):
        epoch_loss = 0
        num_batches = 0
        
        for step, batch in enumerate(train_loader):
            outputs = model(**batch)
            loss = outputs.loss
            
            loss.backward()
            xm.optimizer_step(optimizer)
            scheduler.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            losses.append(loss.item())
            num_batches += 1
            global_step += 1
            
            # Print progress
            if step % 50 == 0:
                xm.master_print(f"Epoch {epoch}, Step {step}/{len(train_loader)}, "
                               f"Loss: {loss.item():.4f}")
            
            # Save checkpoint every 200 steps (less frequent, more stable)
            if global_step % 200 == 0:
                xm.master_print(f"Step {global_step}: Synchronizing for checkpoint...")
                xm.rendezvous('checkpoint')  # Sync all processes
                
                if xm.is_master_ordinal():
                    # Save checkpoint info only (not the model yet)
                    checkpoint_info = {
                        'step': global_step,
                        'epoch': epoch,
                        'avg_loss': float(np.mean(losses[-200:]))
                    }
                    with open(f"{output_dir}/checkpoint_{global_step}.json", 'w') as f:
                        json.dump(checkpoint_info, f, indent=2)
                    print(f"✓ Saved checkpoint info at step {global_step}")
        
        # End of epoch
        if xm.is_master_ordinal():
            print(f"Epoch {epoch} complete. Avg loss: {epoch_loss/num_batches:.4f}")
    
    # Save final model (only once at the end)
    xm.master_print("Training complete! Saving final model...")
    xm.rendezvous('save_model')
    
    if xm.is_master_ordinal():
        # Use xm.save to properly save from TPU
        xm.save(model.state_dict(), f"{output_dir}/model_final.pt")
        tokenizer.save_pretrained(output_dir)
        
        # Save training summary
        with open(f"{output_dir}/training_complete.json", 'w') as f:
            json.dump({
                'total_steps': global_step,
                'final_loss': float(np.mean(losses[-100:])),
                'epochs': config.num_train_epochs
            }, f, indent=2)
        
        print(f"✓ Model saved to {output_dir}/")

def main():
    xmp.spawn(train_bert, args=())

if __name__ == "__main__":
    main()
