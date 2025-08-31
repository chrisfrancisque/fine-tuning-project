"""
BERT training with full model saves and live accuracy
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

def evaluate_model(model, eval_loader, device):
    """Quick evaluation on subset for live accuracy"""
    model.eval()
    preds = []
    labels = []
    
    # Only evaluate on first 100 batches for speed
    for i, batch in enumerate(eval_loader):
        if i >= 100:
            break
        with torch.no_grad():
            outputs = model(**batch)
            pred = torch.argmax(outputs.logits, dim=-1)
            preds.extend(pred.cpu().numpy())
            labels.extend(batch['labels'].cpu().numpy())
    
    accuracy = accuracy_score(labels, preds)
    model.train()
    return accuracy

def train_bert(index=None):
    """Training with full saves and accuracy monitoring"""
    
    device = xm.xla_device()
    
    # Setup output directory
    if xm.is_master_ordinal():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results_full_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"Output directory: {output_dir}")
        print(f"{'='*60}\n")
        
        with open('warmed_baseline_60pct/baseline_info.json', 'r') as f:
            baseline_info = json.load(f)
        baseline_acc = baseline_info['accuracy']
        print(f"Starting from baseline: {baseline_acc:.2%}\n")
    
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained('warmed_baseline_60pct')
    tokenizer = AutoTokenizer.from_pretrained('warmed_baseline_60pct')
    model.to(device)
    
    # Load data
    train_dataset, eval_dataset, _ = load_and_prepare_dataset(config)
    train_loader, eval_loader = create_dataloaders(train_dataset, eval_dataset, config)
    train_loader = pl.MpDeviceLoader(train_loader, device)
    eval_loader = pl.MpDeviceLoader(eval_loader, device)
    
    # Setup training
    total_steps = len(train_loader) * config.num_train_epochs
    checkpoint_every = 100  # Save every 100 steps
    
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
    checkpoint_num = 0
    recent_losses = []
    
    xm.master_print(f"Starting training: {total_steps} total steps")
    xm.master_print(f"Checkpoints every {checkpoint_every} steps\n")
    
    for epoch in range(config.num_train_epochs):
        epoch_start = time.time()
        
        for step, batch in enumerate(train_loader):
            outputs = model(**batch)
            loss = outputs.loss
            
            loss.backward()
            xm.optimizer_step(optimizer)
            scheduler.step()
            optimizer.zero_grad()
            
            recent_losses.append(loss.item())
            global_step += 1
            
            # Regular progress updates
            if step % 25 == 0:
                avg_loss = np.mean(recent_losses[-25:]) if recent_losses else loss.item()
                xm.master_print(f"[Epoch {epoch+1}/{config.num_train_epochs}] "
                               f"Step {step}/{len(train_loader)} | "
                               f"Global: {global_step}/{total_steps} | "
                               f"Loss: {avg_loss:.4f}")
            
            # Save checkpoint with model weights
            if global_step % checkpoint_every == 0:
                xm.master_print(f"\n{'='*50}")
                xm.master_print(f"Checkpoint at step {global_step}")
                
                # Synchronize all processes
                xm.rendezvous(f'checkpoint_{global_step}')
                
                if xm.is_master_ordinal():
                    # Evaluate for accuracy
                    print("Evaluating accuracy...")
                    accuracy = evaluate_model(model, eval_loader, device)
                    
                    # Prepare checkpoint directory
                    checkpoint_dir = os.path.join(output_dir, f"checkpoint_step_{global_step:04d}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    
                    # Save using xm.save (TPU-safe method)
                    model_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
                    xm.save(model.state_dict(), model_path)
                    
                    # Save tokenizer and config
                    tokenizer.save_pretrained(checkpoint_dir)
                    model.config.save_pretrained(checkpoint_dir)
                    
                    # Save checkpoint info
                    checkpoint_info = {
                        'step': global_step,
                        'epoch': epoch,
                        'accuracy': float(accuracy),
                        'improvement_from_baseline': float(accuracy - baseline_acc),
                        'avg_loss': float(np.mean(recent_losses[-100:])),
                        'checkpoint_number': checkpoint_num
                    }
                    
                    with open(os.path.join(checkpoint_dir, 'checkpoint_info.json'), 'w') as f:
                        json.dump(checkpoint_info, f, indent=2)
                    
                    print(f"✓ Accuracy: {accuracy:.4f} ({accuracy-baseline_acc:+.4f} from baseline)")
                    print(f"✓ Saved model to {checkpoint_dir}")
                    print(f"{'='*50}\n")
                    
                    checkpoint_num += 1
                
                # Ensure model is back in training mode
                model.train()
        
        # End of epoch
        epoch_time = time.time() - epoch_start
        xm.master_print(f"\nEpoch {epoch+1} complete in {epoch_time:.1f}s\n")
    
    # Final save
    xm.master_print("\n" + "="*60)
    xm.master_print("Training complete! Saving final model...")
    xm.rendezvous('final_save')
    
    if xm.is_master_ordinal():
        # Final evaluation
        final_accuracy = evaluate_model(model, eval_loader, device)
        
        # Save final model
        final_dir = os.path.join(output_dir, "final_model")
        os.makedirs(final_dir, exist_ok=True)
        
        xm.save(model.state_dict(), os.path.join(final_dir, "pytorch_model.bin"))
        tokenizer.save_pretrained(final_dir)
        model.config.save_pretrained(final_dir)
        
        # Summary
        summary = {
            'baseline_accuracy': float(baseline_acc),
            'final_accuracy': float(final_accuracy),
            'total_improvement': float(final_accuracy - baseline_acc),
            'total_steps': global_step,
            'total_checkpoints': checkpoint_num
        }
        
        with open(os.path.join(output_dir, 'training_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nFINAL RESULTS:")
        print(f"  Baseline: {baseline_acc:.4f}")
        print(f"  Final: {final_accuracy:.4f}")
        print(f"  Improvement: +{final_accuracy - baseline_acc:.4f}")
        print(f"  Saved to: {output_dir}/")
        print("="*60)

def main():
    xmp.spawn(train_bert, args=())

if __name__ == "__main__":
    main()
