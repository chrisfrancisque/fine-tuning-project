"""
BERT Full Fine-tuning with Strategic Checkpoint Saving
Uses pre-created warmed baseline and saves checkpoints at ~10% improvement intervals
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

# Try to import TPU libraries
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp
    TPU_AVAILABLE = True
except ImportError:
    TPU_AVAILABLE = False
    print("TPU libraries not available. Will use CPU/GPU.")


def load_warmed_baseline(baseline_path='warmed_baseline_60pct'):
    """Load the pre-created warmed baseline model"""
    
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(
            f"Baseline not found at {baseline_path}. "
            f"Please run 'python create_baseline_once.py' first!"
        )
    
    print(f"Loading warmed baseline from {baseline_path}...")
    
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(baseline_path)
    tokenizer = AutoTokenizer.from_pretrained(baseline_path)
    
    # Load metadata
    with open(os.path.join(baseline_path, 'baseline_info.json'), 'r') as f:
        baseline_info = json.load(f)
    
    print(f"✓ Loaded baseline with {baseline_info['accuracy']:.2%} accuracy")
    print(f"  Created: {baseline_info.get('created_at', 'Unknown')}")
    print(f"  Method: {baseline_info.get('method', 'Unknown')}")
    
    # Ensure all parameters are unfrozen for full fine-tuning
    for param in model.parameters():
        param.requires_grad = True
    
    return model, tokenizer, baseline_info['accuracy']


def estimate_steps_per_improvement(train_dataloader, baseline_accuracy=0.60, target_improvement=0.10):
    """
    Estimate training steps needed for each 10% accuracy improvement.
    Based on typical BERT fine-tuning curves on SST-2.
    """
    
    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * config.num_train_epochs
    
    # Empirical estimates based on BERT fine-tuning patterns
    # Early improvements are faster, later ones are slower
    
    if baseline_accuracy < 0.70:
        # 60% -> 70%: relatively quick
        steps_for_10pct = int(total_steps * 0.15)
    elif baseline_accuracy < 0.80:
        # 70% -> 80%: moderate pace
        steps_for_10pct = int(total_steps * 0.25)
    else:
        # 80% -> 90%: slower improvement
        steps_for_10pct = int(total_steps * 0.35)
    
    # Ensure minimum interval
    steps_for_10pct = max(steps_for_10pct, steps_per_epoch // 2)
    
    return steps_for_10pct


def save_checkpoint(model, tokenizer, metrics, output_dir, checkpoint_name):
    """Save a model checkpoint with metadata"""
    
    checkpoint_dir = os.path.join(output_dir, checkpoint_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save model weights (creates pytorch_model.bin ~418MB)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    
    # Save metadata
    metadata = {
        **metrics,
        'checkpoint_name': checkpoint_name,
        'saved_at': datetime.now().isoformat(),
        'model_file': 'pytorch_model.bin',
        'model_size_mb': 418  # Approximate
    }
    
    with open(os.path.join(checkpoint_dir, 'checkpoint_info.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Checkpoint saved: {checkpoint_name} (Accuracy: {metrics.get('accuracy', 0):.2%})")
    
    return checkpoint_dir


def train_bert_with_checkpoints(index=None):
    """Main training function with checkpoint saving at ~10% improvement intervals"""
    
    # Setup device
    if TPU_AVAILABLE and index is not None:
        device = xm.xla_device()
        is_master = xm.is_master_ordinal()
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        is_master = True
    
    # Create output directory
    if is_master:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results_10pct_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        print(f"Results will be saved to: {output_dir}")
    else:
        output_dir = None
    
    # Synchronize output_dir if using TPU
    if TPU_AVAILABLE and index is not None:
        output_dir = xm.mesh_reduce('broadcast', output_dir, lambda x: x[0] if x else None)
    
    # Load the pre-created warmed baseline
    if is_master:
        print("\n" + "="*60)
        print("STARTING BERT FULL FINE-TUNING")
        print("="*60)
    
    model, tokenizer, baseline_accuracy = load_warmed_baseline('warmed_baseline_60pct')
    model.to(device)
    
    # Load data for full training
    train_dataset, eval_dataset, _ = load_and_prepare_dataset(config)
    train_dataloader, eval_dataloader = create_dataloaders(
        train_dataset, eval_dataset, config
    )
    
    # Wrap dataloaders for TPU if available
    if TPU_AVAILABLE and index is not None:
        train_dataloader = pl.MpDeviceLoader(train_dataloader, device)
        eval_dataloader = pl.MpDeviceLoader(eval_dataloader, device)
    
    # Calculate checkpoint intervals
    steps_per_checkpoint = estimate_steps_per_improvement(
        train_dataloader,
        baseline_accuracy, 
        target_improvement=0.10
    )
    
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
    all_accuracies = []
    checkpoint_info = []
    
    # Training loop
    model.train()
    
    for epoch in range(config.num_train_epochs):
        epoch_start = time.time()
        epoch_loss = 0
        num_batches = 0
        
        for step, batch in enumerate(train_dataloader):
            # Move batch to device
            if not (TPU_AVAILABLE and index is not None):
                batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Optimizer step
            if TPU_AVAILABLE and index is not None:
                xm.optimizer_step(optimizer)
            else:
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad()
            
            # Tracking
            epoch_loss += loss.item()
            all_losses.append(loss.item())
            num_batches += 1
            global_step += 1
            
            # Log progress
            if step % 50 == 0 and is_master:
                print(f"Epoch {epoch}, Step {step}/{len(train_dataloader)}, "
                      f"Loss: {loss.item():.4f}, Global Step: {global_step}")
            
            # Save checkpoint at intervals
            if global_step % steps_per_checkpoint == 0 and is_master:
                print(f"\nEvaluating for checkpoint at step {global_step}...")
                
                # Quick evaluation
                model.eval()
                
                eval_preds = []
                eval_labels = []
                eval_losses = []
                
                with torch.no_grad():
                    for eval_batch in eval_dataloader:
                        if not (TPU_AVAILABLE and index is not None):
                            eval_batch = {k: v.to(device) for k, v in eval_batch.items()}
                        
                        outputs = model(**eval_batch)
                        preds = torch.argmax(outputs.logits, dim=-1)
                        
                        eval_preds.extend(preds.cpu().numpy())
                        eval_labels.extend(eval_batch['labels'].cpu().numpy())
                        eval_losses.append(outputs.loss.item())
                
                checkpoint_accuracy = accuracy_score(eval_labels, eval_preds)
                checkpoint_loss = np.mean(eval_losses)
                
                # Calculate improvement from baseline
                improvement = checkpoint_accuracy - baseline_accuracy
                
                # Save checkpoint
                checkpoint_name = f'checkpoint_{checkpoint_count:02d}_step_{global_step}_acc_{checkpoint_accuracy:.4f}'
                
                metrics = {
                    'step': global_step,
                    'epoch': epoch,
                    'epoch_progress': step / len(train_dataloader),
                    'accuracy': float(checkpoint_accuracy),
                    'improvement_from_baseline': float(improvement),
                    'train_loss': float(epoch_loss / num_batches),
                    'eval_loss': float(checkpoint_loss)
                }
                
                # Save checkpoint (moves model to CPU temporarily)
                model_cpu = model.cpu()
                checkpoint_path = save_checkpoint(
                    model_cpu, tokenizer, metrics, output_dir, checkpoint_name
                )
                
                # Move model back to device
                model.to(device)
                
                # Track checkpoint info
                checkpoint_info.append({
                    'checkpoint': checkpoint_count,
                    'path': checkpoint_path,
                    **metrics
                })
                
                all_accuracies.append(checkpoint_accuracy)
                checkpoint_count += 1
                
                print(f"  Improvement from baseline: {improvement:.2%}")
                
                # Back to training
                model.train()
        
        # End of epoch
        epoch_time = time.time() - epoch_start
        avg_epoch_loss = epoch_loss / num_batches
        
        if is_master:
            print(f"\nEpoch {epoch} completed in {epoch_time:.1f}s")
            print(f"  Average loss: {avg_epoch_loss:.4f}")
    
    # Final evaluation and save
    if is_master:
        print("\n" + "="*60)
        print("FINAL EVALUATION")
        print("="*60)
        
        model.eval()
        
        final_preds = []
        final_labels = []
        final_probs = []
        
        with torch.no_grad():
            for batch in eval_dataloader:
                if not (TPU_AVAILABLE and index is not None):
                    batch = {k: v.to(device) for k, v in batch.items()}
                
                outputs = model(**batch)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)
                
                final_preds.extend(preds.cpu().numpy())
                final_labels.extend(batch['labels'].cpu().numpy())
                final_probs.extend(probs.cpu().numpy())
        
        final_accuracy = accuracy_score(final_labels, final_preds)
        total_improvement = final_accuracy - baseline_accuracy
        
        # Save final model
        final_name = f'final_model_acc_{final_accuracy:.4f}'
        final_metrics = {
            'step': global_step,
            'epoch': config.num_train_epochs,
            'accuracy': float(final_accuracy),
            'improvement_from_baseline': float(total_improvement),
            'total_checkpoints': checkpoint_count
        }
        
        model_cpu = model.cpu()
        save_checkpoint(model_cpu, tokenizer, final_metrics, output_dir, final_name)
        
        # Generate plots
        print("\nGenerating visualizations...")
        
        # Training loss plot
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(all_losses, alpha=0.7)
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.grid(True, alpha=0.3)
        
        # Mark checkpoint positions
        for info in checkpoint_info:
            plt.axvline(x=info['step'], color='red', alpha=0.3, linestyle='--')
        
        # Accuracy progression
        plt.subplot(1, 2, 2)
        checkpoint_steps = [info['step'] for info in checkpoint_info]
        checkpoint_accs = [info['accuracy'] for info in checkpoint_info]
        
        plt.plot([0] + checkpoint_steps + [global_step], 
                 [baseline_accuracy] + checkpoint_accs + [final_accuracy], 
                 'o-', linewidth=2, markersize=8)
        plt.axhline(y=baseline_accuracy, color='green', alpha=0.5, 
                    linestyle='--', label=f'Baseline ({baseline_accuracy:.2%})')
        plt.xlabel('Training Steps')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Progression')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_progress.png'), dpi=150)
        plt.close()
        
        # Confusion matrix
        cm = confusion_matrix(final_labels, final_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Final Confusion Matrix (Accuracy: {final_accuracy:.2%})')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
        plt.close()
        
        # Save comprehensive summary
        summary = {
            'training_config': {
                'model': config.model_name,
                'learning_rate': config.learning_rate,
                'batch_size': config.total_train_batch_size,
                'epochs': config.num_train_epochs,
                'warmup_steps': config.warmup_steps,
                'weight_decay': config.weight_decay
            },
            'baseline': {
                'accuracy': float(baseline_accuracy),
                'source': 'warmed_baseline_60pct'
            },
            'results': {
                'final_accuracy': float(final_accuracy),
                'total_improvement': float(total_improvement),
                'total_steps': global_step,
                'checkpoints_saved': checkpoint_count,
                'checkpoint_interval': steps_per_checkpoint
            },
            'checkpoints': checkpoint_info,
            'output_directory': output_dir
        }
        
        with open(os.path.join(output_dir, 'training_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print final summary
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"Baseline Accuracy: {baseline_accuracy:.2%}")
        print(f"Final Accuracy: {final_accuracy:.2%}")
        print(f"Total Improvement: {total_improvement:.2%}")
        print(f"Checkpoints Saved: {checkpoint_count}")
        print(f"\nCheckpoint Progression:")
        for i, info in enumerate(checkpoint_info):
            print(f"  {i+1}. Step {info['step']:5d}: {info['accuracy']:.2%} "
                  f"(+{info['improvement_from_baseline']:.2%} from baseline)")
        print(f"\n📁 Results saved to: {output_dir}/")
        print(f"💾 Each checkpoint contains pytorch_model.bin (~418MB)")
        print("="*60)


def main():
    """Main entry point"""
    
    # Check if baseline exists
    if not os.path.exists('warmed_baseline_60pct'):
        print("ERROR: Warmed baseline not found!")
        print("Please run: python create_baseline_once.py")
        return
    
    if TPU_AVAILABLE:
        # Run on TPU
        xmp.spawn(train_bert_with_checkpoints, args=())
    else:
        # Run on CPU/GPU
        train_bert_with_checkpoints()


if __name__ == "__main__":
    main()
