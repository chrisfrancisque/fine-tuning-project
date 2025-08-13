import os
import torch
import sys

try:
    import torch_xla
    import torch_xla.core.xla_model as xm 
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp
    TPU_AVAILABLE = True

except ImportError:
    TPU_AVAILABLE = False
    print("Warning: TPU libraries not available. This script should be run on a TPU VM.")
    
from transformers import AdamW, get_linear_schedule_with_warmup
import time

from config import config
from data_utils import load_and_prepare_dataset, create_dataloaders
from model_utils import create_model, compute_metrics
import numpy as np
from load_baseline import load_baseline_model  # Add this new import
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime

def create_output_dir():
    """Directory for saving results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def plot_training_loss(losses, steps, output_dir):
    """Plot training loss vs steps"""
    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, 'b-', linewidth=2)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Training Loss', fontsize =12)
    plt.title('BERT Fine-tuning: Training Loss vs Steps', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_loss.png'), dpi=300)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, output_dir):
    """Plot Confusion Matrix"""
    cm = confusion_matrix(y_true, y_pred)

    #Flip Confusion matrix for normal layout
    cm = np.flipud(np.fliplr(cm))

    
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap = 'Blues',
               xticklabels=['Positive', 'Negative'],
                yticklabels=['Positive', 'Negative'])  
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix of BERT on SST-2', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()

def calculate_metrics(y_true, y_pred, y_proba, output_dir):
    """Calulate and save all metrics"""

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

    try:
        roc_auc = roc_auc_score(y_true, y_proba[:,1])
    except:
        roc_auc = None

    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc) if roc_auc else 'N/A'
    }

    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        f.write("BERT Fine-tuning Results on SST-2\n")
        f.write("="*50 + "\n\n")
        f.write(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")
        f.write(f"ROC-AUC: {roc_auc:.4f}\n" if roc_auc else "ROC-AUC: N/A\n")
    
    return metrics




def train_bert_on_tpu(index):
    """Training function for each TPU core"""
    # Set config for TPU
    config.use_tpu = True
    
    # Get TPU device
    device = xm.xla_device()
    
    #Only print from master
    if xm.is_master_ordinal():
       output_dir = create_output_dir()
       xm.master_print(f"Results will be saved to: {output_dir}")
    else:
        output_dir= None
    
    #Synchronize output_dir across all processes
    output_dir = xm.mesh_reduce('broadcast', output_dir, lambda x: x[0] if x else None)

     # Only print from master
    if xm.is_master_ordinal():
        print(f"Starting training on TPU core {index}")
        print(f"Total batch size: {config.total_train_batch_size}")
    
    # Load data on ALL processes (ignore tokenizer from dataset loading)
    train_dataset, eval_dataset, _ = load_and_prepare_dataset(config)  # <-- CHANGED: underscore instead of tokenizer
    
    # Synchronize to ensure data is loaded
    xm.rendezvous("data_loading")
    
    
    if xm.is_master_ordinal():
        xm.master_print("Loading warmed baseline model...")
    
    model, tokenizer, baseline_info = load_baseline_model(
        baseline_path='baseline_model_seed42',
        device='cpu'  # Load to CPU first, then move to TPU
    )
    model.to(device)
    
    if xm.is_master_ordinal():
        xm.master_print(f"✓ Using warmed baseline model")
        xm.master_print(f"  Baseline accuracy: {baseline_info.get('warm_up_accuracy', 0):.4f}")
        xm.master_print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        xm.master_print("="*50)
    
    
    # Create dataloaders  
    train_dataloader, eval_dataloader = create_dataloaders(
        train_dataset, eval_dataset, config
    )
    
    # Wrap dataloaders for TPU
    train_device_loader = pl.MpDeviceLoader(train_dataloader, device)
    eval_device_loader = pl.MpDeviceLoader(eval_dataloader, device)
    # Create dataloaders
    train_dataloader, eval_dataloader = create_dataloaders(
        train_dataset, eval_dataset, config
    )
    
    # Wrap dataloaders for TPU
    train_device_loader = pl.MpDeviceLoader(train_dataloader, device)
    eval_device_loader = pl.MpDeviceLoader(eval_dataloader, device)
    
    # Create model
    model = create_model(config)
    model.to(device)
    
    # Calculate total training steps
    total_steps = len(train_dataloader) * config.num_train_epochs
    
    # Create optimizer and scheduler
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

    #Tracking
    all_losses = []
    all_steps = []
    epoch_train_losses = []
    epoch_val_losses = []

    
    # Training loop
    model.train()
    global_step = 0


    for epoch in range(config.num_train_epochs):
        epoch_start = time.time()
        total_loss = 0
        
        for step, batch in enumerate(train_device_loader):
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # TPU-specific optimizer step
            xm.optimizer_step(optimizer)
            scheduler.step()
            optimizer.zero_grad()
            
            #Track Loss
            loss_value= loss.item()
            total_loss += loss_value
            global_step +=1

            # Store for plotting (only on master)
            if xm.is_master_ordinal():
                all_losses.append(loss_value)
                all_steps.append(global_step)

            
            # Log progress
            if step % 50 == 0:
                xm.master_print(
                    f"Epoch {epoch}, Step {step}/{len(train_dataloader)}, "
                    f"Loss: {loss.item():.4f}"
                )
        
        # End of epoch
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / len(train_dataloader)
        epoch_train_losses.append(avg_loss)

        xm.master_print(
            f"Epoch {epoch} completed in {epoch_time:.1f}s, "
            f"Average Loss: {avg_loss:.4f}"
        )

        # Validation at end of each epoch
        xm.master_print("Running Validation...")
        model.eval()
        eval_loss = 0
        all_predictions = []
        all_labels = []
        all_proba = []


            
        with torch.no_grad():
            for batch in eval_device_loader:
                outputs = model(**batch)
                loss = outputs.loss
                logits = outputs.logits

                eval_loss += loss.item()

                # Predictions and Proababilities
                proba = torch.softmax(logits, dim = -1)
                predictions = torch.argmax(logits, dim=-1)

                # Collect for metrics (only on master)
                if xm.is_master_ordinal():
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(batch['labels'].cpu().numpy())
                    all_proba.extend(proba.cpu().numpy())
        
        avg_eval_loss = eval_loss / len(eval_dataloader)
        epoch_val_losses.append(avg_eval_loss)
        xm.master_print(f"Validation Loss: {avg_eval_loss:.4f}")
        
        model.train()
    
    # Generate plots and metrics (only on master)
    if xm.is_master_ordinal():
        xm.master_print("\nGenerating plots and calculating metrics...")
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_proba = np.array(all_proba)
        
        # Plot training loss vs steps
        plot_training_loss(all_losses, all_steps, output_dir)
        xm.master_print("✓ Training loss plot saved")
        
        # Plot confusion matrix
        plot_confusion_matrix(all_labels, all_predictions, output_dir)
        xm.master_print("✓ Confusion matrix saved")
        
        # Calculate and save metrics
        metrics = calculate_metrics(all_labels, all_predictions, all_proba, output_dir)
        xm.master_print("✓ Metrics calculated and saved")
        
        # Print final metrics
        xm.master_print("\nFinal Metrics:")
        xm.master_print(f"Accuracy: {metrics['accuracy']:.4f}")
        xm.master_print(f"Precision: {metrics['precision']:.4f}")
        xm.master_print(f"Recall: {metrics['recall']:.4f}")
        xm.master_print(f"F1-Score: {metrics['f1_score']:.4f}")
        xm.master_print(f"ROC-AUC: {metrics['roc_auc']:.4f}" if metrics['roc_auc'] != 'N/A' else "ROC-AUC: N/A")
        
        # Save model
       # Save model and tokenizer
        model_path = os.path.join(output_dir, 'bert_tpu_model')
        xm.save(model.state_dict(), f"{model_path}.pt")
        
        # Save tokenizer too (it's from baseline)
        tokenizer.save_pretrained(model_path)
        
        xm.master_print(f"Model saved to {model_path}.pt")
        xm.master_print(f"Tokenizer saved to {model_path}/")
        
        xm.master_print(f"\nAll results saved to: {output_dir}")
    

            
          

def main():
    if not TPU_AVAILABLE:
        print("Error: TPU libraries not available!")
        print("This script must be run on a TPU VM.")
        return
    
    # Launch training on all TPU cores
    xmp.spawn(train_bert_on_tpu, args=())

if __name__ == "__main__":
    main()
