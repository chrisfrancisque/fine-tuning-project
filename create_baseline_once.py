"""
Create a baseline model with 500 samples - targeting ~60% accuracy
"""

import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score
from data_utils import load_and_prepare_dataset, create_dataloaders
from config import config
import json
import os
from datetime import datetime

def create_warmed_baseline_60pct():
    """Create baseline with 500 samples"""
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load fresh BERT model
    print("Loading BERT-base-uncased...")
    model = AutoModelForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2
    )
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # ONLY train classifier layer
    trainable_params = []
    frozen_params = []
    
    for name, param in model.named_parameters():
        if 'classifier' in name:
            param.requires_grad = True
            trainable_params.append(name)
            print(f"Training: {name}")
        else:
            param.requires_grad = False
            frozen_params.append(name)
    
    print(f"\n✓ Trainable parameters: {len(trainable_params)}")
    print(f"✓ Frozen parameters: {len(frozen_params)}")
    
    # Count actual parameters
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())
    print(f"✓ Training {trainable_count:,} / {total_count:,} parameters ({trainable_count/total_count*100:.2f}%)")
    
    model.to(device)
    
    # Use 500 samples instead of 1000
    temp_config = config
    temp_config.train_samples = 500  # Reduced to 500
    train_dataset, eval_dataset, _ = load_and_prepare_dataset(temp_config)
    train_loader, eval_loader = create_dataloaders(train_dataset, eval_dataset, temp_config)
    
    # Calculate actual number of batches
    num_batches_available = len(train_loader)
    print(f"✓ Total batches available: {num_batches_available}")
    
    # Same learning rate that worked
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=2e-3
    )
    
    print(f"\nTraining ONLY classifier head...")
    print(f"Samples: 500")
    print(f"Max steps: 40 (or {num_batches_available} if less)")
    print(f"Learning rate: 2e-3")
    print("=" * 60)
    
    # Train for up to 40 steps (but will stop at 31 with 500 samples)
    model.train()
    total_loss = 0
    num_batches = 0
    max_steps = 40
    
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= max_steps:
            print(f"Stopping at {max_steps} steps")
            break
            
        batch = {k: v.to(device) for k, v in batch.items()}
        
        outputs = model(**batch)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        num_batches += 1
        
        if num_batches % 5 == 0:
            print(f"  Step {num_batches}, Loss: {loss.item():.4f}")
    
    avg_train_loss = total_loss / num_batches
    
    # Evaluate
    print(f"\nCompleted {num_batches} training steps")
    print("Evaluating baseline...")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    
    print(f"\n✓ Baseline accuracy: {accuracy:.4f}")
    print(f"✓ Average training loss: {avg_train_loss:.4f}")
    
    # Unfreeze all parameters for future full fine-tuning
    for param in model.parameters():
        param.requires_grad = True
    
    # Save the baseline
    output_dir = "warmed_baseline_60pct"
    os.makedirs(output_dir, exist_ok=True)
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save metadata
    metadata = {
        'accuracy': float(accuracy),
        'seed': 42,
        'train_samples': 500,
        'train_steps': num_batches,
        'max_steps': max_steps,
        'learning_rate': 2e-3,
        'train_loss': float(avg_train_loss),
        'created_at': datetime.now().isoformat(),
        'method': 'classifier_only_500samples',
        'trainable_params': trainable_count,
        'total_params': total_count
    }
    
    with open(f"{output_dir}/baseline_info.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✓ Baseline saved to {output_dir}/")
    print(f"✓ Accuracy: {accuracy:.4f}")
    print(f"✓ Trained for {num_batches} steps on 500 samples")
    print(f"✓ Ready for full fine-tuning!")
    print(f"{'='*60}")
    
    return model, tokenizer, accuracy

if __name__ == "__main__":
    model, tokenizer, accuracy = create_warmed_baseline_60pct()
    
    # Verify the saved model loads correctly
    print("\nVerifying saved model...")
    test_model = AutoModelForSequenceClassification.from_pretrained("warmed_baseline_60pct")
    print("✓ Model loads successfully")
