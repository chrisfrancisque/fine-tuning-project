"""
Create a warmed baseline model with ~60% accuracy.
This script should be run ONCE to create a consistent starting point.
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
    """Create and save a 60% accuracy baseline"""
    
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
    
    # Store initial classifier weights to verify training
    initial_classifier_weight = model.classifier.weight.data.clone()
    
    # Freeze encoder, train only classifier and pooler
    trainable_params = []
    frozen_params = []
    for name, param in model.named_parameters():
        if 'encoder' in name:
            param.requires_grad = False
            frozen_params.append(name)
        else:
            param.requires_grad = True
            trainable_params.append(name)
            print(f"Training: {name}")
    
    print(f"\nTrainable parameters: {len(trainable_params)}")
    print(f"Frozen parameters: {len(frozen_params)}")
    
    model.to(device)
    
    # Load data - use subset for baseline training
    temp_config = config
    temp_config.train_samples = 5000  # Enough for stable training
    train_dataset, eval_dataset, _ = load_and_prepare_dataset(temp_config)
    train_loader, eval_loader = create_dataloaders(train_dataset, eval_dataset, temp_config)
    
    # High learning rate for classifier/pooler only
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3
    )
    
    print(f"\nTraining classifier head to ~60% accuracy...")
    print("=" * 60)
    
    # Train until 60% accuracy
    best_accuracy = 0.0
    training_history = []
    
    for epoch in range(10):  # Max 10 epochs
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = total_loss / num_batches
        
        # Evaluate
        model.eval()
        all_preds = []
        all_labels = []
        eval_loss = 0
        
        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
                eval_loss += outputs.loss.item()
        
        accuracy = accuracy_score(all_labels, all_preds)
        avg_eval_loss = eval_loss / len(eval_loader)
        
        training_history.append({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'eval_loss': avg_eval_loss,
            'accuracy': float(accuracy)
        })
        
        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, "
              f"Eval Loss = {avg_eval_loss:.4f}, Accuracy = {accuracy:.4f}")
        
        if accuracy >= 0.60:
            best_accuracy = accuracy
            print(f"\n✓ Target reached: {accuracy:.4f}")
            break
        
        best_accuracy = max(best_accuracy, accuracy)
    
    # Check that classifier actually changed
    final_classifier_weight = model.classifier.weight.data
    weight_change = torch.mean(torch.abs(final_classifier_weight - initial_classifier_weight.to(device)))
    print(f"\nClassifier weight change: {weight_change:.6f}")
    
    if weight_change < 0.001:
        print("⚠️ WARNING: Classifier weights barely changed!")
    else:
        print("✓ Classifier successfully trained")
    
    # Unfreeze all parameters for future full fine-tuning
    for param in model.parameters():
        param.requires_grad = True
    
    # Save the warmed baseline
    output_dir = "warmed_baseline_60pct"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model (this creates pytorch_model.bin ~418MB)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save detailed metadata
    metadata = {
        'accuracy': float(best_accuracy),
        'seed': 42,
        'train_samples': 5000,
        'method': 'classifier_pooler_training',
        'epochs_trained': len(training_history),
        'classifier_weight_change': float(weight_change),
        'training_history': training_history,
        'created_at': datetime.now().isoformat(),
        'bert_model': 'bert-base-uncased',
        'num_labels': 2
    }
    
    with open(f"{output_dir}/baseline_info.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✓ Baseline saved to {output_dir}/")
    print(f"✓ Final accuracy: {best_accuracy:.4f}")
    print(f"✓ Model size: ~418MB (pytorch_model.bin)")
    print(f"{'='*60}")
    
    return model, tokenizer, best_accuracy

if __name__ == "__main__":
    model, tokenizer, accuracy = create_warmed_baseline_60pct()
    
    # Verify the saved model loads correctly
    print("\nVerifying saved model...")
    test_model = AutoModelForSequenceClassification.from_pretrained("warmed_baseline_60pct")
    print("✓ Model loads successfully")
    
    # Check file sizes
    import os
    baseline_dir = "warmed_baseline_60pct"
    for file in os.listdir(baseline_dir):
        filepath = os.path.join(baseline_dir, file)
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"  {file}: {size_mb:.2f} MB")
