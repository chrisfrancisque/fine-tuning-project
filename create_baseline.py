import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW
from tqdm import tqdm
import json
from datetime import datetime
from sklearn.metrics import accuracy_score
import random
import numpy as np

from config import config
from data_utils import load_and_prepare_dataset, create_dataloaders

def set_seed(seed=42):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_baseline_local():
    """Create baseline model on local machine"""
    
    # Set seed for reproducible initialization
    set_seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directory for baseline
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    baseline_dir = f"baseline_model_seed42"  # Fixed name for easy copying
    os.makedirs(baseline_dir, exist_ok=True)
    print(f"Creating baseline model in: {baseline_dir}")
    
    # Load data
    print("\nLoading SST-2 dataset...")
    train_dataset, eval_dataset, tokenizer = load_and_prepare_dataset(config)
    train_dataloader, eval_dataloader = create_dataloaders(
        train_dataset, eval_dataset, config
    )
    
    # Create model with FIXED SEED
    print("Loading BERT-base model with seed=42...")
    model = AutoModelForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2,
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        label2id={"NEGATIVE": 0, "POSITIVE": 1}
    )
    model.to(device)
    
    # Save the initial random weights of classifier
    initial_classifier_weight = model.classifier.weight.data.clone().cpu()
    initial_classifier_bias = model.classifier.bias.data.clone().cpu()
    
    # FREEZE all layers except classifier
    print("\nFreezing all layers except classifier...")
    trainable_params = 0
    frozen_params = 0
    
    for name, param in model.named_parameters():
        if 'classifier' in name:
            param.requires_grad = True
            trainable_params += param.numel()
            print(f"  Trainable: {name} ({param.numel()} params)")
        else:
            param.requires_grad = False
            frozen_params += param.numel()
    
    print(f"\nParameter counts:")
    print(f"  Trainable (classifier): {trainable_params:,}")
    print(f"  Frozen (BERT encoder): {frozen_params:,}")
    print(f"  Total: {trainable_params + frozen_params:,}")
    
    # Create optimizer ONLY for classifier
    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params_list, lr=config.learning_rate)
    
    # TRAIN FOR EXACTLY 1 EPOCH
    print("\n" + "="*60)
    print("WARMING UP CLASSIFIER (1 EPOCH)")
    print("="*60)
    
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(train_dataloader, desc="Training classifier")
    for batch in progress_bar:
        batch = {k: v.to(device) for k, v in batch.items()}
        
        outputs = model(**batch)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        num_batches += 1
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_train_loss = total_loss / num_batches
    print(f"Average training loss: {avg_train_loss:.4f}")
    
    # EVALUATE the warmed model
    print("\nEvaluating warmed baseline...")
    model.eval()
    all_predictions = []
    all_labels = []
    eval_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            
            eval_loss += outputs.loss.item()
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_predictions)
    avg_eval_loss = eval_loss / len(eval_dataloader)
    
    print(f"Validation Loss: {avg_eval_loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Check classifier weights changed
    final_classifier_weight = model.classifier.weight.data.clone().cpu()
    final_classifier_bias = model.classifier.bias.data.clone().cpu()
    
    weight_change = torch.mean(torch.abs(final_classifier_weight - initial_classifier_weight))
    bias_change = torch.mean(torch.abs(final_classifier_bias - initial_classifier_bias))
    
    print(f"\nClassifier weight change: {weight_change:.6f}")
    print(f"Classifier bias change: {bias_change:.6f}")
    
    # Save baseline info
    baseline_info = {
        'timestamp': timestamp,
        'seed': 42,
        'device': str(device),
        'warm_up_epochs': 1,
        'trainable_params': trainable_params,
        'frozen_params': frozen_params,
        'total_params': trainable_params + frozen_params,
        'train_samples': len(train_dataset),
        'warm_up_train_loss': avg_train_loss,
        'warm_up_eval_loss': avg_eval_loss,
        'warm_up_accuracy': accuracy,
        'classifier_weight_change': float(weight_change),
        'classifier_bias_change': float(bias_change),
        'model_name': 'bert-base-uncased',
        'learning_rate': config.learning_rate,
        'batch_size': config.per_device_train_batch_size
    }
    
    with open(os.path.join(baseline_dir, 'baseline_info.json'), 'w') as f:
        json.dump(baseline_info, f, indent=4)
    
    # IMPORTANT: Unfreeze all parameters before saving
    print("\nUnfreezing all parameters before saving...")
    for param in model.parameters():
        param.requires_grad = True
    
    # Save the model
    model.save_pretrained(baseline_dir)
    tokenizer.save_pretrained(baseline_dir)
    
    # Also save raw PyTorch checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'baseline_info': baseline_info
    }, os.path.join(baseline_dir, 'pytorch_model.bin'))
    
    print("\n" + "="*60)
    print("BASELINE MODEL CREATED SUCCESSFULLY!")
    print("="*60)
    print(f"Location: {baseline_dir}/")
    print("\nFiles created:")
    print("  - config.json (model config)")
    print("  - pytorch_model.bin (model weights)")
    print("  - tokenizer files")
    print("  - baseline_info.json (training info)")
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Copy this folder to each repository:")
    print(f"   cp -r {baseline_dir} ../full-fine-tuning-repo/")
    print(f"   cp -r {baseline_dir} ../lora-repo/")
    print(f"   cp -r {baseline_dir} ../bert-mft-repo/")
    print("\n2. In each repo, load the model:")
    print(f"   model = AutoModelForSequenceClassification.from_pretrained('./{baseline_dir}')")
    print("\n3. All three methods will start from the EXACT same weights!")
    print("="*60)
    
    return baseline_dir

if __name__ == "__main__":
    baseline_path = create_baseline_local()
    print(f"\n✓ Baseline model ready at: {baseline_path}")