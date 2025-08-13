import torch
from config import config
from data_utils import load_and_prepare_dataset, create_dataloaders
from load_baseline import load_baseline_model
from transformers import AdamW

# Override config for quick test
config.train_samples = 100  # Just 100 samples
config.num_train_epochs = 1  # Just 1 epoch
config.per_device_train_batch_size = 10  # Small batch

def test_baseline_loading():
    print("="*60)
    print("TESTING BASELINE MODEL LOADING")
    print("="*60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load baseline model
    print("\n1. Loading baseline model...")
    model, tokenizer, baseline_info = load_baseline_model(
        baseline_path='baseline_model_seed42',
        device=device
    )
    
    print(f"\n✓ Baseline loaded successfully!")
    print(f"  - Warm-up accuracy: {baseline_info.get('warm_up_accuracy', 0):.4f}")
    print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load small dataset
    print("\n2. Loading small dataset (100 samples)...")
    train_dataset, eval_dataset, _ = load_and_prepare_dataset(config)
    train_dataloader, eval_dataloader = create_dataloaders(
        train_dataset, eval_dataset, config
    )
    print(f"✓ Dataset loaded: {len(train_dataset)} train, {len(eval_dataset)} eval")
    
    # Test forward pass
    print("\n3. Testing forward pass...")
    model.train()
    batch = next(iter(train_dataloader))
    batch = {k: v.to(device) for k, v in batch.items()}
    
    outputs = model(**batch)
    loss = outputs.loss
    print(f"✓ Forward pass successful! Loss: {loss.item():.4f}")
    
    # Test backward pass
    print("\n4. Testing backward pass...")
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"✓ Backward pass successful!")
    
    # Quick evaluation
    print("\n5. Quick evaluation on baseline...")
    model.eval()
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for i, batch in enumerate(eval_dataloader):
            if i >= 5:  # Just test 5 batches
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            predictions = torch.argmax(outputs.logits, dim=-1)
            total_correct += (predictions == batch['labels']).sum().item()
            total_samples += len(batch['labels'])
    
    quick_accuracy = total_correct / total_samples if total_samples > 0 else 0
    print(f"✓ Quick evaluation accuracy: {quick_accuracy:.4f}")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print("\nBaseline model is working correctly.")
    print("Ready for full training with train_bert_local.py or train_bert_tpu.py")
    
    return True

if __name__ == "__main__":
    test_baseline_loading()