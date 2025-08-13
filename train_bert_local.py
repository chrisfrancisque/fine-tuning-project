import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import time
import os

from config import config
from data_utils import load_and_prepare_dataset, create_dataloaders
from model_utils import create_model, compute_metrics

def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:

        #Move batch to device

        batch = {k: v.to(device) for k,v in batch.items()}

        #Forward Pass
        outputs = model(**batch)
        loss = outputs.loss

        # Backwards Pass
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})


    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    """Evaluate Model"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss
        logits = outputs.logits

        total_loss += loss.item()

        predictions = torch.argmax(logits, dim=1)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(batch['labels'].cpu().numpy())
    
    metrics = compute_metrics(all_predictions, all_labels)
    avg_loss = total_loss / len(dataloader)

    return avg_loss, metrics

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load datasets
    train_dataset, eval_dataset, _ = load_and_prepare_dataset(config)  # Ignore tokenizer from here
    
    # Load baseline model
    model, tokenizer, baseline_info = create_model(config, use_baseline=True, device=device)
    model.to(device)
    
    print(f"\n{'='*50}")
    print(f"Using warmed baseline model")
    print(f"Baseline accuracy: {baseline_info.get('warm_up_accuracy', 0):.4f}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*50}\n")
    
    # Create dataloaders
    train_dataloader, eval_dataloader = create_dataloaders(
        train_dataset, eval_dataset, config
    )

    #Calculate total training steps
    total_steps = len(train_dataloader) * config.num_train_epochs

    # Create optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr = config.learning_rate,
        weight_decay=config.weight_decay
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
    )

     # Training loop
    print(f"\nStarting training for {config.num_train_epochs} epochs...")
    print(f"Total training steps: {total_steps}")
    print(f"Batch size: {config.per_device_train_batch_size}")
    
    for epoch in range(config.num_train_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{config.num_train_epochs}")
        print(f"{'='*50}")

        start_time = time.time()
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, device)
        train_time = time.time() - start_time

        eval_loss, eval_metrics = evaluate(model, eval_dataloader, device)

        print(f"\nEpoch {epoch + 1} Results:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Eval Loss: {eval_loss:.4f}")
        print(f"  Eval Accuracy: {eval_metrics['accuracy']:.4f}")
        print(f"  Time: {train_time:.1f}s")

    os.makedirs(config.output_dir, exist_ok=True)
    model_path = os.path.join(config.output_dir, 'bert_local_model')
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"\nModel saved to {model_path}")

if __name__ == "__main__":
    main()