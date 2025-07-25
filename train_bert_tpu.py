import os
import torch

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

def train_bert_on_tpu(index):
    """Training function for each TPU core"""
    # Set config for TPU
    config.use_tpu = True
    
    # Get TPU device
    device = xm.xla_device()
    
    # Only print from master
    if xm.is_master_ordinal():
        print(f"Starting training on TPU core {index}")
        print(f"Total batch size: {config.total_train_batch_size}")
    
    # Load data (only on master to avoid duplication)
    if xm.is_master_ordinal():
        train_dataset, eval_dataset, tokenizer = load_and_prepare_dataset(config)
    
    # Synchronize to ensure data is loaded
    xm.rendezvous("data_loading")
    
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
    
    # Training loop
    model.train()
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
            
            total_loss += loss.item()
            
            # Log progress
            if step % 50 == 0:
                xm.master_print(
                    f"Epoch {epoch}, Step {step}/{len(train_dataloader)}, "
                    f"Loss: {loss.item():.4f}"
                )
        
        # End of epoch
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / len(train_dataloader)
        xm.master_print(
            f"Epoch {epoch} completed in {epoch_time:.1f}s, "
            f"Average Loss: {avg_loss:.4f}"
        )
        
        # Evaluation (optional, can be slow on TPU)
        if epoch == config.num_train_epochs - 1:  # Only evaluate at the end
            model.eval()
            eval_loss = 0
            
            with torch.no_grad():
                for batch in eval_device_loader:
                    outputs = model(**batch)
                    eval_loss += outputs.loss.item()
            
            avg_eval_loss = eval_loss / len(eval_dataloader)
            xm.master_print(f"Evaluation Loss: {avg_eval_loss:.4f}")
            
            model.train()
    
    # Save model (only from master)
    if xm.is_master_ordinal():
        os.makedirs(config.output_dir, exist_ok=True)
        model_path = os.path.join(config.output_dir, 'bert_tpu_model')
        xm.save(model.state_dict(), f"{model_path}.pt")
        print(f"Model saved to {model_path}.pt")

def main():
    if not TPU_AVAILABLE:
        print("Error: TPU libraries not available!")
        print("This script must be run on a TPU VM.")
        return
    
    # Launch training on all TPU cores
    xmp.spawn(train_bert_on_tpu, args=())

if __name__ == "__main__":
    main()
