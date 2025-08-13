import os
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
from torch.utils.data import DataLoader
import logging
from datetime import datetime
import json
from tqdm import tqdm

from config import TrainingConfig
from dataset_utils import load_and_prepare_dataset
from lora_model import create_lora_model, get_lora_state_dict
from evaluate import evaluate_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_directories(config):
    """Create output directories"""

    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.logging_dir, exist_ok=True)
    os.makedirs(f"{config.output_dir}/gradients", exist_ok=True)


def track_gradients(model, step, writer_dict):
    gradient_data ={
        'step': step,
        'lora_gradients': {},
        'base_gradients': {},
        'statistics': {}
    }

    lora_grads = []
    base_grads = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()

            if "lora_" in name:
                lora_grads.append(grad_norm)
                gradient_data['lora_gradients'][name] = grad_norm
            elif param.requires_grad:
                base_grads.append(grad_norm)
                gradient_data['base_gradients'][name] = grad_norm
    
    #Calculate Statistics
    if lora_grads:
        gradient_data['statistics']['lora_mean'] = sum(lora_grads)/len(lora_grads)
        gradient_data['statistics']['lora_max'] = max(lora_grads)
        gradient_data['statistics']['lora_min'] = min(lora_grads)

    if base_grads:
        gradient_data['statistics']['base_mean'] = sum(base_grads)/ len(base_grads)
        gradient_data['statistics']['base_max'] = max(base_grads)

    
    filename = f"{writer_dict['gradient_dir']}/gradient_step_{step}.json"
    with open(filename, 'w') as f:
        json.dump(gradient_data, f, indent =2)

    return gradient_data['statistics']

def train_on_tpu(index, config):
    #device setup
    device = xm.xla_device()
    xm.set_replication(device, [device])

    is_master = xm.is_master_ordinal()

    if is_master:
        logger.info(f"Starting traing on TPU core {index}")
        setup_directories(config)

    #Load data on ALL processes not just master
    train_dataset, eval_dataset, tokenizer = load_and_prepare_dataset(config)

    model = create_lora_model(config)
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = config.learning_rate,
        weight_decay = config.weight_decay
    )


    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True
    )

    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=config.per_device_train_batch_size,
        drop_last=True
    )

    train_device_loader = pl.MpDeviceLoader(train_dataloader, device)

    writer_dict = {
        'gradient_dir' : f"{config.output_dir}/gradients"
    }

    num_training_steps = len(train_dataloader) * config.num_train_epochs
    num_warmup_steps = config.warmup_steps

    if is_master:
        logger.info(f"Total training steps: {num_training_steps}")
        logger.info(f"Warmup steps: {num_warmup_steps}")

    
    # Training Loop
    global_step = 0

    for epoch in range(config.num_train_epochs):
        model.train()
        epoch_loss = 0

        if is_master:
            pbar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            
            #Forward Pass with autocast for bfloat16
            with torch.autocast('xla', dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss

            #backward pass
            loss.backward()

            if config.track_gradients and global_step % config.gradient_tracking_steps == 0:
                if is_master:
                    grad_stats = track_gradients(model, global_step, writer_dict)
                    logger.info(f"Step {global_step} gradient stats: {grad_stats}")

            
            xm.optimizer_step(optimizer)
            optimizer.zero_grad()

            #Update learning rate
            if global_step < num_warmup_steps:
                lr = config.learning_rate * global_step / num_warmup_steps
            else:
                progress = (global_step - num_warmup_steps) / (num_training_steps - num_warmup_steps)
                lr = config.learning_rate * 0.5 * (1+ torch.cos(torch.tensor(3.14159 * progress)))

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr.item() if hasattr(lr, 'item') else lr

            epoch_loss += loss.item()
            global_step += 1

            if is_master and step%10 == 0:
                pbar.update(10)
                pbar.set_postfix({'loss:': loss.item(), 'lr': lr})

        if is_master:
            pbar.close()
            avg_loss = epoch_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch} average loss: {avg_loss:.4f}")

            #Evaluate at end of epoch
            logger.info("Running evaluation")
            eval_metrics = evaluate_model(model, eval_dataset, config, device)
            logger.info(f"Epoch {epoch} eval metrics: {eval_metrics}")

            #Save checkpoint
            checkpoint_path = f"{config.output_dir}/checkpoint_epoch_{epoch}"
            os.makedirs(checkpoint_path, exist_ok=True)

            #Save LoRA weights only
            lora_state_dict = get_lora_state_dict(model)
            xm.save(lora_state_dict, f"{checkpoint_path}/adapter_model.bin")

            #Save configuration
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)

    if is_master:
        logger.info("Training Completed")

def main():
    config = TrainingConfig()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if config.use_tpu:
        xmp.spawn(train_on_tpu, args=(config,), nprocs=config.tpu_num_cores)
    else:
        train_on_tpu(0, config)
if __name__ =="__main__":
    main()






