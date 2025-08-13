import torch
from transformers import AutoModelForSequenceClassification
from peft import (
    get_peft_model, 
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training
)
import logging

logger = logging.getLogger(__name__)

def create_lora_model(config):
    #Loading Model
    logger.info(f"Loading base model: {config.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=config.num_labels,
        torch_dtype=torch.float32
    )

    if config.use_lora:

        logger.info("Applying LoRA configuration")
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r = config.lora_config.r,
            lora_alpha = config.lora_config.lora_dropout,
            target_modules = config.lora_config.target_modules,
            bias = config.lora_config.bais,
            use_rslora=config.lora_config.use_rslora,
            modules_to_save=config.lora_config.modules_to_save
        )

        model = get_peft_model(model, lora_config)

        #print tranable parameter info
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        trainable_percent = 100 * trainable_params / all_params

        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"All parameters: {all_params:,}")
        logger.info(f"Percentage trainable: {trainable_percent:.4f}%")

        #print which modules are being trained
        logger.info("Modules being trained:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(f"  {name}: {param.shape}")
    
    return model


def print_model_architecture(model):
    """Helper function to understand model structure for debugging"""

    logger.info("\nModel architecture:")
    for name, module in model.names_modules():
        if len(list(module.children())) == 0: # Leaf modules only
            logger.info(f"  {name}: {module.__class__.__name__}")

def get_lora_state_dict(model):
    """Extract only LoRA parameters for saving"""
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if "lora_" in name or "modules_to_save" in name:
            lora_state_dict[name] = param.data.cpu()
    return lora_state_dict


