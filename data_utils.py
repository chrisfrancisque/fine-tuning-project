# Fixed data_utils.py
from datasets import load_dataset, DownloadConfig
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch

def load_and_prepare_dataset(config):
    """Load and tokenize SST-2 dataset with fix for ax dataset issue"""
    print("Loading SST-2 dataset with forced fresh download...")
    
    # Force fresh download to avoid cache issues
    download_config = DownloadConfig(
        force_download=True,
        resume_download=False,
        num_proc=1
    )
    
    try:
        # Primary method: Force fresh download
        dataset = load_dataset(
            'glue',
            'sst2',
            download_config=download_config,
            verification_mode='no_checks'
        )
        print("✓ Successfully loaded SST-2 dataset")
        
    except Exception as e:
        print(f"Primary method failed: {e}")
        print("Trying alternative method...")
        
        # Fallback method: Use alternative namespace
        dataset = load_dataset('nyu-mll/glue', 'sst2')
        print("✓ Successfully loaded SST-2 using alternative source")
    
    # Print dataset info
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Validation samples: {len(dataset['validation'])}")
    print(f"First sample: {dataset['train'][0]}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    def tokenize_function(examples):
        return tokenizer(
            examples['sentence'],
            padding='max_length',
            truncation=True,
            max_length=config.max_seq_length
            # Don't use return_tensors='pt' here
        )
    
    # Tokenize datasets
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['idx', 'sentence']
    )
    
    
    # Rename 'label' to 'labels' for BERT
    tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')
    
    # Set format for PyTorch
    tokenized_datasets.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

    # Select subset if specified
    if config.train_samples > 0:
        train_dataset = tokenized_datasets['train'].select(range(config.train_samples))
    else:
        train_dataset = tokenized_datasets['train']
    
    eval_dataset = tokenized_datasets['validation']

    return train_dataset, eval_dataset, tokenizer

def create_dataloaders(train_dataset, eval_dataset, config):
    """Create PyTorch dataloaders"""
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.per_device_train_batch_size,
        shuffle=True,
        drop_last=True
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config.per_device_eval_batch_size,
        drop_last=True
    )

    return train_dataloader, eval_dataloader