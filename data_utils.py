from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch

def load_and_prepare_dataset(config):
    """Load and tokenize SST-2 dataset"""
    print("Loading SST-2 dataset")
    dataset = load_dataset('glue', config.dataset_name)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    def tokenize_function(examples):
        return tokenizer(
            examples['sentence'],
            padding = 'max_length',
            truncation = True,
            max_length=config.max_seq_length,
            return_tensors ='pt'
        )
    
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched = True,
        remove_columns =['idx', 'sentence']
    )

    if config.train_samples>0:
        train_dataset = tokenized_datasets['train'].select(range(config.train_samples))
    else:
        train_dataset = tokenized_datasets['train']
    
    eval_dataset = tokenized_datasets['validation']

    return train_dataset, eval_dataset, tokenizer

def create_dataloaders(train_dataset, eval_dataset, config):
    """Create PyTorch dataloaders"""
    train_dataloader = DataLoader(
        train_dataset,
        batch_size = config.per_device_train_batch_size,
        shuffle = True,
        drop_last=True
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config.per_device_eval_batch_size,
        drop_last = True
    )

    return train_dataloader, eval_dataloader

