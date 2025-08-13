from datasets import load_dataset, DownloadConfig
from transformers import AutoTokenizer
import logging

logger = logging.getLogger(__name__)

def load_and_prepare_dataset(config):
    """Dataset loader with cache fix"""

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)


    download_config = DownloadConfig(
        forced_download=True,
        resume_download=False, 
        num_proc=1
    )

    # Load SST-2 dataset
    logger.info("Loading SST-2 dataset")
    dataset = load_dataset(
        'glue',
        'sst2',
        download_config=download_config,
        verification_mode='no_checks'
    )

    def tokenize_function(examples):
        return tokenizer(
            examples['sentence'],
            padding='max_length',
            truncation=True,
            max_length=config.max_seq_length
        )
    
    logger.info("Tokenizing datasets")
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns =['sentence', 'idx']
    )

    #Rename 'label' to 'labels' as bert expects 'labels'
    tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')

    tokenized_datasets.set_format(
        'torch',
        columns=['input_ids', 'attention_mask', 'labels']
    )

    train_dataset = tokenized_datasets['train']
    if config.train_samples > 0:
        train_dataset = train_dataset.select(range(config.train_samples))
    
    eval_dataset = tokenized_datasets['validation']

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Eval samples: {len(eval_dataset)}")

    return train_dataset, eval_dataset, tokenizer

