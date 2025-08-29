"""
Configuration for BERT Full Fine-tuning
"""
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # Model Settings
    model_name: str = 'bert-base-uncased'
    num_labels: int = 2
    
    # Data Settings
    dataset_name: str = 'sst2'
    max_seq_length: int = 128
    train_samples: int = 10000  # Use -1 for full dataset
    
    # Training Settings
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    learning_rate: float = 2e-5
    num_train_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    
    # TPU Settings
    tpu_num_cores: int = 8
    use_tpu: bool = False
    
    @property
    def total_train_batch_size(self):
        if self.use_tpu:
            return self.per_device_train_batch_size * self.tpu_num_cores
        return self.per_device_train_batch_size

config = TrainingConfig()
