# configuration
import os
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # Model Settings

    model_name: str = 'bert-base-uncased'
    num_labels: int = 2

    #Data settings
    dataset_name: str = 'sst2'
    max_seq_length: int = 128
    train_samples: int = 10000

    #Training Settings
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 8
    learning_rate: float = 2e-5
    num_train_epochs: int = 3
    weight_decay: float = 0.01

    #TPU settings
    tpu_num_cores: int = 8

    #Paths
    output_dir: str = './outputs'
    logging_dir: str = './logs'

    #Environment
    use_tpu: bool = False
    use_mixed_precision: bool = True

    @property
    def total_train_batch_size(self):
        if self.use_tpu:
            return self.per_device_train_batch_size * self.tpu_num_cores
        return self.per_device_train_batch_size

config = TrainingConfig()