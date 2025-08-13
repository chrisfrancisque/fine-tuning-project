from dataclasses import dataclass
from typing import Optional, List

@dataclass
class LoRAConfig:
    """Lora-specific configuration parameters"""

    #Core Lora parameters
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1

    
    #Which layers to apply LoRA to
    target_modules: List[str] = None # Will default to ["query", "value"]

    #Advanced options
    bias: str = "none"
    modules_to_save: List[str] = None
    use_rslora: bool = True

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["query", "value"]
        if self.modules_to_save is None:
            self.modules_to_save = ["classifier"]

@dataclass
class TrainingConfig:
    """Training Configurations"""

    model_name: str = 'bert-base-uncased'
    num_labels: int = 2
    dataset_name: str = 'sst2'
    max_seq_length: int = 128
    train_samples: int = 100

    per_device_train_batch_size: int = 128
    per_device_eval_batch_size: int = 16

    learning_rate: float = 5e-4
    num_train_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01

    # TPU settings
    tpu_num_cores: int = 8
    output_dir: str = './outputs'
    logging_dir: str = './logs'
    use_tpu: bool = True

    use_lora: bool = True
    lora_config: LoRAConfig = None

    track_gradients: bool = True
    gradient_tracking_steps: int = 100

    def __post_init__(self):
        if self.lora_config is None:
            self.lora_config = LoRAConfig()

    
    @property
    def total_train_batch_size(self):
        if self.use_tpu:
            return self.per_device_train_batch_size * self.tpu_num_cores
        return self.per_device_train_batch_size

