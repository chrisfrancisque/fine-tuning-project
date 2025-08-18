from dataclasses import dataclass, asdict
from typing import Optional, Dict

@dataclass
class TrainingConfig:
    # Model / data
    model_name_or_path: str = "bert-base-uncased"
    baseline_state_dict_path: str = "baseline_model_seed42/pytorch_model.bin"
    task_name: str = "sst2"
    max_seq_length: int = 128

    # Batching / schedule
    per_core_train_batch_size: int = 8       # 8 per TPU core -> 64 global on v3-8
    per_core_eval_batch_size: int = 32
    train_samples_per_epoch: int = 5000      # target samples per epoch (pre-drop_last)
    num_epochs: int = 3
    learning_rate: float = 8e-6
    warmup_steps: int = 75
    weight_decay: float = 0.01

    # Repro
    seed: int = 42

    # I/O
    output_root: str = "checkpoints_gradual"  # a timestamped dir will be created under this
    save_tokenizer_once: bool = True

    # XLA runtime
    use_xla: bool = True

    # Misc
    notes: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)

PROFILES: Dict[str, TrainingConfig] = {}

# Your v3-8 plan
PROFILES["v3_8_gradual"] = TrainingConfig(
    per_core_train_batch_size=8,
    per_core_eval_batch_size=32,
    train_samples_per_epoch=5000,
    num_epochs=3,
    learning_rate=8e-6,
    warmup_steps=75,
    weight_decay=0.01,
    max_seq_length=128,
)

# An alternate, more aggressive profile (example; not used)
PROFILES["default_v6e"] = TrainingConfig(
    per_core_train_batch_size=16,
    per_core_eval_batch_size=64,
    train_samples_per_epoch=10000,
    num_epochs=3,
    learning_rate=2e-5,
    warmup_steps=500,
    weight_decay=0.01,
    max_seq_length=128,
)

def get_config(name: str) -> TrainingConfig:
    if name not in PROFILES:
        raise ValueError(f"Unknown profile '{name}'. Available: {list(PROFILES.keys())}")
    return PROFILES[name]
