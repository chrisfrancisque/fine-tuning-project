# BERT Full Fine-Tuning with Checkpoint Analysis

A comprehensive implementation of BERT-base full parameter fine-tuning (110M parameters) on SST-2, with checkpoint saving for comparative analysis with LoRA and MFT approaches.

## 🎯 Project Overview

This project is part of a three-way comparison study:
- **Full Fine-Tuning** (this repo): Updates all 109.5M parameters
- **[LoRA](https://github.com/chrisfrancisque/bert-lora-tpu)**: Updates only ~40K parameters
- **[MFT (Mask Fine-Tuning)](https://github.com/chrisfrancisque/BERT-MFT)**: Zeros ~4M parameters without training

All methods start from the same warmed baseline model (~55% accuracy) for fair comparison.

## 📊 Key Features

- ✅ Warmed baseline model with pre-trained classifier
- ✅ Checkpoint saving after each epoch
- ✅ TPU v5/v6e optimized implementation
- ✅ Comprehensive metrics and visualizations
- ✅ Compatible with MFT checkpoint analysis
- ✅ Local and cloud deployment options

## 🏗️ Project Structure

bert-tpu-project/
├── baseline_model_seed42/      # Warmed baseline model (shared across all methods)
│   ├── pytorch_model.bin       # Model weights (Git LFS)
│   ├── config.json             # Model configuration
│   └── baseline_info.json      # Training info
├── config.py                   # Centralized configuration
├── data_utils.py              # Dataset loading and preprocessing
├── model_utils.py             # Model creation and metrics
├── train_bert_local.py        # Local training script
├── train_bert_tpu.py          # TPU training script (with checkpointing)
├── load_baseline.py           # Baseline model loader
├── load_checkpoint.py         # Checkpoint loading utilities
├── create_baseline.py         # Generate warmed baseline
├── requirements.txt           # Python dependencies
└── README.md                  # This file

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or 3.11 (NOT 3.13 - PyTorch XLA incompatible)
- 16GB+ RAM for local training
- Google Cloud account with TPU quota (for TPU training)

### Local Setup

```bash
# Clone repository
git clone https://github.com/chrisfrancisque/bert-tpu-project.git
cd bert-tpu-project

# Create virtual environment (use Python 3.10)
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Pull baseline model (Git LFS)
git lfs pull

# Verify baseline model
python load_baseline.py

# Test with small dataset (100 samples, 1 epoch)
python train_bert_local.py --test

# Train for 3 epochs with checkpointing
python train_bert_local.py

# Results will be in: outputs/results_YYYYMMDD_HHMMSS/
# Checkpoints in: outputs/results_*/checkpoint_epoch_*/

# Set environment
export PROJECT_ID=your-project-id
export ZONE=us-east1-d  # or us-central1-a for v5e
export TPU_NAME=bert-experiments

# Create TPU v6e (latest generation)
gcloud compute tpus tpu-vm create $TPU_NAME \
  --project=$PROJECT_ID \
  --zone=$ZONE \
  --accelerator-type=v6e-8 \
  --version=tpu-ubuntu2204-base \
  --preemptible  # 70% cost savings

  # SSH into TPU
gcloud compute tpus tpu-vm ssh $TPU_NAME \
  --project=$PROJECT_ID \
  --zone=$ZONE

# On TPU VM, setup environment
sudo apt update && sudo apt install -y python3.10-venv git
python3.10 -m venv ~/tpu-env
source ~/tpu-env/bin/activate

# Install PyTorch XLA
pip install --upgrade pip
pip install torch==2.7.0
pip install 'torch_xla[tpu]==2.7.0' \
  -f https://storage.googleapis.com/libtpu-wheels/index.html

# Clone and setup project
git clone https://github.com/chrisfrancisque/bert-tpu-project.git
cd bert-tpu-project
pip install -r requirements.txt

# Pull baseline model
git lfs pull


# Set environment variables
export PJRT_DEVICE=TPU
export XLA_USE_BF16=1
export TORCH_COMPILE_DISABLE=1

# Verify TPU access
python -c "import torch_xla.core.xla_model as xm; print(f'TPU devices: {xm.xla_real_devices()}')"

# Run training with checkpoints
python train_bert_tpu.py

# Monitor progress
tail -f outputs/results_*/logs/*.log

# Download all results and checkpoints
gcloud compute tpus tpu-vm scp --recurse \
  $TPU_NAME:~/bert-tpu-project/results_* \
  ./tpu_results/ \
  --project=$PROJECT_ID \
  --zone=$ZONE

  # Delete TPU when done (IMPORTANT - avoid charges)
gcloud compute tpus tpu-vm delete $TPU_NAME \
  --project=$PROJECT_ID \
  --zone=$ZONE

  @dataclass
class TrainingConfig:
    model_name: str = 'bert-base-uncased'
    num_labels: int = 2
    dataset_name: str = 'sst2'
    max_seq_length: int = 128
    train_samples: int = 10000  # Use -1 for full dataset
    per_device_train_batch_size: int = 16
    learning_rate: float = 2e-5
    num_train_epochs: int = 3
    warmup_steps: int = 500

    torch==2.7.0
transformers==4.36.0
datasets==2.14.0
numpy==1.24.3  # Important: Use 1.x, not 2.x
scikit-learn>=1.3.0
matplotlib>=3.5.0
seaborn>=0.12.0
tqdm
pyarrow==11.0.0
fsspec==2023.5.0

🐛 Troubleshooting
Issue: AttributeError: module 'torch._dynamo'
Solution: Use Python 3.10, not 3.13

Issue: NumPy 1.x cannot run in NumPy 2.x
Solution: pip install numpy==1.24.3

Issue: File too large for GitHub
Solution: Use Git LFS: git lfs track "*.bin"

Issue: TPU not available
Solution: Try different zones

Issue: Dataset loading error
Solution: Clear cache: rm -rf ~/.cache/huggingface

TPU-Specific Tips

Always use drop_last=True for consistent batch sizes
Use xm.optimizer_step() instead of optimizer.step()
Monitor with xm.master_print() to avoid duplicate outputs
Set TORCH_COMPILE_DISABLE=1 to avoid compilation hangs

📊 Outputs
Each training run creates:

results_YYYYMMDD_HHMMSS/
├── checkpoint_epoch_0/         # Model checkpoint
│   ├── pytorch_model.bin      # Model weights
│   ├── config.json            # Model config
│   ├── checkpoint_info.json   # Training metrics
│   └── tokenizer files        # Tokenizer
├── checkpoint_epoch_1/
├── checkpoint_epoch_2/
├── final_model/               # Final trained model
├── training_summary.json      # All metrics
├── training_loss.png         # Loss curve
├── confusion_matrix.png      # Final confusion matrix
└── metrics.json              # Final metrics

 Related Projects

BERT-LoRA: Parameter-efficient fine-tuning
BERT-MFT: Mask Fine-Tuning without training