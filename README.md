(READ THIS IN CODE)

BERT Fine-tuning on Google Cloud TPU v3-8
A comprehensive implementation of BERT-base fine-tuning using Google Cloud TPUs.

Overview
This project implements full parameter fine-tuning of BERT-base (110M parameters) on the Stanford Sentiment Treebank (SST-2) dataset using Google Cloud TPU v3-8. The implementation showcases the migration from local development to cloud-based distributed training, handling real-world challenges in dependency management, TPU optimization, and cost-effective resource utilization.

Table of Contents

Project Structure
Requirements
Google Cloud TPU Setup
Code Architecture
Troubleshooting Guide
Lessons Learned


Project Structure

bert-tpu-project/
├── config.py                 # Centralized configuration
├── data_utils.py            # Dataset loading and preprocessing
├── model_utils.py           # Model creation and metrics
├── train_bert_local.py      # Local training script
├── train_bert_tpu.py        # TPU training script
├── train_sst2_working.py    # Fallback training with synthetic data
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore patterns
└── README.md               # This file


Software Requirements

Python 3.10 (Critical: PyTorch XLA 2.7.0 doesn't support Python 3.13)
Google Cloud SDK (for TPU management)
Git (for version control)
torch==2.7.0
transformers==4.36.0
datasets==2.14.0
numpy==1.24.3  # Important: Use 1.x, not 2.x
pyarrow==11.0.0
pandas==2.0.3
scikit-learn==1.3.0
tqdm
evaluate
tensorboard


Create Virtual Environment
bash# Use Python 3.10 for compatibility
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install Dependencies
bashpip install --upgrade pip
pip install -r requirements.txt



Google Cloud TPU Setup
Prerequisites

Google Cloud account with billing enabled
Sufficient quota for TPU v3-8 in your preferred region
gcloud CLI installed and configured

1. Enable Required APIs
bashgcloud services enable tpu.googleapis.com compute.googleapis.com storage-component.googleapis.com
2. Create TPU VM
Option A: Direct Creation
bashexport TPU_NAME=bert-tpu-v3
export ZONE=europe-west4-a  # Better availability than US zones
export PROJECT_ID=your-project-id

gcloud compute tpus tpu-vm create $TPU_NAME \
  --project=$PROJECT_ID \
  --zone=$ZONE \
  --accelerator-type=v3-8 \
  --version=tpu-ubuntu2204-base \
  --preemptible  # 70% cost savings
Option B: Queued Resources (If Capacity Unavailable)
bashexport QUEUED_RESOURCE_ID=bert-tpu-queue-$(date +%s)

gcloud compute tpus queued-resources create $QUEUED_RESOURCE_ID \
  --node-id=$TPU_NAME \
  --project=$PROJECT_ID \
  --zone=$ZONE \
  --accelerator-type=v3-8 \
  --runtime-version=tpu-ubuntu2204-base

Monitor status
watch -n 60 "gcloud compute tpus queued-resources describe $QUEUED_RESOURCE_ID \
  --project=$PROJECT_ID --zone=$ZONE --format='value(state.state)'"
3. SSH to TPU VM
bashgcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE
4. Set Up TPU Environment
bash# Install Python 3.10 venv (if needed)
sudo apt update
sudo apt install python3.10-venv -y

Create virtual environment
python3.10 -m venv ~/tpu-env
source ~/tpu-env/bin/activate

Install PyTorch XLA for TPU
pip install --upgrade pip
pip install torch==2.7.0
pip install 'torch_xla[tpu]==2.7.0' \
  -f https://storage.googleapis.com/libtpu-wheels/index.html \
  -f https://storage.googleapis.com/libtpu-releases/index.html

Install other dependencies
pip install transformers==4.36.0 datasets==2.14.0 numpy==1.24.3
pip install scikit-learn tqdm tensorboard

Set environment variables
export PJRT_DEVICE=TPU
export XLA_USE_BF16=1

Verify TPU access
python -c "import torch_xla.core.xla_model as xm; print(f'TPU devices: {xm.xla_real_devices()}')"
Expected output:
TPU devices: ['TPU:0', 'TPU:1', 'TPU:2', 'TPU:3', 'TPU:4', 'TPU:5', 'TPU:6', 'TPU:7']

Transfer Code to TPU
From your local machine:
bash# Option 1: Using gcloud (if installed)
gcloud compute tpus tpu-vm scp *.py $TPU_NAME:~/ --zone=$ZONE

Option 2: Using git (on TPU VM)
git clone https://github.com/chrisfrancisque/fine-tuning-project.git
cd fine-tuning-project

6. Run Training on TPU
bashpython train_bert_tpu.py
🏗️ Code Architecture
Configuration Module (config.py)
python@dataclass
class TrainingConfig:
    model_name: str = 'bert-base-uncased'
    num_labels: int = 2
    dataset_name: str = 'sst2'
    max_seq_length: int = 128
    train_samples: int = 10000  # Subset for faster training
    per_device_train_batch_size: int = 32
    learning_rate: float = 2e-5
    num_train_epochs: int = 3
    
    
    
Data Pipeline (data_utils.py)

Loads SST-2 dataset from GLUE benchmark
Handles tokenization with proper padding for TPU (static shapes)
Implements fallback to synthetic data for robustness
Ensures drop_last=True for consistent batch sizes

Local Training (train_bert_local.py)

Standard PyTorch training loop
CPU/GPU compatible
Progress bars with tqdm
Evaluation after each epoch

TPU Training (train_bert_tpu.py)
Key TPU-specific changes:
pythonimport torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

Multi-core execution
xmp.spawn(train_function, args=())

TPU-optimized data loading
device_loader = pl.MpDeviceLoader(dataloader, device)

XLA-aware optimizer step
xm.optimizer_step(optimizer)


Troubleshooting Guide
Issue 1: Python Version Incompatibility
Error: AttributeError: module 'torch._dynamo' has no attribute 'external_utils'
Solution: Use Python 3.10 or 3.11, not 3.13
bashpython3.10 -m venv venv
source venv/bin/activate

Issue 2: NumPy 2.x Compatibility
Error: A module compiled using NumPy 1.x cannot be run in NumPy 2.x
Solution: Downgrade to NumPy 1.x
bashpip install numpy==1.24.3

Issue 3: Dataset Loading Errors
Error: ValueError: Invalid pattern: '**' can only be an entire path component
Solution: Update or downgrade dependencies
bashpip install datasets==2.12.0 pyarrow==11.0.0 fsspec==2023.10.0

Issue 4: TPU Availability
Error: There is no more capacity in the zone
Solution:

Try different zones (europe-west4-a often has better availability)
Use queued resources
Try preemptible instances
Check availability during off-peak hours

Issue 5: GLUE Dataset Cache Issues
Error: NotImplementedError: Loading a dataset cached in a LocalFileSystem is not supported
Solution: Clear cache and use fallback
bashrm -rf ~/.cache/huggingface/datasets


Lessons Learned
1. TPU Programming Patterns

Always use drop_last=True for consistent batch sizes
Static shapes are mandatory (no dynamic padding)
Use xm.optimizer_step() instead of optimizer.step()
Multi-core execution requires xmp.spawn()

2. Performance Insights

TPUs excel with large batch sizes (256 vs 32)
Compilation overhead is amortized over many steps
Data loading can be a bottleneck - use pl.MpDeviceLoader
BF16 precision enabled by default on TPU

3. Cost Management

Preemptible instances offer huge savings with minimal risk
Europe regions often have better availability
Queued resources prevent manual waiting
Always set deletion reminders

4. Development Workflow

Test thoroughly locally before TPU deployment
Keep synthetic data fallbacks for robustness
Version pin all dependencies
Document environment setup meticulously

5. Debugging Tips

TPU errors often appear on all cores (8x same error)
Use xm.master_print() to avoid duplicate outputs
Check shapes with tensor.shape before operations
Monitor memory with xm.get_memory_info()




