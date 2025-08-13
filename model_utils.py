from transformers import AutoModelForSequenceClassification
import torch
from sklearn.metrics import accuracy_score
import numpy as np
from load_baseline import load_baseline_model 

def create_model(config, use_baseline=True, device='cpu'):
    """Create BERT model for sequence classification"""
    
    if use_baseline:
        # Load the warmed baseline model
        print("Loading warmed baseline model from baseline_model_seed42")
        model, tokenizer, baseline_info = load_baseline_model(
            baseline_path='baseline_model_seed42',
            device=device
        )
        print(f"Starting from baseline with {baseline_info.get('warm_up_accuracy', 0):.4f} accuracy")
        return model, tokenizer, baseline_info
    else:
        # Original code path for testing/comparison
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=config.num_labels,
            id2label={0: "NEGATIVE", 1: "POSITIVE"},
            label2id={"NEGATIVE": 0, "POSITIVE": 1}
        )
        tokenizer = None
        baseline_info = {}
        return model, tokenizer, baseline_info

def compute_metrics(predictions, labels):
    """Compute accuracy metric"""
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": accuracy_score(labels, predictions)}