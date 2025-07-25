from transformers import AutoModelForSequenceClassification
import torch
from sklearn.metrics import accuracy_score
import numpy as np

def create_model(config):
    """Creat BERT model for sequence classification"""
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=config.num_labels,
        id2label ={0: "NEGATIVE", 1: "POSITIVE"},
        label2id = {"NEGATIVE": 0, "POSITIVE": 1}
    )
    return model

def compute_metrics(predictions, labels):
    """Compute accuracy metric"""
    predictions = np.argmax(predictions, axis =1)
    return {"accuracy": accuracy_score(labels, predictions)}
