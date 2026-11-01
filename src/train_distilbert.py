# src/train_distilbert.py

import os
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve
)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "fake_job_postings.csv"
MODEL_DIR = PROJECT_ROOT / "models" / "distilbert"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_NAME = "distilbert-base-uncased"

TEXT_COLUMNS = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits"
]

MAX_LENGTH = 128
BATCH_SIZE = 4
EPOCHS = 1
LEARNING_RATE = 2e-5

# CPU pilot: maximum number of training records.
# Change to None later to use the entire training split.
MAX_TRAIN_SAMPLES = 2000

RANDOM_STATE = 42
DEVICE = torch.device("cpu")


# --------------------------------------------------
# REPRODUCIBILITY
# --------------------------------------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------
# DATASET CLASS
# --------------------------------------------------

class JobDataset(Dataset):

    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        self.labels = torch.tensor(
            labels,
            dtype=torch.long
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {
            key: value[index]
            for key, value in self.encodings.items()
        }

        item["labels"] = self.labels[index]

        return item


# --------------------------------------------------
# LOAD AND PREPARE DATA
# --------------------------------------------------

def load_data():

    print("\nLoading EMSCAD dataset...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = TEXT_COLUMNS + ["fraudulent"]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df = df.dropna(subset=["fraudulent"]).copy()

    df["fraudulent"] = df["fraudulent"].astype(int)

    # Combine the five text columns.
    df["combined_text"] = (
        df[TEXT_COLUMNS]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.strip()
    )

    df = df[df["combined_text"] != ""].copy()

    texts = df["combined_text"].tolist()
    labels = df["fraudulent"].tolist()

    print("Total records:", len(df))
    print("Genuine records:", labels.count(0))
    print("Fraudulent records:", labels.count(1))

    return texts, labels


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

# def evaluate_model(model, loader):

#     model.eval()

#     all_labels = []
#     all_probabilities = []

#     with torch.no_grad():

#         for batch in loader:

#             labels = batch["labels"].to(DEVICE)

#             inputs = {
#                 key: value.to(DEVICE)
#                 for key, value in batch.items()
#                 if key != "labels"
#             }

#             outputs = model(**inputs)

#             probabilities = torch.softmax(
#                 outputs.logits,
#                 dim=1
#             )[:, 1]

#             all_labels.extend(labels.cpu().numpy())
#             all_probabilities.extend(
#                 probabilities.cpu().numpy()
#             )

#     predictions = (
#         np.array(all_probabilities) >= 0.5
#     ).astype(int)

#     y_true = np.array(all_labels)

#     metrics = {
#         "accuracy": float(
#             accuracy_score(y_true, predictions)
#         ),
#         "precision": float(
#             precision_score(
#                 y_true, predictions, zero_division=0
#             )
#         ),
#         "recall": float(
#             recall_score(
#                 y_true, predictions, zero_division=0
#             )
#         ),
#         "f1_score": float(
#             f1_score(
#                 y_true, predictions, zero_division=0
#             )
#         ),
#         "pr_auc": float(
#             average_precision_score(
#                 y_true, all_probabilities
#             )
#         ),
#         "roc_auc": float(
#             roc_auc_score(
#                 y_true, all_probabilities
#             )
#         ),
#         "confusion_matrix": confusion_matrix(
#             y_true, predictions
#         ).tolist()
#     }

#     print("\nClassification Report:")
#     print(
#         classification_report(
#             y_true,
#             predictions,
#             target_names=["Genuine", "Fraudulent"],
#             zero_division=0
#         )
#     )

#     return metrics

def get_probabilities(model, loader):

    model.eval()

    all_labels = []
    all_probabilities = []

    with torch.no_grad():

        for batch in loader:

            labels = batch["labels"].to(DEVICE)

            inputs = {
                key: value.to(DEVICE)
                for key, value in batch.items()
                if key != "labels"
            }

            outputs = model(**inputs)

            probabilities = torch.softmax(
                outputs.logits,
                dim=1
            )[:, 1]

            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    return (
        np.array(all_labels),
        np.array(all_probabilities)
    )


def evaluate_model(model, loader, threshold=0.5):

    y_true, probabilities = get_probabilities(
        model, loader
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    metrics = {
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "precision": float(
            precision_score(
                y_true, predictions, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true, predictions, zero_division=0
            )
        ),
        "f1_score": float(
            f1_score(
                y_true, predictions, zero_division=0
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_true, probabilities
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true, probabilities
            )
        ),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(
            y_true, predictions
        ).tolist()
    }

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            predictions,
            target_names=["Genuine", "Fraudulent"],
            zero_division=0
        )
    )

    return metrics


def find_best_threshold(model, validation_loader):

    y_true, probabilities = get_probabilities(
        model, validation_loader
    )

    precision, recall, thresholds = (
        precision_recall_curve(
            y_true, probabilities
        )
    )

    # Calculate F1 for every available threshold.
    f1_values = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-10)
    )

    best_index = np.argmax(f1_values)

    best_threshold = float(
        thresholds[best_index]
    )

    print("\nValidation threshold selection:")
    print("Best threshold:", best_threshold)
    print("Validation F1:", f1_values[best_index])
    print("Validation precision:", precision[best_index])
    print("Validation recall:", recall[best_index])

    return best_threshold

# --------------------------------------------------
# MAIN TRAINING FUNCTION
# --------------------------------------------------

def main():

    set_seed(RANDOM_STATE)

    torch.set_num_threads(
        min(4, os.cpu_count() or 1)
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("DISTILBERT FAKE JOB DETECTION")
    print("=" * 55)

    print("Device:", DEVICE)
    print("Model:", MODEL_NAME)

    # Load dataset
    texts, labels = load_data()

    # Create a stratified split.
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=labels
    )

    print("\nFull training split:", len(X_train))
    print("Testing split:", len(X_test))

    # Use a stratified training pilot.
    if (
        MAX_TRAIN_SAMPLES is not None
        and MAX_TRAIN_SAMPLES < len(X_train)
    ):

        X_train, _, y_train, _ = train_test_split(
            X_train,
            y_train,
            train_size=MAX_TRAIN_SAMPLES,
            random_state=RANDOM_STATE,
            stratify=y_train
        )

        # Separate training and validation data.
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=y_train
        )

        print("\nActual training records:", len(X_fit))
        print("Validation records:", len(X_val))
        print("Testing records:", len(X_test))

        print(
            "Pilot training records:",
            len(X_train)
        )

    # Load tokenizer
    print("\nLoading DistilBERT tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    # Prepare datasets
    # Prepare datasets

    print("\nTokenizing training data...")

    train_dataset = JobDataset(
        X_fit, y_fit, tokenizer
    )

    print("Tokenizing validation data...")

    val_dataset = JobDataset(
        X_val, y_val, tokenizer
    )

    print("Tokenizing testing data...")

    test_dataset = JobDataset(
        X_test, y_test, tokenizer
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    # Load pretrained model
    print("\nLoading pretrained DistilBERT model...")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )

    model.to(DEVICE)

    # Calculate class weights for imbalance.
    class_counts = np.bincount(
        np.array(y_fit),
        minlength=2
    )

    class_weights = len(y_fit) / (
        2.0 * class_counts
    )

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32
    ).to(DEVICE)

    loss_function = torch.nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    total_steps = len(train_loader) * EPOCHS

    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.1,
        total_iters=max(total_steps, 1)
    )

    # Training
    print("\nStarting CPU training...")
    print("Training batches:", len(train_loader))

    start_time = time.time()

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0.0

        for step, batch in enumerate(train_loader):

            labels_batch = batch["labels"].to(DEVICE)

            inputs = {
                key: value.to(DEVICE)
                for key, value in batch.items()
                if key != "labels"
            }

            optimizer.zero_grad()

            outputs = model(**inputs)

            loss = loss_function(
                outputs.logits,
                labels_batch
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if (step + 1) % 25 == 0:
                print(
                    f"Epoch {epoch + 1}/{EPOCHS} | "
                    f"Batch {step + 1}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f}"
                )

        average_loss = total_loss / len(train_loader)

        print(
            f"\nEpoch {epoch + 1} completed. "
            f"Average loss: {average_loss:.4f}"
        )

    training_time = time.time() - start_time

    # Evaluate
    # Select threshold using validation data.
    print("\nFinding best classification threshold...")

    best_threshold = find_best_threshold(
        model,
        val_loader
    )

    # Evaluate only once on the untouched test set.
    print("\nEvaluating DistilBERT on test data...")

    metrics = evaluate_model(
        model,
        test_loader,
        threshold=best_threshold
    )

    metrics["training_records"] = len(X_fit)
    metrics["validation_records"] = len(X_val)
    metrics["testing_records"] = len(X_test)
    metrics["epochs"] = EPOCHS
    metrics["max_length"] = MAX_LENGTH
    metrics["training_time_seconds"] = round(
        training_time, 2
    )
    metrics["training_mode"] = "CPU pilot"
    metrics["training_sample_limit"] = MAX_TRAIN_SAMPLES

    # Save model and tokenizer
    print("\nSaving model...")

    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    metrics_path = (
        REPORT_DIR / "distilbert_metrics.json"
    )

    with open(metrics_path, "w") as file:
        json.dump(metrics, file, indent=4)

    print("\n" + "=" * 55)
    print("DISTILBERT TRAINING COMPLETED")
    print("=" * 55)

    for name, value in metrics.items():
        print(f"{name}: {value}")

    print("\nModel saved to:", MODEL_DIR)
    print("Metrics saved to:", metrics_path)


if __name__ == "__main__":
    main()