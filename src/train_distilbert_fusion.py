
import json
import time
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel

from scipy.sparse import csr_matrix, hstack

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, normalize
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    classification_report
)

# ------------------------------------------------
# CONFIGURATION
# ------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "fake_job_postings.csv"
ENCODER_PATH = ROOT / "models" / "distilbert"
OUTPUT_PATH = ROOT / "models" / "distilbert_fusion.joblib"
REPORT_PATH = ROOT / "reports" / "distilbert_fusion_metrics.json"

TEXT_COLUMNS = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits"
]

CAT_COLUMNS = [
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
    "department",
    "location",
    "salary_range"
]

NUM_COLUMNS = [
    "telecommuting",
    "has_company_logo",
    "has_questions"
]

MAX_LENGTH = 128
BATCH_SIZE = 8
MAX_TRAIN_SAMPLES = 2000
RANDOM_STATE = 42

DEVICE = torch.device("cpu")


# ------------------------------------------------
# REPRODUCIBILITY
# ------------------------------------------------

def set_seed():

    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)


# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------

def load_data():

    print("\nLoading dataset...")

    df = pd.read_csv(DATA_PATH)

    required = (
        TEXT_COLUMNS
        + CAT_COLUMNS
        + NUM_COLUMNS
        + ["fraudulent"]
    )

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing dataset columns: {missing}"
        )

    df = df.dropna(subset=["fraudulent"]).copy()
    df["fraudulent"] = df["fraudulent"].astype(int)

    df["combined_text"] = (
        df[TEXT_COLUMNS]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
    )

    df["combined_text"] = (
        df["combined_text"].str.strip()
    )

    df = df[df["combined_text"] != ""].copy()

    # Preserve the raw structured attributes.
    X = df[
        ["combined_text"] + CAT_COLUMNS + NUM_COLUMNS
    ].copy()

    y = df["fraudulent"].to_numpy()

    return X, y


# ------------------------------------------------
# DISTILBERT EMBEDDING EXTRACTION
# ------------------------------------------------

def extract_embeddings(texts, tokenizer, encoder):

    print("\nExtracting DistilBERT embeddings...")

    all_embeddings = []

    encoder.eval()

    dataset = list(texts)

    for start in range(0, len(dataset), BATCH_SIZE):

        batch_texts = dataset[
            start:start + BATCH_SIZE
        ]

        tokens = tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        tokens = {
            key: value.to(DEVICE)
            for key, value in tokens.items()
        }

        with torch.no_grad():

            outputs = encoder(**tokens)

            hidden = outputs.last_hidden_state

            mask = (
                tokens["attention_mask"]
                .unsqueeze(-1)
                .expand(hidden.size())
                .float()
            )

            # Mean pooling, excluding padding tokens.
            summed = torch.sum(
                hidden * mask,
                dim=1
            )

            counts = torch.clamp(
                mask.sum(dim=1),
                min=1e-9
            )

            embeddings = summed / counts

        all_embeddings.append(
            embeddings.cpu().numpy()
        )

        if (
            (start // BATCH_SIZE + 1) % 50 == 0
            or start + BATCH_SIZE >= len(dataset)
        ):
            print(
                f"Processed {min(start + BATCH_SIZE, len(dataset))}"
                f"/{len(dataset)}"
            )

    return np.vstack(all_embeddings)


# ------------------------------------------------
# STRUCTURED FEATURE PROCESSING
# ------------------------------------------------

def create_preprocessor():

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=5
            )
        )
    ])

    numerical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ])

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_pipeline,
                CAT_COLUMNS
            ),
            (
                "numerical",
                numerical_pipeline,
                NUM_COLUMNS
            )
        ],
        sparse_threshold=1.0
    )


# ------------------------------------------------
# THRESHOLD SELECTION
# ------------------------------------------------

def select_threshold(y_true, probabilities):

    precision, recall, thresholds = (
        precision_recall_curve(
            y_true,
            probabilities
        )
    )

    f1_values = (
        2 * precision[:-1] * recall[:-1]
        / (
            precision[:-1]
            + recall[:-1]
            + 1e-10
        )
    )

    best_index = np.argmax(f1_values)

    threshold = float(
        thresholds[best_index]
    )

    print("\nValidation threshold:")
    print("Threshold:", threshold)
    print("Validation F1:", f1_values[best_index])

    return threshold


# ------------------------------------------------
# EVALUATION
# ------------------------------------------------

def evaluate(y_true, probabilities, threshold):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    metrics = {
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),
        "f1_score": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_true,
                probabilities
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities
            )
        ),
        "threshold": threshold,
        "confusion_matrix": confusion_matrix(
            y_true,
            predictions
        ).tolist()
    }

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            predictions,
            target_names=[
                "Genuine",
                "Fraudulent"
            ],
            zero_division=0
        )
    )

    return metrics


# ------------------------------------------------
# MAIN
# ------------------------------------------------

def main():

    set_seed()

    torch.set_num_threads(4)

    print("=" * 55)
    print("DISTILBERT + STRUCTURED FEATURE FUSION")
    print("=" * 55)

    if not ENCODER_PATH.exists():
        raise FileNotFoundError(
            "Train DistilBERT first. Encoder folder not found."
        )

    ROOT.joinpath("models").mkdir(
        parents=True,
        exist_ok=True
    )

    ROOT.joinpath("reports").mkdir(
        parents=True,
        exist_ok=True
    )

    X, y = load_data()

    # Same outer split as the DistilBERT experiment.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Reproduce the 2,000-record pilot.
    if MAX_TRAIN_SAMPLES < len(X_train):

        X_train, _, y_train, _ = train_test_split(
            X_train,
            y_train,
            train_size=MAX_TRAIN_SAMPLES,
            random_state=RANDOM_STATE,
            stratify=y_train
        )

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_train
    )

    print("\nTraining records:", len(X_fit))
    print("Validation records:", len(X_val))
    print("Testing records:", len(X_test))

    # Load the fine-tuned DistilBERT encoder.
    print("\nLoading fine-tuned DistilBERT encoder...")

    tokenizer = AutoTokenizer.from_pretrained(
        str(ENCODER_PATH)
    )

    encoder = AutoModel.from_pretrained(
        str(ENCODER_PATH)
    )

    encoder.to(DEVICE)

    # Extract embeddings separately.
    start_time = time.time()

    emb_train = extract_embeddings(
        X_fit["combined_text"].tolist(),
        tokenizer,
        encoder
    )

    emb_val = extract_embeddings(
        X_val["combined_text"].tolist(),
        tokenizer,
        encoder
    )

    emb_test = extract_embeddings(
        X_test["combined_text"].tolist(),
        tokenizer,
        encoder
    )

    print("\nEmbedding shape:", emb_train.shape)

    # Normalize text embeddings.
    emb_train = normalize(emb_train)
    emb_val = normalize(emb_val)
    emb_test = normalize(emb_test)

    # Fit structured feature preprocessing on training data only.
    print("\nProcessing structured features...")

    preprocessor = create_preprocessor()

    structured_train = preprocessor.fit_transform(
        X_fit[CAT_COLUMNS + NUM_COLUMNS]
    )

    structured_val = preprocessor.transform(
        X_val[CAT_COLUMNS + NUM_COLUMNS]
    )

    structured_test = preprocessor.transform(
        X_test[CAT_COLUMNS + NUM_COLUMNS]
    )

    # Combine text and structured features.
    fusion_train = hstack([
        csr_matrix(emb_train),
        structured_train
    ]).tocsr()

    fusion_val = hstack([
        csr_matrix(emb_val),
        structured_val
    ]).tocsr()

    fusion_test = hstack([
        csr_matrix(emb_test),
        structured_test
    ]).tocsr()

    print(
        "\nCombined feature shape:",
        fusion_train.shape
    )

    # Train classifier.
    print("\nTraining fusion classifier...")

    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=1500,
        solver="liblinear",
        random_state=RANDOM_STATE
    )

    classifier.fit(
        fusion_train,
        y_fit
    )

    # Select threshold on validation data.
    val_probabilities = classifier.predict_proba(
        fusion_val
    )[:, 1]

    threshold = select_threshold(
        y_val,
        val_probabilities
    )

    # Final test evaluation.
    test_probabilities = classifier.predict_proba(
        fusion_test
    )[:, 1]

    metrics = evaluate(
        y_test,
        test_probabilities,
        threshold
    )

    metrics["training_records"] = len(X_fit)
    metrics["validation_records"] = len(X_val)
    metrics["testing_records"] = len(X_test)
    metrics["embedding_dimension"] = int(
        emb_train.shape[1]
    )
    metrics["training_mode"] = "CPU feature-level fusion"
    metrics["max_length"] = MAX_LENGTH
    metrics["training_time_seconds"] = round(
        time.time() - start_time,
        2
    )

    # Save classifier and structured preprocessor.
    joblib.dump(
        {
            "classifier": classifier,
            "preprocessor": preprocessor,
            "threshold": threshold,
            "text_columns": TEXT_COLUMNS,
            "categorical_columns": CAT_COLUMNS,
            "numeric_columns": NUM_COLUMNS,
            "max_length": MAX_LENGTH
        },
        OUTPUT_PATH
    )

    with open(REPORT_PATH, "w") as file:
        json.dump(metrics, file, indent=4)

    print("\n" + "=" * 55)
    print("DISTILBERT FUSION COMPLETED")
    print("=" * 55)

    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nSaved model:", OUTPUT_PATH)
    print("Saved metrics:", REPORT_PATH)


if __name__ == "__main__":
    main()
