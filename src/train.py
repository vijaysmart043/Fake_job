
import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocessing import load_dataset, prepare_features


RANDOM_STATE = 42

MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


def main():
    print("Loading dataset...")

    df = load_dataset()
    df = prepare_features(df)

    X = df["combined_text"]
    y = df["fraudulent"]

    # Stratified split preserves the fraud/genuine ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    # Baseline: TF-IDF + Logistic Regression
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=30000,
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True,
                stop_words="english",
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ])

    print("\nTraining baseline model...")
    model.fit(X_train, y_train)

    print("Evaluating model...")
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": "TF-IDF + Logistic Regression",
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1_score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "pr_auc": average_precision_score(y_test, probabilities),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }

    # Save trained model
    model_path = MODEL_DIR / "baseline_model.joblib"
    joblib.dump(model, model_path)

    # Save evaluation metrics
    metrics_path = REPORT_DIR / "baseline_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("\nBASELINE MODEL RESULTS")
    print("-" * 40)

    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name}: {value:.4f}")
        else:
            print(f"{name}: {value}")

    print("\nModel saved:", model_path)
    print("Metrics saved:", metrics_path)


if __name__ == "__main__":
    main()
