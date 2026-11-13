
import json
from pathlib import Path

import joblib
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier

from src.preprocessing import (
    load_dataset,
    prepare_features,
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
)


RANDOM_STATE = 42

MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


def main():

    print("Loading dataset...")

    df = load_dataset()
    df = prepare_features(df)

    # Use combined text and structured features
    X = df[
        ["combined_text"] + CATEGORICAL_COLUMNS + NUMERIC_COLUMNS
    ]

    y = df["fraudulent"]

    # Stratified split - same configuration as baseline
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    # Calculate class imbalance weight using training data only
    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())

    scale_pos_weight = negative_count / positive_count

    print("Genuine training records:", negative_count)
    print("Fraudulent training records:", positive_count)
    print("Scale positive weight:", round(scale_pos_weight, 4))

    # Text and structured feature preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "text",
                TfidfVectorizer(
                    max_features=30000,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                    stop_words="english",
                    dtype=np.float32,
                ),
                "combined_text",
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=5,
                ),
                CATEGORICAL_COLUMNS,
            ),
            (
                "numeric",
                SimpleImputer(strategy="median"),
                NUMERIC_COLUMNS,
            ),
        ],
        remainder="drop",
    )

    # XGBoost classifier
    xgb_model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=2,
        reg_lambda=2.0,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        max_bin=256,
        eval_metric="aucpr",
        n_jobs=4,
        random_state=RANDOM_STATE,
    )

    # Complete pipeline
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", xgb_model),
    ])

    print("\nTraining XGBoost model...")
    model.fit(X_train, y_train)

    print("\nEvaluating XGBoost model...")

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, predictions)

    metrics = {
        "model": "TF-IDF + Structured Features + XGBoost",
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "pr_auc": float(
            average_precision_score(y_test, probabilities)
        ),
        "roc_auc": float(
            roc_auc_score(y_test, probabilities)
        ),
        "confusion_matrix": cm.tolist(),
        "scale_pos_weight": float(scale_pos_weight),
        "training_records": int(len(X_train)),
        "testing_records": int(len(X_test)),
    }

    # Save trained model
    model_path = MODEL_DIR / "xgboost_model.joblib"
    joblib.dump(model, model_path)

    # Save metrics
    metrics_path = REPORT_DIR / "xgboost_metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("\nXGBOOST MODEL RESULTS")
    print("-" * 45)

    for name, value in metrics.items():
        print(f"{name}: {value}")

    print("\nModel saved:", model_path)
    print("Metrics saved:", metrics_path)

    print("\nXGBoost training completed successfully!")


if __name__ == "__main__":
    main()
