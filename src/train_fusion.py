
import json
from pathlib import Path

import joblib

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.preprocessing import (
    load_dataset,
    prepare_features,
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
)

RANDOM_STATE = 42

Path("models").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)


def main():
    print("Loading and preparing dataset...")

    df = prepare_features(load_dataset())

    X = df.drop(columns=["fraudulent", "job_id"])
    y = df["fraudulent"]

    # Same random seed and stratification as baseline
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train = X.loc[train_idx]
    X_test = X.loc[test_idx]
    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx]

    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    # Text feature extraction
    text_transformer = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        stop_words="english",
    )

    # Categorical feature extraction
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        min_frequency=5,
    )

    # Numeric feature handling
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    # Combine all feature types
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "text",
                text_transformer,
                "combined_text",
            ),
            (
                "categorical",
                categorical_transformer,
                CATEGORICAL_COLUMNS,
            ),
            (
                "numeric",
                numeric_transformer,
                NUMERIC_COLUMNS,
            ),
        ]
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1500,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ])

    print("\nTraining fusion model...")
    model.fit(X_train, y_train)

    print("Evaluating fusion model...")

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": "Text + Structured Feature Fusion",
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
        "confusion_matrix": confusion_matrix(
            y_test, predictions
        ).tolist(),
    }

    joblib.dump(model, "models/fusion_model.joblib")

    with open(
        "reports/fusion_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=4)

    print("\nFUSION MODEL RESULTS")
    print("-" * 40)

    for name, value in metrics.items():
        print(f"{name}: {value}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Genuine", "Fraudulent"],
        zero_division=0,
    ))

    print("\nModel saved: models/fusion_model.joblib")
    print("Metrics saved: reports/fusion_metrics.json")


if __name__ == "__main__":
    main()
