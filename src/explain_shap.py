
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import shap

from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix, hstack


ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "fake_job_postings.csv"
ENCODER_PATH = ROOT / "models" / "distilbert"
FUSION_PATH = ROOT / "models" / "distilbert_fusion.joblib"

OUTPUT_PATH = ROOT / "reports" / "shap_explanation.json"

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

DEVICE = torch.device("cpu")
MAX_LENGTH = 128
RANDOM_STATE = 42


def create_embeddings(texts, tokenizer, encoder):

    encoder.eval()
    embeddings = []

    for start in range(0, len(texts), 4):

        batch = texts[start:start + 4]

        tokens = tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        with torch.no_grad():

            output = encoder(**tokens)

            hidden = output.last_hidden_state

            mask = (
                tokens["attention_mask"]
                .unsqueeze(-1)
                .expand(hidden.size())
                .float()
            )

            pooled = (
                (hidden * mask).sum(dim=1)
                / mask.sum(dim=1).clamp(min=1e-9)
            )

        embeddings.append(pooled.numpy())

    return np.vstack(embeddings)


def main():

    print("Loading fusion model...")

    saved = joblib.load(FUSION_PATH)

    classifier = saved["classifier"]
    preprocessor = saved["preprocessor"]

    print("Loading DistilBERT encoder...")

    tokenizer = AutoTokenizer.from_pretrained(
        str(ENCODER_PATH)
    )

    encoder = AutoModel.from_pretrained(
        str(ENCODER_PATH)
    ).to(DEVICE)

    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    df = df.dropna(subset=["fraudulent"]).copy()
    df["fraudulent"] = df["fraudulent"].astype(int)

    df["combined_text"] = (
        df[TEXT_COLUMNS]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
    )

    X = df[
        ["combined_text"] + CAT_COLUMNS + NUM_COLUMNS
    ].copy()

    y = df["fraudulent"].to_numpy()

    # Reproduce the training split.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    X_train, _, y_train, _ = train_test_split(
        X_train,
        y_train,
        train_size=2000,
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

    # Select a small background from training records.
    background_df = X_fit.iloc[:20].copy()

    # Explain one held-out test record.
    sample_df = X_test.iloc[[0]].copy()

    print("Creating background embeddings...")

    background_embeddings = create_embeddings(
        background_df["combined_text"].tolist(),
        tokenizer,
        encoder
    )

    print("Creating sample embedding...")

    sample_embeddings = create_embeddings(
        sample_df["combined_text"].tolist(),
        tokenizer,
        encoder
    )

    background_structured = preprocessor.transform(
        background_df[CAT_COLUMNS + NUM_COLUMNS]
    )

    sample_structured = preprocessor.transform(
        sample_df[CAT_COLUMNS + NUM_COLUMNS]
    )

    background_matrix = hstack([
        csr_matrix(background_embeddings),
        background_structured
    ]).toarray()

    sample_matrix = hstack([
        csr_matrix(sample_embeddings),
        sample_structured
    ]).toarray()

    print("Creating SHAP explainer...")

    explainer = shap.LinearExplainer(
        classifier,
        background_matrix
    )

    shap_result = explainer(sample_matrix)

    values = np.asarray(shap_result.values)

    # Support both binary SHAP output formats.
    if values.ndim == 3:
        values = values[0, :, 1]
    else:
        values = values[0]

    feature_names = (
        ["distilbert_embedding_" + str(i)
         for i in range(768)]
        + list(preprocessor.get_feature_names_out())
    )

    # Aggregate all text embedding contributions.
    text_contribution = float(
        np.sum(values[:768])
    )

    # Group structured features by their original column.
    grouped = {}

    for index, name in enumerate(feature_names[768:]):

        contribution = float(values[768 + index])

        original_feature = name

        for col in CAT_COLUMNS + NUM_COLUMNS:

            if (
                name.startswith("categorical__" + col + "_")
                or name == "numerical__" + col
            ):
                original_feature = col
                break

        grouped[original_feature] = (
            grouped.get(original_feature, 0.0)
            + contribution
        )

    result = {
        "prediction": (
            "Fraudulent"
            if classifier.predict_proba(sample_matrix)[0, 1]
            >= saved["threshold"]
            else "Genuine"
        ),
        "fraud_score": float(
            classifier.predict_proba(sample_matrix)[0, 1]
        ),
        "text_contribution": text_contribution,
        "structured_feature_contributions": grouped,
        "top_structured_features": sorted(
            [
                {
                    "feature": name,
                    "contribution": value
                }
                for name, value in grouped.items()
            ],
            key=lambda item: abs(item["contribution"]),
            reverse=True
        )
    }

    print("\nSHAP EXPLANATION")
    print("=" * 45)

    print("Prediction:", result["prediction"])
    print("Fraud score:", result["fraud_score"])

    print(
        "Combined text contribution:",
        result["text_contribution"]
    )

    print("\nStructured feature contributions:")

    for item in result["top_structured_features"]:
        print(
            item["feature"],
            "=>",
            round(item["contribution"], 5)
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(OUTPUT_PATH, "w") as file:
        json.dump(result, file, indent=4)

    print("\nSHAP result saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
