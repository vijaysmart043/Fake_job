
import json
from pathlib import Path

import joblib
import pandas as pd
from lime.lime_text import LimeTextExplainer

from src.preprocessing import load_dataset


ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "baseline_model.joblib"
DATA_PATH = ROOT_DIR / "data" / "fake_job_postings.csv"
REPORT_PATH = ROOT_DIR / "reports" / "lime_explanation.json"

RANDOM_STATE = 42
NUM_FEATURES = 15


def main():
    print("Loading baseline model...")
    model = joblib.load(MODEL_PATH)

    print("Loading dataset...")
    df = load_dataset(DATA_PATH)

    # Select one fraudulent example for explanation.
    fraud_rows = df[df["fraudulent"] == 1]

    if fraud_rows.empty:
        raise ValueError("No fraudulent examples found in the dataset.")

    sample = fraud_rows.sample(
        n=1,
        random_state=RANDOM_STATE
    ).iloc[0]

    # Combine the job advertisement text.
    # text = combined_text(pd.DataFrame([sample])).iloc[0]
    text_columns = [
        "title",
        "company_profile",
        "description",
        "requirements",
        "benefits"
    ]

    text = " ".join(
        str(sample.get(column, ""))
        for column in text_columns
        if pd.notna(sample.get(column))
    )

    print("\nExplaining a fraudulent job advertisement...")
    print("Job title:", sample.get("title", "Unknown"))

    # LIME uses the model's probability predictions.
    explainer = LimeTextExplainer(
        class_names=["Genuine", "Fraud"],
        random_state=RANDOM_STATE
    )

    explanation = explainer.explain_instance(
        text_instance=text,
        classifier_fn=model.predict_proba,
        labels=[1],
        num_features=NUM_FEATURES,
        num_samples=1000
    )

    # Obtain model prediction.
    probabilities = model.predict_proba([text])[0]
    prediction = int(model.predict([text])[0])

    fraud_score = float(probabilities[1])

    # Extract word-level contributions.
    contributions = explanation.as_list(label=1)

    print("\nLIME EXPLANATION")
    print("=" * 45)
    print("Job title:", sample.get("title", "Unknown"))
    print("Actual label:", "Fraud" if sample["fraudulent"] == 1 else "Genuine")
    print("Model prediction:", "Fraud" if prediction == 1 else "Genuine")
    print(f"Fraud score: {fraud_score:.4f}")

    print("\nWord and phrase contributions:")
    for word, weight in contributions:
        direction = "Fraud" if weight > 0 else "Genuine"
        print(f"{word:35s} {weight:+.5f} ({direction})")

    # Save explanation.
    result = {
        "job_title": str(sample.get("title", "Unknown")),
        "actual_label": int(sample["fraudulent"]),
        "prediction": prediction,
        "prediction_label": "Fraud" if prediction == 1 else "Genuine",
        "fraud_score": fraud_score,
        "explanation_method": "LIME",
        "model": "TF-IDF + Logistic Regression",
        "num_features": NUM_FEATURES,
        "num_samples": 1000,
        "feature_contributions": [
            {
                "feature": word,
                "weight": float(weight),
                "direction": "Fraud" if weight > 0 else "Genuine"
            }
            for word, weight in contributions
        ]
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=4)

    # Save an interactive HTML explanation.
    html_path = ROOT_DIR / "reports" / "lime_explanation.html"
    explanation.save_to_file(str(html_path))

    print("\nLIME explanation saved:")
    print(REPORT_PATH)

    print("\nInteractive HTML explanation saved:")
    print(html_path)


if __name__ == "__main__":
    main()
