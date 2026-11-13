
import json
from pathlib import Path

import joblib
import pandas as pd
from lime.lime_text import LimeTextExplainer

from src.preprocessing import (
    load_dataset,
    prepare_features,
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
)


MODEL_PATH = Path("models/xgboost_model.joblib")
REPORT_DIR = Path("reports")

REPORT_DIR.mkdir(exist_ok=True)


def main():

    print("Loading XGBoost model...")

    model = joblib.load(MODEL_PATH)

    print("Loading dataset...")

    df = load_dataset()
    df = prepare_features(df)

    # Select one fraudulent advertisement
    fraud_rows = df[df["fraudulent"] == 1]

    if fraud_rows.empty:
        raise ValueError("No fraudulent records found.")

    sample = fraud_rows.iloc[0]

    original_text = sample["combined_text"]

    # Keep structured features fixed while LIME varies the text
    structured_data = {
        column: sample[column]
        for column in CATEGORICAL_COLUMNS + NUMERIC_COLUMNS
    }

    def predict_text(texts):

        records = []

        for text in texts:

            record = {
                "combined_text": text,
                **structured_data
            }

            records.append(record)

        input_df = pd.DataFrame(records)

        return model.predict_proba(input_df)

    print("Initializing LIME...")

    explainer = LimeTextExplainer(
        class_names=["Genuine", "Fraud"],
        random_state=42
    )

    print("Generating XGBoost LIME explanation...")

    explanation = explainer.explain_instance(
        text_instance=original_text,
        classifier_fn=predict_text,
        labels=[1],
        num_features=15,
        num_samples=1000
    )

    prediction = int(
        model.predict(
            pd.DataFrame([{
                "combined_text": original_text,
                **structured_data
            }])
        )[0]
    )

    fraud_score = float(
        predict_text([original_text])[0][1]
    )

    feature_weights = explanation.as_list(label=1)

    result = {
        "actual_label": int(sample["fraudulent"]),
        "prediction": prediction,
        "fraud_score": fraud_score,
        "explanation_type": "Text-only LIME; structured features held constant",
        "top_features": [
            {
                "feature": feature,
                "contribution": float(weight),
                "direction": (
                    "Fraud" if weight > 0 else "Genuine"
                )
            }
            for feature, weight in feature_weights
        ]
    }

    report_path = REPORT_DIR / "xgboost_lime_explanation.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=4)

    # Save LIME visualization
    html_path = REPORT_DIR / "xgboost_lime_explanation.html"

    with open(html_path, "w", encoding="utf-8") as file:
        file.write(explanation.as_html(labels=[1]))

    print("\nXGBOOST LIME EXPLANATION")
    print("-" * 45)

    print(
        "Actual label:",
        "Fraud" if result["actual_label"] else "Genuine"
    )

    print(
        "Prediction:",
        "Fraud" if prediction else "Genuine"
    )

    print("Fraud score:", round(fraud_score, 4))

    print("\nTop contributing text features:")

    for feature, weight in feature_weights:

        direction = "Fraud" if weight > 0 else "Genuine"

        print(
            f"{feature}: {weight:.5f} ({direction})"
        )

    print("\nJSON report saved:", report_path)
    print("HTML explanation saved:", html_path)


if __name__ == "__main__":
    main()
