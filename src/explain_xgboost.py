
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

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

    # Select one fraudulent advertisement for local explanation
    fraud_rows = df[df["fraudulent"] == 1]

    if fraud_rows.empty:
        raise ValueError("No fraudulent records found.")

    sample = fraud_rows.iloc[[0]]

    X_sample = sample[
        ["combined_text"] + CATEGORICAL_COLUMNS + NUMERIC_COLUMNS
    ]

    print("Transforming selected advertisement...")

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    X_transformed = preprocessor.transform(X_sample)

    # Obtain transformed feature names
    feature_names = preprocessor.get_feature_names_out()

    # Convert sparse matrix to dense for the single selected sample
    X_dense = X_transformed.toarray() if hasattr(
        X_transformed, "toarray"
    ) else np.asarray(X_transformed)

    X_explain = pd.DataFrame(
        X_dense,
        columns=feature_names
    )

    print("Generating SHAP explanation...")

    explainer = shap.TreeExplainer(classifier)

    shap_values = explainer.shap_values(X_explain)

    # Handle binary classification output
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    contributions = shap_values[0]

    explanation = pd.DataFrame({
        "feature": feature_names,
        "contribution": contributions,
        "feature_value": X_dense[0],
    })

    explanation["abs_contribution"] = (
        explanation["contribution"].abs()
    )

    explanation = explanation.sort_values(
        by="abs_contribution",
        ascending=False
    )

    # Save top feature contributions
    top_features = explanation.head(30)

    prediction = int(model.predict(X_sample)[0])
    fraud_score = float(model.predict_proba(X_sample)[0, 1])

    result = {
        "actual_label": int(sample["fraudulent"].iloc[0]),
        "prediction": prediction,
        "fraud_score": fraud_score,
        "top_features": top_features[
            ["feature", "contribution", "feature_value"]
        ].to_dict(orient="records"),
    }

    report_path = REPORT_DIR / "xgboost_shap_explanation.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=4)

    # SHAP waterfall plot
    expected_value = explainer.expected_value

    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = np.asarray(expected_value).flatten()[-1]

    shap_explanation = shap.Explanation(
        values=contributions,
        base_values=float(expected_value),
        data=X_dense[0],
        feature_names=feature_names,
    )

    plt.figure(figsize=(12, 8))

    shap.plots.waterfall(
        shap_explanation,
        max_display=15,
        show=False
    )

    plt.tight_layout()

    plot_path = REPORT_DIR / "xgboost_shap_waterfall.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("\nXGBOOST SHAP EXPLANATION")
    print("-" * 45)

    print("Actual label:", "Fraud" if result["actual_label"] else "Genuine")
    print("Prediction:", "Fraud" if prediction else "Genuine")
    print("Fraud score:", round(fraud_score, 4))

    print("\nTop contributing features:")

    for _, row in top_features.head(15).iterrows():
        direction = (
            "Fraud"
            if row["contribution"] > 0
            else "Genuine"
        )

        print(
            f'{row["feature"]}: '
            f'{row["contribution"]:.5f} ({direction})'
        )

    print("\nSHAP report saved:", report_path)
    print("SHAP plot saved:", plot_path)


if __name__ == "__main__":
    main()
