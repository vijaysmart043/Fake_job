
import joblib
import numpy as np

MODEL_PATH = "models/baseline_model.joblib"


def load_model():
    return joblib.load(MODEL_PATH)


def explain_text(model, text, top_n=10):
    """
    Show the words and phrases that pushed the prediction
    towards fraudulent or genuine.
    """
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]

    cleaned_text = str(text).lower()

    # Convert input text into TF-IDF features
    vector = vectorizer.transform([cleaned_text])

    # Calculate each feature's contribution
    feature_names = vectorizer.get_feature_names_out()
    contributions = np.asarray(
        vector.multiply(classifier.coef_[0]).toarray()
    )[0]

    active_indices = vector.indices

    active_contributions = [
        (feature_names[index], contributions[index])
        for index in active_indices
    ]

    # Sort by contribution
    fraud_signals = sorted(
        [item for item in active_contributions if item[1] > 0],
        key=lambda item: item[1],
        reverse=True,
    )[:top_n]

    genuine_signals = sorted(
        [item for item in active_contributions if item[1] < 0],
        key=lambda item: item[1],
    )[:top_n]

    probability = model.predict_proba([cleaned_text])[0][1]
    prediction = int(probability >= 0.5)

    return {
        "prediction": prediction,
        "fraud_probability": float(probability),
        "fraud_signals": fraud_signals,
        "genuine_signals": genuine_signals,
    }


if __name__ == "__main__":
    model = load_model()

    sample_text = """
    Work from home opportunity.
    No experience required.
    Earn a very high income immediately.
    Pay a registration fee to start working.
    Send your personal details to apply.
    """

    result = explain_text(model, sample_text)

    print("\nJOB ADVERTISEMENT ANALYSIS")
    print("-" * 40)

    print(
        "Prediction:",
        "Fraudulent" if result["prediction"] else "Genuine"
    )

    print(
        "Fraud probability:",
        f"{result['fraud_probability']:.2%}"
    )

    print("\nWords supporting fraud prediction:")
    for word, score in result["fraud_signals"]:
        print(f"{word}: {score:.4f}")

    print("\nWords supporting genuine prediction:")
    for word, score in result["genuine_signals"]:
        print(f"{word}: {score:.4f}")
