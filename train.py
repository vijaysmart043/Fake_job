"""Train and evaluate the fake-job-posting classifier."""

import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.data import DATA_PATH, load_data
from src.evaluation import evaluate_model
from src.models import build_model, save_model

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "fake_job_model.joblib"
REPORT_PATH = ROOT / "reports" / "metrics.json"


def main() -> None:
    texts, labels = load_data(DATA_PATH)
    if labels.nunique() != 2 or labels.value_counts().min() < 2:
        raise ValueError("Training needs at least two examples of each label (0 and 1).")

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    model = build_model()
    model.fit(train_texts, train_labels)

    metrics = evaluate_model(model, test_texts, test_labels)
    save_model(model, MODEL_PATH)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Model saved to {MODEL_PATH}")
    print(f"Evaluation report saved to {REPORT_PATH}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")


if __name__ == "__main__":
    main()
