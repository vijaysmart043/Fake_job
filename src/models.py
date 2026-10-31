"""Model construction, training, and persistence."""

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_model() -> Pipeline:
    """Create a TF-IDF and logistic-regression classifier."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=100_000)),
            ("classifier", LogisticRegression(max_iter=1_000, class_weight="balanced")),
        ]
    )


def save_model(model: Pipeline, path: str | Path) -> None:
    """Persist a fitted model, creating its parent directory when needed."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination)


def load_model(path: str | Path) -> Pipeline:
    """Load a previously saved model."""
    return joblib.load(path)
