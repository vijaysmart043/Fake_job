"""Loading and preparing the job-postings dataset."""

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "fake_job_postings.csv"
TARGET_COLUMN = "fraudulent"
TEXT_COLUMNS = ("title", "company_profile", "description", "requirements", "benefits")


def load_data(path: str | Path = DATA_PATH) -> tuple[pd.Series, pd.Series]:
    """Return combined posting text and binary fraud labels."""
    frame = pd.read_csv(path)
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Dataset must contain a '{TARGET_COLUMN}' column.")

    available_text_columns = [column for column in TEXT_COLUMNS if column in frame.columns]
    if not available_text_columns:
        raise ValueError(f"Dataset must contain at least one of: {', '.join(TEXT_COLUMNS)}.")

    frame = frame.dropna(subset=[TARGET_COLUMN]).copy()
    if frame.empty:
        raise ValueError(f"Dataset contains no labeled rows. Add data to {path}.")

    labels = pd.to_numeric(frame[TARGET_COLUMN], errors="raise").astype(int)
    if not labels.isin((0, 1)).all():
        raise ValueError(f"'{TARGET_COLUMN}' must contain only 0 and 1 labels.")

    text = frame[available_text_columns].fillna("").astype(str).agg(" ".join, axis=1)
    return text, labels
