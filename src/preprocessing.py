
import re
import pandas as pd

# Text columns used by the model
TEXT_COLUMNS = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
]

# Structured categorical features
CATEGORICAL_COLUMNS = [
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
    "department",
    "location",
    "salary_range",
]

# Structured binary features
NUMERIC_COLUMNS = [
    "telecommuting",
    "has_company_logo",
    "has_questions",
]


def clean_text(value):
    """Clean an individual text field."""
    if pd.isna(value):
        return ""

    text = str(value).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def load_dataset(path="data/fake_job_postings.csv"):
    """Load the EMSCAD dataset."""
    df = pd.read_csv(path)

    required_columns = (
        TEXT_COLUMNS
        + CATEGORICAL_COLUMNS
        + NUMERIC_COLUMNS
        + ["fraudulent"]
    )

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=["fraudulent"]).copy()
    df["fraudulent"] = df["fraudulent"].astype(int)

    return df


def prepare_features(df):
    """Create cleaned text and structured feature columns."""
    df = df.copy()

    for column in TEXT_COLUMNS:
        df[column] = df[column].apply(clean_text)

    # Combine the text fields into one model input
    df["combined_text"] = df[TEXT_COLUMNS].agg(" ".join, axis=1)

    # Fill missing categorical values
    for column in CATEGORICAL_COLUMNS:
        df[column] = df[column].fillna("unknown").astype(str)

    # Fill missing binary values
    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        df[column] = df[column].fillna(0)

    return df
