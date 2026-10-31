
import pandas as pd

JOB_CATEGORIES = {
    "IT & Software": [
        "software", "developer", "programmer", "software engineer",
        "web developer", "python", "java", "database",
        "information technology", "network", "cyber security",
        "data analyst", "data scientist", "testing"
    ],

    "Telecalling & Voice Process": [
        "telecaller", "telecalling", "call center", "call centre",
        "voice process", "inbound calls", "outbound calls",
        "customer calling", "phone support"
    ],

    "Non-Voice & Chat Support": [
        "non voice", "non-voice", "chat support", "email support",
        "back office", "ticket support", "chat process"
    ],

    "Marketing": [
        "marketing", "digital marketing", "seo", "social media",
        "content marketing", "brand promotion", "advertising"
    ],

    "Sales & Business Development": [
        "sales", "business development", "sales executive",
        "field sales", "relationship manager"
    ],

    "BPO & KPO": [
        "bpo", "kpo", "process associate", "process executive",
        "operations executive", "customer service"
    ],

    "Data Entry & Back Office": [
        "data entry", "typing", "computer operator",
        "back office executive", "form filling"
    ],

    "Finance & Accounting": [
        "accountant", "accounting", "finance", "banking",
        "payroll", "auditor", "financial analyst"
    ],

    "HR & Recruitment": [
        "human resources", "hr executive", "recruiter",
        "recruitment", "talent acquisition"
    ],

    "Education": [
        "teacher", "teaching", "tutor", "lecturer",
        "academic counsellor", "education"
    ],

    "Healthcare": [
        "healthcare", "medical", "nurse", "hospital",
        "pharmacy", "medical coding"
    ],
}


def categorize_job(row):
    """Assign one approximate job category using job text."""

    text = " ".join([
        str(row.get("title", "")),
        str(row.get("function", "")),
        str(row.get("industry", "")),
        str(row.get("description", "")),
    ]).lower()

    for category, keywords in JOB_CATEGORIES.items():
        if any(keyword in text for keyword in keywords):
            return category

    return "Other"


def get_category_report(csv_path="data/fake_job_postings.csv"):
    """Calculate category-wise genuine and fraudulent job counts."""

    df = pd.read_csv(csv_path)

    df["job_category"] = df.apply(categorize_job, axis=1)

    report = (
        df.groupby("job_category")["fraudulent"]
        .agg(
            total_jobs="count",
            fake_jobs="sum",
        )
        .reset_index()
    )

    report["genuine_jobs"] = (
        report["total_jobs"] - report["fake_jobs"]
    )

    report["fraud_percentage"] = (
        report["fake_jobs"] / report["total_jobs"] * 100
    )

    return report.sort_values(
        "fraud_percentage",
        ascending=False,
    )
