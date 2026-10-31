
from src.job_categories import get_category_report

report = get_category_report()

print("\nJOB CATEGORY FRAUD REPORT")
print("=" * 60)

print(report.to_string(index=False))

print("\nCategory analysis completed!")
