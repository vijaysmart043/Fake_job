
import kagglehub
import shutil
from pathlib import Path

print("Downloading EMSCAD dataset...")

download_path = Path(
    kagglehub.dataset_download(
        "shivamb/real-or-fake-fake-jobposting-prediction"
    )
)

print("Downloaded files:", list(download_path.iterdir()))

data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

csv_files = list(download_path.rglob("fake_job_postings.csv"))

if not csv_files:
    raise FileNotFoundError(
        "fake_job_postings.csv was not found in the downloaded dataset."
    )

destination = data_dir / "fake_job_postings.csv"
shutil.copy2(csv_files[0], destination)

print("Dataset saved successfully:", destination)
print("File size:", round(destination.stat().st_size / (1024 * 1024), 2), "MB")
