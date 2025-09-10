# fetch_data_s3.py
import os
import boto3

# --- CONFIGURATION ---
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")  # set your region
S3_BUCKET = "moviemate-datasets"  # replace with your S3 bucket name

# List of essential files to download
FILES = [
    "credits.csv",
    "keywords.csv",
    "movies_metadata.csv",
    "movies_metadata_updated.csv",
    "ratings.csv"  # or ratings_small.csv for testing
]

# Create data folder if it doesn't exist
os.makedirs("data", exist_ok=True)

# --- INITIALIZE S3 CLIENT ---
s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

# --- DOWNLOAD FILES ---
for file_name in FILES:
    local_path = os.path.join("data", file_name)
    if os.path.exists(local_path):
        print(f"[INFO] {file_name} already exists, skipping download.")
        continue

    try:
        print(f"[INFO] Downloading {file_name} from S3...")
        s3.download_file(S3_BUCKET, file_name, local_path)
        print(f"[INFO] {file_name} downloaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to download {file_name}: {e}")

print("[INFO] All files processed!")

