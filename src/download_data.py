# src/download_data.py
import os
import boto3

# Map of files to S3 keys
FILE_MAP = {
    "movies_metadata.csv": "data/movies_metadata.csv",
    "credits.csv": "data/credits.csv",
    "keywords.csv": "data/keywords.csv",
    "ratings.csv": "data/ratings.csv",
    "links.csv": "data/links.csv",
    "movies_metadata_updated.csv": "data/movies_metadata_updated.csv",
    "movies_metadata_ai.csv": "data/movies_metadata_ai.csv",
    "ratings_small.csv": "data/ratings_small.csv",
    "tfidf_matrix.pkl": "data/tfidf_matrix.pkl",
    "tfidf_vectorizer.pkl": "data/tfidf_vectorizer.pkl",
    "processed_metadata.pkl": "data/processed_metadata.pkl",
}


# Pull these from environment
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

def download_from_s3(s3_bucket, s3_key, local_path):
    """Download a file from S3 if it doesn't exist locally."""
    if os.path.exists(local_path):
        return

    print(f"[INFO] Downloading {s3_key} from S3 bucket {s3_bucket}...")
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_DEFAULT_REGION,
    )
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3.download_file(s3_bucket, s3_key, local_path)
    print(f"[INFO] Saved to {local_path}")

def download_data():
    """Download all files from S3 if missing."""
    if not S3_BUCKET_NAME:
        raise ValueError("[ERROR] S3_BUCKET_NAME not set in environment!")

    for filename, s3_key in FILE_MAP.items():
        local_path = os.path.join("data", filename)
        download_from_s3(S3_BUCKET_NAME, s3_key, local_path)

if __name__ == "__main__":
    download_data()
