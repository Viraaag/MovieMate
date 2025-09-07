# src/download_data.py
import os
import requests

# Map of files to Google Drive IDs
FILE_MAP = {
    "movies_metadata.csv": "1AURS3n9O6pJG7a_qkirKcSvL0vGt8fFm",
    "credits.csv": "1GOO7s4HymUqcfDMM6OcA9vggxpvl4O7h",
    "keywords.csv": "1Qq5YsgwnzvRlkNW8erK1u3B8g9Rs4g53",
    "ratings.csv": "1mkenpgUTJOg3TJ7NkpDaXMz7w3V6nTeQ",
    "links.csv": "1CEw0G3IsF6VEOhmXhMgF4jd-GRDiuvE8",
    "movies_metadata_updated.csv": "1RsYgsE7vsQZ2Ct2ic80RRLACPymzn2un",
    "movies_metadata_ai.csv": "11wayi_2wVPcQWLYOgvhbnQTU98PEXjHn",
    "ratings_small.csv": "10f6OIlpFUWWh1yR90kra-Two-eNx3Dz_",
    "tfidf_matrix.pkl": "1CApLW_uks1DplYxSXdqHvhRonbNYOV7G",
    "tfidf_vectorizer.pkl": "1tMruoELss1mAD0OD6IZ_-NZVuk0PZON6",
    "processed_metadata.pkl": "169RVGFSGD-vdjHSwpF0ATDjaqj4hYzTT",
}

def download_from_gdrive(file_id, destination):
    """Download a file from Google Drive public link."""
    url = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(url, stream=True)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            f.write(chunk)

def download_data():
    """Download all files if missing."""
    for filename, file_id in FILE_MAP.items():
        path = os.path.join("data", filename)
        if not os.path.exists(path):
            print(f"[INFO] Downloading {filename}...")
            download_from_gdrive(file_id, path)
        else:
            print(f"[INFO] {filename} already exists, skipping.")

if __name__ == "__main__":
    download_data()
