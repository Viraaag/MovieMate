#!/bin/bash
# fetch_data.sh
# Downloads all MovieMate datasets and model files from Google Drive into data/
# Skips files that already exist

# Ensure gdown is installed and up-to-date
pip install --upgrade gdown

echo "[INFO] Creating data/ folder..."
mkdir -p data

declare -A files=(
  ["credits.csv"]="1GOO7s4HymUqcfDMM6OcA9vggxpvl4O7h"
  ["keywords.csv"]="1Qq5YsgwnzvRlkNW8erK1u3B8g9Rs4g53"
  ["links.csv"]="1CEw0G3IsF6VEOhmXhMgF4jd-GRDiuvE8"
  ["movies_metadata.csv"]="1AURS3n9O6pJG7a_qkirKcSvL0vGt8fFm"
  ["movies_metadata_updated.csv"]="1RsYgsE7vsQZ2Ct2ic80RRLACPymzn2un"
  ["movies_metadata_ai.csv"]="11wayi_2wVPcQWLYOgvhbnQTU98PEXjHn"
  ["ratings_small.csv"]="10f6OIlpFUWWh1yR90kra-Two-eNx3Dz_"
  ["ratings.csv"]="1mkenpgUTJOg3TJ7NkpDaXMz7w3V6nTeQ"
  ["tfidf_matrix.pkl"]="1CApLW_uks1DplYxSXdqHvhRonbNYOV7G"
  ["tfidf_vectorizer.pkl"]="1tMruoELss1mAD0OD6IZ_-NZVuk0PZON6"
  ["processed_metadata.pkl"]="169RVGFSGD-vdjHSwpF0ATDjaqj4hYzTT"
)

for file in "${!files[@]}"; do
    if [ -f "data/$file" ]; then
        echo "[INFO] $file already exists, skipping download."
    else
        echo "[INFO] Downloading $file..."
        # Use full Google Drive URL and --fuzzy to handle large file confirmation automatically
        gdown "https://drive.google.com/uc?id=${files[$file]}" --fuzzy -O data/$file
    fi
done

echo "[INFO] All files processed!"
