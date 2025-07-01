import os
import pandas as pd

# Step 1: Get base project path (i.e., Movie Recommender Systems)
current_dir = os.path.dirname(os.path.abspath(__file__))     # .../src
project_root = os.path.dirname(current_dir)                  # .../Movie Recommender Systems
data_dir = os.path.join(project_root, "data")

# Step 2: Load data
def load_data():
    print("Looking in:", data_dir)
    print("Files found:", os.listdir(data_dir))

    ratings = pd.read_csv(os.path.join(data_dir, "ratings_small.csv"))
    movies = pd.read_csv(os.path.join(data_dir, "movies_metadata.csv"), low_memory=False)

    credits = pd.read_csv(os.path.join(data_dir, "credits.csv"))
    keywords = pd.read_csv(os.path.join(data_dir, "keywords.csv"))
    return ratings, movies, credits, keywords

# Optional: test loading directly
if __name__ == "__main__":
    ratings, movies, credits, keywords = load_data()
    print("Data loaded successfully.")

import sys
print(sys.executable)

def clean_metadata(metadata):
    metadata['genres'] = metadata['genres'].fillna('[]')
    metadata['cast'] = metadata['cast'].fillna('[]')
    metadata['overview'] = metadata['overview'].fillna('')
    
    def parse_list(x):
        try:
            return " ".join([i['name'] for i in ast.literal_eval(x)])
        except:
            return ''
    
    metadata['genres_clean'] = metadata['genres'].apply(parse_list)
    metadata['cast_clean'] = metadata['cast'].apply(parse_list)

    metadata['combined_features'] = (
        metadata['genres_clean'] + " " +
        metadata['cast_clean'] + " " +
        metadata['overview']
    )
    
    return metadata
