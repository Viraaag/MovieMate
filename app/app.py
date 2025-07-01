import sys
import os

# Add the root directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.recommender import HybridRecommender
import streamlit as st
import pandas as pd

# Initialize recommender
@st.cache_resource
def load_recommender():
    return HybridRecommender(
        ratings_path="data/ratings_small.csv",
        metadata_path="data/movies_metadata.csv",
        credits_path="data/credits.csv",
        keywords_path="data/keywords.csv"
    )

recommender = load_recommender()

st.title("🎬 MovieMate Recommender")

# Input fields
movie_title = st.text_input("Enter a movie title:", "Avatar")
user_id = st.number_input("Enter User ID:", min_value=1, value=1)
top_n = st.slider("Number of recommendations:", 1, 20, 5)

# Recommend button
if st.button("Get Recommendations"):
    try:
        movie_id = recommender.get_movie_id_from_title(movie_title)
        recommendations = recommender.hybrid_recommend(user_id, movie_id, top_n=top_n)

        st.success(f"Top {top_n} recommendations for '{movie_title}':")

        # Display styled recommendations
        st.dataframe(recommendations.style.bar("match percentage", color="#00cc99"))

        # Download CSV
        csv = recommendations.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, f"{movie_title}_recommendations.csv", "text/csv")
    except Exception as e:
        st.error(f"Error: {e}")
