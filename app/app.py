from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os

# Add the root directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.recommender import HybridRecommender

app = Flask(__name__)
CORS(app, resources={r"/recommend": {
    "origins": [
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:3000",
        "http://localhost:5000"
    ]
}})

# Compute base project directory (one level up from this file)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

recommender = HybridRecommender(
    ratings_path=os.path.join(BASE_DIR, "data/ratings_small.csv"),
    metadata_path=os.path.join(BASE_DIR, "data/movies_metadata.csv"),
    credits_path=os.path.join(BASE_DIR, "data/credits.csv"),
    keywords_path=os.path.join(BASE_DIR, "data/keywords.csv")
)


@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    movie_title = data.get('movie_title')
    user_id = data.get('user_id')
    num_recommendations = data.get('num_recommendations', 5)
    
    try:
        movie_id, suggestions, error = recommender.get_movie_id_from_title(movie_title)
        if movie_id is None:
            return jsonify({
                "recommendations": [],
                "suggestions": suggestions,
                "error": error
            }), 400
        recommendations = recommender.hybrid_recommend(user_id, movie_id, top_n=num_recommendations)
        recs = [
            {
                "title": row["title"],
                "match": float(row["match percentage"]),
                "genres": row.get("genres", []) or [],  # Always a list
                "release_year": row.get("release_year") or 0,
                "vote_average": row.get("vote_average") or 0.0,
                "poster_url": row.get("poster_url") or "",
                "imdb_url": row.get("imdb_link") or "",  # Make sure this key matches frontend
                "why_recommended": row.get("Why Recommended") or "",
            }
            for _, row in recommendations.iterrows()
        ]


        return jsonify({
            "recommendations": recs,
            "suggestions": [],
            "error": None
        })
    except Exception as e:
        return jsonify({
            "recommendations": [],
            "suggestions": [],
            "error": str(e)
        }), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
