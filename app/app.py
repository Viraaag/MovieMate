from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os

# Add the root directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.recommender import HybridRecommender

app = Flask(__name__)
CORS(app, resources={r"/recommend": {"origins": ["http://localhost:5173", "http://localhost:3000", "http://localhost:5000"]}})

# Initialize recommender (cache in global scope)
recommender = HybridRecommender(
    ratings_path="data/ratings_small.csv",
    metadata_path="data/movies_metadata.csv",
    credits_path="data/credits.csv",
    keywords_path="data/keywords.csv"
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
        # Format recommendations for JSON
        recs = [
            {"title": row["title"], "match": float(row["match percentage"])}
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
