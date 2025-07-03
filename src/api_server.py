from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.recommender import HybridRecommender
import uvicorn
import os
import pandas as pd

class RecommendRequest(BaseModel):
    user_id: int
    movie_title: str
    top_n: int = 5

app = FastAPI()

# Allow CORS for local frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the recommender once at startup
recommender = HybridRecommender(
    ratings_path=os.path.join("data", "ratings_small.csv"),
    metadata_path=os.path.join("data", "movies_metadata.csv"),
    credits_path=os.path.join("data", "credits.csv"),
    keywords_path=os.path.join("data", "keywords.csv")
)

@app.post("/api/recommend")
async def recommend(req: RecommendRequest):
    try:
        movie_id = recommender.get_movie_id_from_title(req.movie_title)
        recs_df = recommender.hybrid_recommend(req.user_id, movie_id, top_n=req.top_n)
        recommendations = [
            {
                "title": row["title"],
                "genres": row["genres"],
                "year": int(row["year"]) if not pd.isna(row["year"]) else None,
                "match_percentage": float(row["match percentage"]),
                "why_recommended": row["Why Recommended"],
            }
            for _, row in recs_df.iterrows()
        ]
        return {"recommendations": recommendations}
    except Exception as e:
        return {"recommendations": [], "error": str(e)}

if __name__ == "__main__":
    uvicorn.run("src.api_server:app", host="0.0.0.0", port=8000, reload=True) 