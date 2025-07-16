from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.recommender import HybridRecommender
import uvicorn
import os
import pandas as pd
from typing import List, Optional

class RecommendRequest(BaseModel):
    movie_title: str
    user_id: int
    num_recommendations: int = 5

class RecommendationResponse(BaseModel):
    title: str
    genres: str
    year: Optional[int]
    match_percentage: float
    why_recommended: str

class RecommendResponse(BaseModel):
    recommendations: List[RecommendationResponse]
    error: Optional[str] = None
    suggestions: Optional[List[str]] = None

app = FastAPI(title="MovieMate API", version="1.0.0")

# Allow CORS for local frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global recommender instance
recommender: Optional[HybridRecommender] = None

@app.on_event("startup")
async def startup_event():
    global recommender
    print("[INFO] Initializing MovieMate recommender...")
    try:
        recommender = HybridRecommender(
            ratings_path=os.path.join("data", "ratings_small.csv"),
            metadata_path=os.path.join("data", "movies_metadata.csv"),
            credits_path=os.path.join("data", "credits.csv"),
            keywords_path=os.path.join("data", "keywords.csv")
        )
        print("[INFO] Recommender initialized successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to initialize recommender: {e}")
        raise e

@app.get("/")
async def root():
    return {"message": "MovieMate API is running!", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "recommender_loaded": recommender is not None}

@app.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest):
    if recommender is None:
        raise HTTPException(status_code=500, detail="Recommender not initialized")
    
    try:
        # Get movie ID from title
        movie_id = recommender.get_movie_id_from_title(req.movie_title)
        
        # Get recommendations
        recs_df = recommender.hybrid_recommend(
            user_id=req.user_id, 
            movie_id=movie_id, 
            top_n=req.num_recommendations
        )
        
        # Convert to response format
        recommendations = []
        for _, row in recs_df.iterrows():
            recommendations.append(RecommendationResponse(
                title=str(row["title"]),
                genres=str(row["genres"]),
                year=int(row["year"]) if isinstance(row["year"], (int, float)) and str(row["year"]) != "nan" else None,
                match_percentage=float(row["match percentage"]),
                why_recommended=str(row["Why Recommended"])
            ))
        
        return RecommendResponse(recommendations=recommendations)
        
    except ValueError as e:
        error_msg = str(e)
        suggestions = []
        
        # Try to provide suggestions for movie titles
        if "not found" in error_msg.lower() and recommender.metadata is not None:
            try:
                # Get similar titles using difflib
                import difflib
                all_titles = recommender.metadata["title"].str.lower().tolist()
                close_matches = difflib.get_close_matches(
                    req.movie_title.lower(), 
                    all_titles, 
                    n=3, 
                    cutoff=0.6
                )
                suggestions = [title.title() for title in close_matches]
            except:
                pass
        
        return RecommendResponse(
            recommendations=[],
            error=error_msg,
            suggestions=suggestions
        )
        
    except Exception as e:
        print(f"[ERROR] Unexpected error in recommend endpoint: {e}")
        return RecommendResponse(
            recommendations=[],
            error=f"An unexpected error occurred: {str(e)}"
        )

@app.get("/api/recommend")
async def legacy_recommend(movie_title: str, user_id: int = 1, top_n: int = 5):
    """Legacy endpoint for backward compatibility"""
    req = RecommendRequest(
        movie_title=movie_title,
        user_id=user_id,
        num_recommendations=top_n
    )
    return await recommend(req)

if __name__ == "__main__":
    uvicorn.run("src.api_server:app", host="0.0.0.0", port=5000, reload=True) 