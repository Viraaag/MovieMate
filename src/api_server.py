from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.recommender import HybridRecommender
import uvicorn
import os
import pandas as pd
from typing import List, Optional
import boto3

# --- Pydantic models ---
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

# --- FastAPI app ---
app = FastAPI(title="MovieMate API", version="1.0.0")

# Allow CORS for local frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global recommender instance ---
recommender: Optional[HybridRecommender] = None

# --- S3 CONFIG ---
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
S3_BUCKET = "moviemate-datasets"  # replace with your bucket name
FILES = [
    "credits.csv",
    "keywords.csv",
    "movies_metadata.csv",
    "movies_metadata_updated.csv",
    "ratings_small.csv"  # or ratings.csv if you prefer
]

def fetch_files_from_s3():
    """Download datasets from S3 if missing"""
    os.makedirs("data", exist_ok=True)
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION,
    )
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
            raise e

@app.on_event("startup")
async def startup_event():
    global recommender
    print("[INFO] Ensuring datasets are available...")
    fetch_files_from_s3()
    
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

# --- Health & root endpoints ---
@app.get("/")
async def root():
    return {"message": "MovieMate API is running!", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "recommender_loaded": recommender is not None}

# --- Recommendation endpoints ---
@app.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest):
    if recommender is None:
        raise HTTPException(status_code=500, detail="Recommender not initialized")
    try:
        movie_id = recommender.get_movie_id_from_title(req.movie_title)
        recs_df = recommender.hybrid_recommend(
            user_id=req.user_id,
            movie_id=movie_id,
            top_n=req.num_recommendations
        )
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
        if "not found" in error_msg.lower() and recommender.metadata is not None:
            try:
                import difflib
                all_titles = recommender.metadata["title"].str.lower().tolist()
                close_matches = difflib.get_close_matches(req.movie_title.lower(), all_titles, n=3, cutoff=0.6)
                suggestions = [title.title() for title in close_matches]
            except:
                pass
        return RecommendResponse(recommendations=[], error=error_msg, suggestions=suggestions)
    except Exception as e:
        print(f"[ERROR] Unexpected error in recommend endpoint: {e}")
        return RecommendResponse(recommendations=[], error=f"An unexpected error occurred: {str(e)}")

@app.get("/api/recommend")
async def legacy_recommend(movie_title: str, user_id: int = 1, top_n: int = 5):
    req = RecommendRequest(movie_title=movie_title, user_id=user_id, num_recommendations=top_n)
    return await recommend(req)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("src.api_server:app", host="0.0.0.0", port=port, reload=True)
