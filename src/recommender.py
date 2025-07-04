import pandas as pd
import ast
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import difflib
import datetime
from groq import Groq
from dotenv import load_dotenv
import openai
import os
# Load Groq API key from .env file
load_dotenv(dotenv_path="api.env")
openai.api_key = os.getenv("GROQ_API_KEY")



class HybridRecommender:
    def __init__(self, ratings_path, metadata_path, credits_path, keywords_path):
        self.ratings_path = ratings_path
        self.metadata_path = metadata_path
        self.credits_path = credits_path
        self.keywords_path = keywords_path

        self.metadata = None
        self.cf_model = None
        self.tfidf_matrix = None
        self.movie_indices = None

        self._prepare_models()

    def title_overlap_boost(self, base_title, target_title):
        base_words = set(base_title.lower().split())
        target_words = set(target_title.lower().split())
        return len(base_words & target_words) / len(base_words)

    def _prepare_models(self):
        print("[INFO] Loading and merging metadata, credits, and keywords...")
        metadata = pd.read_csv(self.metadata_path, low_memory=False)
        credits = pd.read_csv(self.credits_path)
        keywords = pd.read_csv(self.keywords_path)

        metadata['id'] = metadata['id'].astype(str)
        credits['id'] = credits['id'].astype(str)
        keywords['id'] = keywords['id'].astype(str)

        metadata = metadata[["id", "title", "overview", "genres", "release_date", "original_language", "production_countries"]]

        metadata = metadata.merge(credits, on='id')
        metadata = metadata.merge(keywords, on='id')

        metadata = metadata.dropna(subset=["cast", "crew", "genres", "keywords"])

        def parse_features(text, key=None):
            try:
                if not isinstance(text, str):
                    return ""
                items = ast.literal_eval(text)
                if isinstance(items, list):
                    if key:
                        return " ".join([d.get(key, "") for d in items if isinstance(d, dict)])
                    return " ".join([str(d) for d in items])
                return ""
            except:
                return ""

        def safe_parse_names(text, role=None, limit=None):
            try:
                if not isinstance(text, str):
                    return ""
                items = ast.literal_eval(text)
                if isinstance(items, list):
                    if role:
                        return " ".join([i["name"] for i in items if isinstance(i, dict) and i.get("job") == role])
                    if limit:
                        return " ".join([i["name"] for i in items[:limit] if isinstance(i, dict)])
                    return " ".join([i["name"] for i in items if isinstance(i, dict)])
                return ""
            except:
                return ""

        print("[INFO] Parsing metadata fields and generating soup...")

        metadata["genres"] = metadata["genres"].apply(lambda x: parse_features(x, "name"))
        metadata["keywords"] = metadata["keywords"].apply(lambda x: parse_features(x, "name"))
        metadata["cast"] = metadata["cast"].apply(lambda x: safe_parse_names(x, limit=3))
        metadata["crew"] = metadata["crew"].apply(lambda x: safe_parse_names(x, role="Director"))

        for field in ["overview", "genres", "keywords", "cast", "crew"]:
            metadata[field] = metadata[field].fillna("").astype(str)

        metadata["soup"] = (
            metadata["overview"] + " " +
            metadata["genres"] + " " +
            metadata["keywords"] + " " +
            metadata["cast"] + " " +
            metadata["crew"]
        )

        metadata["year"] = pd.to_datetime(metadata["release_date"], errors="coerce").dt.year.astype("Int64")
        metadata["original_language"] = metadata["original_language"].astype(str)
        metadata["production_countries"] = metadata["production_countries"].apply(lambda x: parse_features(x, "name"))



        if metadata["soup"].apply(lambda x: isinstance(x, float)).any():
            print("[ERROR] Float found in soup column:")
            print(metadata[metadata["soup"].apply(lambda x: isinstance(x, float))][["title", "soup"]])
            raise ValueError("Float detected in soup column")

        self.metadata = metadata.reset_index(drop=True)

        print("[INFO] Building content-based TF-IDF model...")
        tfidf = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = tfidf.fit_transform(self.metadata["soup"])

        self.movie_indices = pd.Series(self.metadata.index, index=self.metadata["id"])

        print("[INFO] Training collaborative filtering model...")
        reader = Reader(line_format='user item rating timestamp', sep=",", skip_lines=1)
        data = Dataset.load_from_file(self.ratings_path, reader=reader)
        trainset, _ = train_test_split(data, test_size=0.2)
        self.cf_model = SVD()
        self.cf_model.fit(trainset)

        print("[INFO] Model training complete.")

    def get_movie_id_from_title(self, title):
        """
        Returns (movie_id, suggestions, error):
        - If exact match: (movie_id, [], None)
        - If close matches: (None, [suggestions], error_message)
        - If not found: (None, [], error_message)
        """
        if self.metadata is None:
            return None, [], "Metadata not initialized."

        title = title.strip().lower()
        self.metadata = self.metadata[self.metadata["title"].apply(lambda x: isinstance(x, str))].copy()
        self.metadata['title_lower'] = self.metadata['title'].str.lower().str.strip()

        result = self.metadata[self.metadata['title_lower'] == title]
        if not result.empty:
            return result['id'].values[0], [], None

        all_titles = self.metadata["title_lower"].tolist()
        close_matches = difflib.get_close_matches(title, all_titles, n=3, cutoff=0.85)

        if close_matches:
            suggestions = [self.metadata[self.metadata["title_lower"] == match]["title"].values[0] for match in close_matches]
            return None, suggestions, f"Movie not found. Did you mean: {', '.join(suggestions)}?"
        else:
            return None, [], f"Movie '{title}' not found in the dataset."

    def fetch_and_add_movie_from_ai(self, title):
        import os
        from openai import OpenAI
        from dotenv import load_dotenv

        load_dotenv("api.env")
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return None, "GROQ_API_KEY not found in api.env"

        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )

        prompt = f"""
        Imagine you're a movie database. Given the title '{title}', provide the following:
        - Overview
        - Genre(s)
        - Release date (YYYY-MM-DD)
        - Top 2 Cast
        - Director
        - Keywords
        - Original Language (ISO 639-1 code)
        - Production Country (e.g., USA, India)

        Respond in JSON format with keys: title, overview, genres, release_date, cast, crew, keywords, original_language, production_countries.
        """

        try:
            response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "user", "content": prompt}]
            )
            import re
            import json5
            content = response.choices[0].message.content.strip()
            json_match = re.search(r"\{[\s\S]*\}", content)
            if not json_match:
                return None, "No valid JSON found in Groq response."
            json_str = json_match.group(0)
            try:
                ai_movie = json5.loads(json_str)
                ai_movie["genres"] = " ".join(ai_movie["genres"]) if isinstance(ai_movie["genres"], list) else str(ai_movie["genres"])
                ai_movie["keywords"] = " ".join(ai_movie["keywords"]) if isinstance(ai_movie["keywords"], list) else str(ai_movie["keywords"])
                def normalize_people_field(field):
                    if isinstance(field, list):
                        names = []
                        for item in field:
                            if isinstance(item, dict) and "name" in item:
                                names.append(item["name"])
                            elif isinstance(item, str):
                                names.append(item)
                        return " ".join(names)
                    return str(field)
                ai_movie["cast"] = normalize_people_field(ai_movie.get("cast", []))
                ai_movie["crew"] = normalize_people_field(ai_movie.get("crew", []))
                ai_movie["production_countries"] = " ".join(ai_movie["production_countries"]) if isinstance(ai_movie["production_countries"], list) else str(ai_movie["production_countries"])
            except Exception as e:
                return None, f"Failed to parse JSON: {e}"
            ai_movie["id"] = str(max(self.metadata["id"].astype(str).astype(int), default=100000) + 1)
            ai_movie["soup"] = (
                ai_movie["overview"] + " " +
                ai_movie["genres"] + " " +
                ai_movie["keywords"] + " " +
                ai_movie["cast"] + " " +
                ai_movie["crew"]
            )
            ai_movie["year"] = pd.to_datetime(ai_movie["release_date"], errors="coerce").year
            self.metadata = pd.concat([self.metadata, pd.DataFrame([ai_movie])], ignore_index=True)
            self.metadata.to_csv("data/movies_metadata_updated.csv", index=False)
            from sklearn.feature_extraction.text import TfidfVectorizer
            tfidf = TfidfVectorizer(stop_words="english")
            self.tfidf_matrix = tfidf.fit_transform(self.metadata["soup"].fillna(""))
            self.movie_indices = pd.Series(self.metadata.index, index=self.metadata["id"])
            return ai_movie["id"], None
        except Exception as e:
            return None, f"Failed to fetch movie from Groq: {e}"

    def content_recommend(self, movie_id, top_n=10):
        if movie_id not in self.movie_indices:
            print(f"[WARN] Movie ID {movie_id} not found in movie indices.")
            return pd.DataFrame()

        idx = self.movie_indices[movie_id]
        cosine_sim = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
        sim_scores = list(enumerate(cosine_sim))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1: top_n + 1]

        movie_indices = [i[0] for i in sim_scores]
        scores = [i[1] for i in sim_scores]

        result = self.metadata.iloc[movie_indices][["id", "title", "genres", "cast", "crew", "year"]].copy()
        result["content_score"] = scores
        return result

    def hybrid_recommend(self, user_id, movie_id, top_n=10, alpha=0.3, preferred_language=None, preferred_country=None):
        content_recs = self.content_recommend(movie_id, top_n=50)  # get more raw candidates

        if preferred_language:
            content_recs = content_recs[content_recs["original_language"] == preferred_language]

        if preferred_country:
            content_recs = content_recs[content_recs["production_countries"].str.contains(preferred_country, na=False)]


        user_rated_ids = pd.read_csv(self.ratings_path)
        user_rated_ids = user_rated_ids[user_rated_ids["userId"] == int(user_id)]["movieId"].astype(str).tolist()
        content_recs = content_recs[~content_recs["id"].isin(user_rated_ids)]

        #content_recs = content_recs[content_recs["content_score"] > 0.2]
        if content_recs.empty:
            raise ValueError("No sufficiently similar movies found to recommend.")

        content_recs["cf_score"] = content_recs["id"].apply(lambda x: self.cf_model.predict(str(user_id), str(x)).est)
        content_recs["hybrid_score"] = alpha * content_recs["cf_score"] + (1 - alpha) * content_recs["content_score"]

        input_genres = set(self.metadata[self.metadata["id"] == movie_id]["genres"].values[0].split())
        genre_overlap = lambda genres: len(input_genres & set(genres.split())) / len(input_genres) if input_genres else 0
        content_recs["genre_boost"] = content_recs["genres"].apply(genre_overlap)
        content_recs["hybrid_score"] += 0.15 * content_recs["genre_boost"]

        content_recs["title_boost"] = content_recs["title"].apply(lambda x: self.title_overlap_boost(self.metadata[self.metadata["id"] == movie_id]["title"].values[0], x))
        content_recs["hybrid_score"] += 0.2 * content_recs["title_boost"]

        min_score = content_recs["hybrid_score"].min()
        max_score = content_recs["hybrid_score"].max()
        eps = 1e-5

        content_recs["match percentage"] = ((content_recs["hybrid_score"] - min_score + eps) / (max_score - min_score + eps)) * 100
        content_recs["match percentage"] = content_recs["match percentage"].round(1)

        # Filter out very low match percentages (optional threshold, tweakable)
        content_recs = content_recs[content_recs["match percentage"] > 1.0]
        if len(content_recs) < top_n:
            print(f"[INFO] Only {len(content_recs)} strong recommendations found (above 5% match).")


        # Final fallback check
        if content_recs.empty:
            print("Error: No sufficiently similar movies found to recommend.")
            return pd.DataFrame()

        print("[DEBUG] Last 5 IDs:", self.metadata["id"].tail())

        input_row = self.metadata[self.metadata["id"].astype(str) == str(movie_id)]
        if input_row.empty:
            raise ValueError(f"[ERROR] Movie with ID {movie_id} not found in metadata after update.")
        input_movie = input_row.iloc[0]



        print("[DEBUG] Checking movie ID:", movie_id)
        print("[DEBUG] Available IDs:", self.metadata["id"].tolist()[:10])  # just the first 10 to avoid overflow
        print("[DEBUG] Matching row:\n", self.metadata[self.metadata["id"] == movie_id])


        input_director = input_movie["crew"]
        input_cast = input_movie["cast"].split()


        content_recs["same_director"] = content_recs["crew"].apply(lambda x: input_director in x)
        content_recs["shared_actor"] = content_recs["cast"].apply(lambda x: any(actor in x.split() for actor in input_cast))

        content_recs.rename(columns={
            "shared_actor": "Shared Actor",
            "same_director": "Same Director"
        }, inplace=True)

        def explain(row):
            reasons = []
            if row["Same Director"]: reasons.append("✅ Same Director")
            if row["Shared Actor"]: reasons.append("🎭 Shared Actor")
            if row["genre_boost"] > 0.3: reasons.append("🎯 Genre Match")
            if row["content_score"] > 0.5: reasons.append("🧠 Content Similarity")
            return ", ".join(reasons)

        content_recs["Why Recommended"] = content_recs.apply(explain, axis=1)

        # Remove 'Shared Actor' and 'Same Director' before returning
        if "Shared Actor" in content_recs.columns:
            content_recs.drop(columns=["Shared Actor", "Same Director"], inplace=True, errors='ignore')

        return content_recs.sort_values("match percentage", ascending=False).head(top_n)[
            ["title", "genres", "year", "match percentage", "Why Recommended"]
        ].reset_index(drop=True)




if __name__ == "__main__":
    recommender = HybridRecommender(
        ratings_path="data/ratings_small.csv",
        metadata_path="data/movies_metadata.csv",
        credits_path="data/credits.csv",
        keywords_path="data/keywords.csv"
    )

    movie_title = input("Enter a movie title: ").strip().lower()
    user_id = input("Enter user ID (default is 1): ") or "1"

    try:
        movie_id, suggestions, error = recommender.get_movie_id_from_title(movie_title)
        if error:
            print(error)
        else:
            print(f"[DEBUG] Matched movie ID: {movie_id}")
            print(recommender.metadata[recommender.metadata["id"] == movie_id][["title", "genres", "cast", "crew", "soup"]])

            recommendations = recommender.hybrid_recommend(int(user_id), movie_id, top_n=5)


            print("\nTop Recommendations:")
            print(recommendations)

            feedback = input("Mark any movies as NOT relevant (comma-separated numbers or 'none'): ")
            if feedback.lower() != "none":
                indices = [int(x.strip()) for x in feedback.split(",") if x.strip().isdigit()]
                if indices:
                    not_liked_titles = recommendations.iloc[indices]["title"].tolist()
                    print("Thanks! You didn't like:", not_liked_titles)

                    # Log feedback to CSV
                    feedback_df = pd.DataFrame({
                        "timestamp": datetime.datetime.now().isoformat(),
                        "user_id": [user_id] * len(not_liked_titles),
                        "title": not_liked_titles
                    })
                    feedback_df.to_csv("feedback_log.csv", mode='a', index=False, header=not pd.io.common.file_exists("feedback_log.csv"))

            save = input("Do you want to save recommendations to a CSV? (y/n): ").strip().lower()
            if save == "y":
                recommendations.to_csv(f"recommendations_for_{movie_title.replace(' ', '_')}.csv", index=False)
                print("Recommendations saved!")
    except Exception as e:
        print(f"Error: {e}")
