import os
import pandas as pd
import ast
import joblib
import requests
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import difflib
import datetime
from groq import Groq

from dotenv import load_dotenv
load_dotenv(dotenv_path="omdbapi.env")



class HybridRecommender:
    def __init__(self, ratings_path, metadata_path, credits_path, keywords_path):
        self.new_movies_buffer = []
        self.ai_data_path = "data/movies_metadata_ai.csv"
        self.tfidf_path = "cache/tfidf_vectorizer.pkl"
        self.tfidf_matrix_path = "cache/tfidf_matrix.pkl"
        self.cf_model_path = "cache/cf_model.pkl"

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

        if os.path.exists(self.ai_data_path):
            ai_movies = pd.read_csv(self.ai_data_path, low_memory=False)
            print(f"[INFO] Loaded {len(ai_movies)} AI-fetched movies.")
            metadata = pd.concat([metadata, ai_movies], ignore_index=True)

        credits = pd.read_csv(self.credits_path)
        keywords = pd.read_csv(self.keywords_path)

        metadata['id'] = metadata['id'].astype(str)
        credits['id'] = credits['id'].astype(str)
        keywords['id'] = keywords['id'].astype(str)

        metadata = metadata[["id", "title", "overview", "genres", "release_date", "original_language", "production_countries"]]
        metadata = metadata.merge(credits, on='id').merge(keywords, on='id')
        metadata = metadata.dropna(subset=["cast", "crew", "genres", "keywords"])

        def parse_features(text, key=None):
            try:
                if not isinstance(text, str): return ""
                items = ast.literal_eval(text)
                if isinstance(items, list):
                    return " ".join([d.get(key, "") if key else str(d) for d in items if isinstance(d, dict)])
            except: return ""
            return ""

        def safe_parse_names(text, role=None, limit=None):
            try:
                if not isinstance(text, str): return ""
                items = ast.literal_eval(text)
                if isinstance(items, list):
                    if role:
                        return " ".join([i["name"] for i in items if isinstance(i, dict) and i.get("job") == role])
                    if limit:
                        return " ".join([i["name"] for i in items[:limit] if isinstance(i, dict)])
                    return " ".join([i["name"] for i in items if isinstance(i, dict)])
            except: return ""
            return ""

        print("[INFO] Parsing metadata fields and generating soup...")
        metadata["genres"] = metadata["genres"].apply(lambda x: parse_features(x, "name"))
        metadata["keywords"] = metadata["keywords"].apply(lambda x: parse_features(x, "name"))
        metadata["cast"] = metadata["cast"].apply(lambda x: safe_parse_names(x, limit=3))
        metadata["crew"] = metadata["crew"].apply(lambda x: safe_parse_names(x, role="Director"))

        for field in ["overview", "genres", "keywords", "cast", "crew"]:
            metadata[field] = metadata[field].fillna("").astype(str)

        metadata["soup"] = (
            metadata["overview"] + " " + metadata["genres"] + " " +
            metadata["keywords"] + " " + metadata["cast"] + " " + metadata["crew"]
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
        os.makedirs(os.path.dirname(self.tfidf_path), exist_ok=True)
        if os.path.exists(self.tfidf_path) and os.path.exists(self.tfidf_matrix_path):
            self.tfidf_vectorizer = joblib.load(self.tfidf_path)
            self.tfidf_matrix = joblib.load(self.tfidf_matrix_path)
        else:
            self.tfidf_vectorizer = TfidfVectorizer(stop_words="english")
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.metadata["soup"])
            joblib.dump(self.tfidf_vectorizer, self.tfidf_path)
            joblib.dump(self.tfidf_matrix, self.tfidf_matrix_path)

        self.movie_indices = pd.Series(self.metadata.index, index=self.metadata["id"])

        print("[INFO] Training collaborative filtering model...")
        os.makedirs(os.path.dirname(self.cf_model_path), exist_ok=True)
        if os.path.exists(self.cf_model_path):
            self.cf_model = joblib.load(self.cf_model_path)
        else:
            reader = Reader(line_format='user item rating timestamp', sep=",", skip_lines=1)
            data = Dataset.load_from_file(self.ratings_path, reader=reader)
            trainset, _ = train_test_split(data, test_size=0.2)
            self.cf_model = SVD()
            self.cf_model.fit(trainset)
            joblib.dump(self.cf_model, self.cf_model_path)

        print("[INFO] Model training complete.")

    def refresh_tfidf_model(self):
        print("[INFO] Refreshing TF-IDF model with new movies...")
        self.tfidf_vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.metadata["soup"].fillna(""))
        self.movie_indices = pd.Series(self.metadata.index, index=self.metadata["id"])
        joblib.dump(self.tfidf_vectorizer, self.tfidf_path)
        joblib.dump(self.tfidf_matrix, self.tfidf_matrix_path)

    def validate_ai_response(self, ai_movie):
        required_keys = ["title", "overview", "genres", "release_date"]
        for key in required_keys:
            if key not in ai_movie or not ai_movie[key]:
                raise ValueError(f"[ERROR] AI response missing: {key}")

    def postprocess_recommendations(self, content_recs):
        if "vote_average" in self.metadata.columns:
            content_recs["hybrid_score"] += 0.05 * self.metadata.loc[content_recs.index, "vote_average"].fillna(0)

    def fetch_omdb_info(self, title, year=None):
        api_key = os.getenv("OMDB_API_KEY")
        query = f"http://www.omdbapi.com/?t={title}&apikey={api_key}"
        if year:
            query += f"&y={year}"
        try:
            response = requests.get(query)
            if response.status_code == 200:
                data = response.json()
                if data.get("Response") == "True":
                    poster = data.get("Poster")
                    imdb_id = data.get("imdbID")
                    imdb_link = f"https://www.imdb.com/title/{imdb_id}/" if imdb_id else None
                    return poster, imdb_link
        except Exception as e:
            print(f"[WARN] OMDb fetch failed for '{title}': {e}")
        return None, None


    def get_movie_id_from_title(self, title):
        if self.metadata is None:
            raise ValueError("Metadata not initialized.")

        title = title.strip().lower()

        self.metadata = self.metadata[self.metadata["title"].apply(lambda x: isinstance(x, str))].copy()
        self.metadata['title_lower'] = self.metadata['title'].str.lower().str.strip()

        result = self.metadata[self.metadata['title_lower'] == title]
        if not result.empty:
            return result['id'].values[0]

        all_titles = self.metadata["title_lower"].tolist()
        close_matches = difflib.get_close_matches(title, all_titles, n=3, cutoff=0.85)

        if close_matches:
            print("\n[INFO] Movie title not found.")
            print("[INFO] Did you mean:")
            for idx, match in enumerate(close_matches, 1):
                original_title = self.metadata[self.metadata["title_lower"] == match]["title"].values[0]
                print(f"{idx}. {original_title}")

            choice = input("Enter the number of the correct movie (or 0 to fetch using AI): ").strip()
            if choice.isdigit():
                choice = int(choice)
                if 1 <= choice <= len(close_matches):
                    selected_title = close_matches[choice - 1]
                    return self.metadata[self.metadata["title_lower"] == selected_title]["id"].values[0]
                elif choice == 0:
                    print("[INFO] Attempting to fetch movie data using AI fallback...")
                    ai_id = self.fetch_and_add_movie_from_ai(title.title())
                    if ai_id:
                        return ai_id
                    else:
                        raise ValueError("[ERROR] AI could not fetch movie data.")
            else:
                raise ValueError("Invalid input.")

        else:
            print(f"[INFO] '{title}' not found in the dataset.")
            use_ai = input("Would you like to try fetching movie data from AI? (y/n): ").strip().lower()
            if use_ai == "y":
                return self.fetch_and_add_movie_from_ai(title.title())
            else:
                raise ValueError(f"No similar movie titles found for '{title}'.")

    def fetch_and_add_movie_from_ai(self, title):
        import os
        from openai import OpenAI
        from dotenv import load_dotenv

    # Load Groq API key from api.env
        load_dotenv("api.env")
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("[ERROR] GROQ_API_KEY not found in api.env")
            return None
            

    # Initialize Groq client
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )

        print(f"[INFO] Querying Groq for metadata of '{title}'...")
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
                messages=[
                    {"role": "user", "content": prompt}
                ]              
            )
            import re
            import json5

            content = response.choices[0].message.content.strip()

            # Extract the first JSON object from the text using regex
            json_match = re.search(r"\{[\s\S]*\}", content)
            if not json_match:
                raise ValueError("No valid JSON found in Groq response.")

                
            json_str = json_match.group(0)
            print("[INFO] Extracted JSON:\n", json_str)

            try:
                ai_movie = json5.loads(json_str)
                # Normalize fields: convert list fields to space-separated strings
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
                print("[ERROR] Failed to parse JSON:", e)
                return None

            ai_movie["id"] = str(max(self.metadata["id"].astype(str).astype(int), default=100000) + 1)


            # Construct soup and year
            ai_movie["soup"] = (
                ai_movie["overview"] + " " +
                ai_movie["genres"] + " " +
                ai_movie["keywords"] + " " +
                ai_movie["cast"] + " " +
                ai_movie["crew"]
            )
            ai_movie["year"] = pd.to_datetime(ai_movie["release_date"], errors="coerce").year

            # Append to metadata
            # Append to metadata (in-memory only for now)
            self.metadata = pd.concat([self.metadata, pd.DataFrame([ai_movie])], ignore_index=True)
            self.new_movies_buffer.append(ai_movie)

            # Persist AI-fetched movie to disk
            if os.path.exists(self.ai_data_path):
                ai_df = pd.read_csv(self.ai_data_path)
                ai_df = pd.concat([ai_df, pd.DataFrame([ai_movie])], ignore_index=True)
            else:
                ai_df = pd.DataFrame([ai_movie])
            ai_df.to_csv(self.ai_data_path, index=False)

            print(f"[INFO] '{title}' added to dataset and saved for future use.")


            # Rebuild TF-IDF model to include the new movie
            from sklearn.feature_extraction.text import TfidfVectorizer
            tfidf = TfidfVectorizer(stop_words="english")
            self.tfidf_matrix = tfidf.fit_transform(self.metadata["soup"].fillna(""))
            self.movie_indices = pd.Series(self.metadata.index, index=self.metadata["id"])

            print(f"[INFO] '{title}' added to the dataset and models updated.")
            print(f"[DEBUG] New movie ID: {ai_movie['id']}")
            print(f"[DEBUG] New movie index in TF-IDF: {self.movie_indices[ai_movie['id']]}")
            return ai_movie["id"]
        except Exception as e:
            print("[ERROR] Failed to fetch movie from Groq:", e)
            return None
    def refresh_tfidf_model(self):
        """Rebuild TF-IDF model after batch of AI movies."""
        print("[INFO] Refreshing TF-IDF model with new movies...")
        tfidf = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = tfidf.fit_transform(self.metadata["soup"].fillna(""))
        self.movie_indices = pd.Series(self.metadata.index, index=self.metadata["id"])


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
        content_recs = self.content_recommend(movie_id, top_n=top_n)
        if preferred_language:
            content_recs = content_recs[content_recs["original_language"] == preferred_language]

        if preferred_country:
            content_recs = content_recs[content_recs["production_countries"].str.contains(preferred_country, na=False)]


        content_recs = content_recs[content_recs["content_score"] > 0.1]
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
        content_recs = content_recs[content_recs["match percentage"] > 5.0]
        if len(content_recs) < top_n:
            print(f"[INFO] Only {len(content_recs)} strong recommendations found (above 5% match).")


        if content_recs.empty:
            raise ValueError("No sufficiently similar movies found to recommend.")


        input_movie = self.metadata[self.metadata["id"] == movie_id].iloc[0]
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
        
        # Fetch posters and IMDb links using OMDb API
        content_recs["Poster"] = None
        content_recs["IMDb Link"] = None

        for idx, row in content_recs.iterrows():
            poster, imdb = self.fetch_omdb_info(row["title"], row["year"])
            content_recs.at[idx, "Poster"] = poster
            content_recs.at[idx, "IMDb Link"] = imdb

        # Return only selected columns (no need to drop explicitly)
        final_cols = ["title", "genres", "year", "match percentage", "Why Recommended", "Poster", "IMDb Link"]
        return content_recs.sort_values("match percentage", ascending=False)[final_cols].reset_index(drop=True)




if __name__ == "__main__":
    recommender = HybridRecommender(
        ratings_path="data/ratings_small.csv",
        metadata_path="data/movies_metadata.csv",
        credits_path="data/credits.csv",
        keywords_path="data/keywords.csv"
    )

    movie_title = input("Enter a movie title: ").strip().lower()
    user_id = "1"


    try:
        movie_id = recommender.get_movie_id_from_title(movie_title)
        # Refresh TF-IDF if new AI movies were added
        if recommender.new_movies_buffer:
            recommender.refresh_tfidf_model()


        # DEBUG: Show matched movie details
        print(f"[DEBUG] Matched movie ID: {movie_id}")
        print(recommender.metadata[recommender.metadata["id"] == movie_id][["title", "genres", "cast", "crew", "soup"]])

        recommendations = recommender.hybrid_recommend(int(user_id), movie_id, top_n=5)


        print("\nTop Recommendations:")
        for i, row in recommendations.iterrows():
            print(f"{i+1}. 🎬 {row['title']} ({row['year']}) - {row['match percentage']}% match")
            print(f"    Why: {row['Why Recommended']}")
            if row['IMDb Link']:
                print(f"    IMDb: {row['IMDb Link']}")
            if row['Poster']:
                print(f"    Poster: {row['Poster']}")




        save = input("Do you want to save recommendations to a CSV? (y/n): ").strip().lower()
        if save == "y":
            recommendations.to_csv(f"recommendations_for_{movie_title.replace(' ', '_')}.csv", index=False)
            print("Recommendations saved!")
    except Exception as e:
        print(f"Error: {e}")