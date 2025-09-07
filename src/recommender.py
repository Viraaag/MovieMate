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
from src.download_data import download_data


# Ensure all datasets exist before loading
download_data()

import pandas as pd

files = [
    "data/movies_metadata.csv",
    "data/movies_metadata_updated.csv",
    "data/movies_metadata_ai.csv",
    "data/credits.csv",
    "data/keywords.csv"
]

for f in files:
    df = pd.read_csv(f)
    print(f"{f}: {df.columns.tolist()}")

if 'movie_id' in df.columns:
    df.rename(columns={'movie_id': 'id'}, inplace=True)



import pandas as pd
import pickle

# Load CSVs
metadata = pd.read_csv("data/movies_metadata.csv", low_memory=False)
credits = pd.read_csv("data/credits.csv")
keywords = pd.read_csv("data/keywords.csv")
ratings = pd.read_csv("data/ratings.csv")
links = pd.read_csv("data/links.csv")
movies_metadata_updated = pd.read_csv("data/movies_metadata_updated.csv")
movies_metadata_ai = pd.read_csv("data/movies_metadata_ai.csv")
ratings_small = pd.read_csv("data/ratings_small.csv")

# Load PKL files
with open("data/tfidf_matrix.pkl", "rb") as f:
    tfidf_matrix = pickle.load(f)

with open("data/tfidf_vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)

with open("data/processed_metadata.pkl", "rb") as f:
    processed_metadata = pickle.load(f)




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

    def _is_nonempty_str(self, x):
        """Return True only if x is a non-empty Python string (not NA)."""
        return isinstance(x, str) and x.strip() != ""

    def title_overlap_boost(self, base_title, target_title):
        base_title = base_title.lower()
        target_title = target_title.lower()

    # Exact franchise name check
        if base_title.split()[0] == target_title.split()[0]:
            return 1.0 if base_title.split()[0] in ["housefull", "iron", "avengers"] else 0.7

        # Partial token match fallback
        base_words = set(base_title.split())
        target_words = set(target_title.split())
        return len(base_words & target_words) / len(base_words)


    def _prepare_models(self):
        print("[INFO] Loading and merging metadata, credits, and keywords...")

        # Load main metadata
        metadata = pd.read_csv(self.metadata_path, low_memory=False)
        metadata['id'] = metadata['id'].astype(str).str.strip()

        # Load credits and keywords
        credits = pd.read_csv(self.credits_path)
        keywords = pd.read_csv(self.keywords_path)
        for df in [credits, keywords]:
            df['id'] = df['id'].astype(str).str.strip()

        # Load AI-fetched movies if any
        if os.path.exists(self.ai_data_path):
            ai_movies = pd.read_csv(self.ai_data_path, low_memory=False)
            ai_movies['id'] = ai_movies['id'].astype(str).str.strip()  # ensure id exists & is clean
            print(f"[INFO] Loaded {len(ai_movies)} AI-fetched movies.")
            metadata = pd.concat([metadata, ai_movies], ignore_index=True)
        # Remove duplicates based on movie 'id' to avoid repeated recommendations
        metadata = metadata.drop_duplicates(subset='id', keep='last').reset_index(drop=True)
            

        # Keep only needed columns in metadata
        metadata = metadata[["id", "title", "overview", "genres", "release_date", "original_language", "production_countries"]]

        # Merge with credits and keywords
        metadata = metadata.merge(credits, on='id').merge(keywords, on='id')
        metadata = metadata.dropna(subset=["cast", "crew", "genres", "keywords"])

        # Parsing functions
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
        import os
        from dotenv import load_dotenv
        load_dotenv("api.env")  
        api_key = os.getenv("OMDB_API_KEY")
        if not api_key:
            # If the user didn't set an OMDB key, skip silently.
            return None, None
        
        base_url = "http://www.omdbapi.com/"
        params = {"t": title, "apikey": api_key}
        
        # Safe year handling: only add 'y' if year exists and is not pd.NA
        if year is not None and pd.notna(year):
            try:
                # cast to int to avoid sending pandas integer/NA types directly
                params["y"] = int(year)
            except Exception:
                # If casting fails, skip attaching year (still search by title)
                pass
        
        try:
            resp = requests.get(base_url, params=params, timeout=8)
            if resp.status_code == 200:
                data = resp.json()
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
                model="llama-3.3-70b-versatile",
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

        result = self.metadata.iloc[movie_indices][["id", "title", "genres", "cast", "crew", "year","original_language", "production_countries"]].copy()
        result["content_score"] = scores
        return result

    def hybrid_recommend(
        self, user_id, movie_id, top_n=10, alpha=0.3,
        preferred_language=None, preferred_country=None
    ):
        import re
        # fetch more candidates so we can filter/boost effectively
        initial_candidates = 50
        content_recs = self.content_recommend(movie_id, top_n=initial_candidates)

        if content_recs is None or content_recs.empty:
            raise ValueError("No candidate movies available from content_recommend.")

        # ensure these columns exist to avoid KeyErrors later
        if "original_language" not in content_recs.columns:
            content_recs["original_language"] = ""
        if "production_countries" not in content_recs.columns:
            content_recs["production_countries"] = ""

        # optional user filters
        if preferred_language:
            content_recs = content_recs[content_recs["original_language"] == preferred_language]

        if preferred_country:
            content_recs = content_recs[content_recs["production_countries"].astype(str).str.contains(preferred_country, na=False)]

        # drop candidates with almost-zero content similarity
        content_recs = content_recs[content_recs["content_score"] > 0.05]
        if content_recs.empty:
            raise ValueError("No sufficiently similar movies found to recommend.")

        # Collaborative Filtering scores (safe predict)
        def safe_predict(iid):
            try:
                return self.cf_model.predict(str(user_id), str(iid)).est
            except Exception:
                return 0.0

        content_recs["cf_score"] = content_recs["id"].apply(safe_predict)

        # base hybrid
        content_recs["hybrid_score"] = alpha * content_recs["cf_score"] + (1 - alpha) * content_recs["content_score"]

        # --- Genre boost ---
        try:
            seed_genres_text = self.metadata[self.metadata["id"] == movie_id]["genres"].values[0]
            input_genres = set(str(seed_genres_text).split()) if self._is_nonempty_str(seed_genres_text) else set()
        except Exception:
            input_genres = set()

        def genre_overlap(genres):
            try:
                gset = set(str(genres).split())
                return len(input_genres & gset) / (len(input_genres) if input_genres else 1)
            except Exception:
                return 0.0

        content_recs["genre_boost"] = content_recs["genres"].apply(genre_overlap)
        content_recs["hybrid_score"] += 0.3 * content_recs["genre_boost"]

        # --- Title boost (token overlap / franchise hint) ---
        seed_title = str(self.metadata[self.metadata["id"] == movie_id]["title"].values[0])
        content_recs["title_boost"] = content_recs["title"].apply(lambda x: self.title_overlap_boost(seed_title, x))
        content_recs["hybrid_score"] += 0.4 * content_recs["title_boost"]

        # --- Director boost ---
        input_movie = self.metadata[self.metadata["id"] == movie_id].iloc[0]
        input_director = input_movie.get("crew", "")
        content_recs["Same Director"] = content_recs["crew"].apply(
            lambda x: self._is_nonempty_str(x) and self._is_nonempty_str(input_director) and (input_director in x)
        )
        content_recs.loc[content_recs["Same Director"], "hybrid_score"] *= 1.5

        # --- Shared actor (robust-ish) ---
        def build_name_list(cast_str):
            # try comma split first, else pair words (heuristic)
            if not self._is_nonempty_str(cast_str):
                return []
            s = cast_str.strip()
            if "," in s:
                return [n.strip().lower() for n in s.split(",") if n.strip()]
            parts = s.split()
            names = []
            i = 0
            while i < len(parts):
                if i + 1 < len(parts):
                    names.append((parts[i] + " " + parts[i + 1]).lower())
                    i += 2
                else:
                    names.append(parts[i].lower())
                    i += 1
            return names

        seed_names = build_name_list(input_movie.get("cast", ""))

        def has_shared_actor(candidate_cast):
            if not self._is_nonempty_str(candidate_cast):
                return False
            low = candidate_cast.lower()
            cand_names = build_name_list(candidate_cast)
            # check for any seed full-name presence in candidate cast string
            for sn in seed_names:
                if sn in low or sn in cand_names:
                    return True
            return False

        content_recs["Shared Actor"] = content_recs["cast"].apply(has_shared_actor)
        content_recs.loc[content_recs["Shared Actor"], "hybrid_score"] *= 1.2

        # --- Language & Country boosts (if present) ---
        if self._is_nonempty_str(input_movie.get("original_language", "")) and "original_language" in content_recs.columns:
            lang = input_movie.get("original_language")
            content_recs.loc[content_recs["original_language"] == lang, "hybrid_score"] *= 1.2

        seed_countries = str(input_movie.get("production_countries", ""))
        if seed_countries:
            mask = content_recs["production_countries"].astype(str).str.contains(seed_countries, na=False)
            content_recs.loc[mask, "hybrid_score"] *= 1.2

        # --- Strong Franchise/Series detection and boost (robust) ---
        stop_words = {"the", "a", "an", "part", "chapter", "volume", "vol", "pt", "pt.", "movie", "episode", "and", "of"}
        def normalize_title_tokens(t):
            if not isinstance(t, str):
                return []
            s = t.lower()
            s = re.sub(r'\(.*?\)', '', s)                 # remove parentheses
            s = re.split(r'[:\-–—]', s)[0]                 # take part before colon/dash
            s = re.sub(r'[^a-z0-9\s]', ' ', s)            # remove punctuation
            tokens = [tok for tok in s.split() if tok and tok not in stop_words]
            return tokens

        def remove_trailing_ordinal(tokens):
            roman = re.compile(r'^(i|ii|iii|iv|v|vi|vii|viii|ix|x)$')
            return [tok for tok in tokens if not tok.isdigit() and not roman.match(tok)]

        seed_tokens = remove_trailing_ordinal(normalize_title_tokens(seed_title))
        seed_root = " ".join(seed_tokens[:2]) if seed_tokens else ""

        def compute_franchise_score(candidate_title):
            cand_tokens = remove_trailing_ordinal(normalize_title_tokens(candidate_title))
            if not cand_tokens or not seed_tokens:
                return 0.0
            cand_join = " ".join(cand_tokens)
            # strong root match
            if seed_root and (seed_root in cand_join or " ".join(cand_tokens[:2]) in " ".join(seed_tokens)):
                return 1.0
            # token Jaccard
            set_seed, set_cand = set(seed_tokens), set(cand_tokens)
            jaccard = len(set_seed & set_cand) / max(len(set_seed | set_cand), 1)
            if jaccard >= 0.5:
                return jaccard
            # fallback: longest matching substring ratio
            s = difflib.SequenceMatcher(None, seed_title.lower(), candidate_title.lower())
            lcs = max((block.size for block in s.get_matching_blocks()), default=0)
            ratio = lcs / max(len(seed_title), len(candidate_title), 1)
            if ratio >= 0.35:
                return ratio
            return 0.0

        content_recs["franchise_score"] = content_recs["title"].astype(str).apply(compute_franchise_score)

        # Strongly promote exact franchise matches
        if not content_recs["hybrid_score"].empty:
            max_h = content_recs["hybrid_score"].max()
            content_recs.loc[content_recs["franchise_score"] >= 0.99, "hybrid_score"] = max_h * 1.5
        content_recs["hybrid_score"] += 1.5 * content_recs["franchise_score"]

        # mark franchise match for explanation
        content_recs["Franchise Match"] = content_recs["franchise_score"] > 0.0

        # --- Normalize to match % (safe)
        min_score = content_recs["hybrid_score"].min()
        max_score = content_recs["hybrid_score"].max()
        eps = 1e-5
        content_recs["match percentage"] = ((content_recs["hybrid_score"] - min_score + eps) /
                                            (max_score - min_score + eps)) * 100
        content_recs["match percentage"] = content_recs["match percentage"].round(1)

        # Filter low matches
        content_recs = content_recs[content_recs["match percentage"] > 5.0]
        if content_recs.empty:
            raise ValueError("No sufficiently similar movies found to recommend.")

        # Explanation column (safe lookups)
        def explain(row):
            reasons = []
            if row.get("Franchise Match", False):
                reasons.append("🎬 Franchise/Sequel")
            if row.get("Same Director", False):
                reasons.append("✅ Same Director")
            if row.get("Shared Actor", False):
                reasons.append("🎭 Shared Actor")
            if row.get("genre_boost", 0) > 0.3:
                reasons.append("🎯 Genre Match")
            if row.get("content_score", 0) > 0.5:
                reasons.append("🧠 Content Similarity")
            if row.get("title_boost", 0) > 0:
                reasons.append("🎬 Title Overlap")
            if "original_language" in row and "original_language" in input_movie:
                if row["original_language"] == input_movie.get("original_language"):
                    reasons.append("🌍 Same Language")
            if "production_countries" in row and "production_countries" in input_movie:
                if str(input_movie.get("production_countries", "")) in str(row["production_countries"]):
                    reasons.append("🇮🇳 Same Country")
            return ", ".join(reasons)

        content_recs["Why Recommended"] = content_recs.apply(explain, axis=1)

        # Fetch posters + IMDb links
        content_recs["Poster"] = None
        content_recs["IMDb Link"] = None
        for idx, row in content_recs.iterrows():
            poster, imdb = self.fetch_omdb_info(row["title"], row.get("year", None))
            content_recs.at[idx, "Poster"] = poster
            content_recs.at[idx, "IMDb Link"] = imdb

        # Final columns & ordering
        final_cols = ["title", "genres", "year", "match percentage", "Why Recommended", "Poster", "IMDb Link"]
        content_recs = content_recs.sort_values("match percentage", ascending=False).head(top_n)
        return content_recs[final_cols].reset_index(drop=True)


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
            print(f"    Why Recommended: {row['Why Recommended']}")
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