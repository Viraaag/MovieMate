import pandas as pd
import ast
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import difflib


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

    def genre_overlap(self, input_genres, genres):
        return len(input_genres & set(genres.split())) / len(input_genres) if input_genres else 0

    def _prepare_models(self):
        print("[INFO] Loading and merging metadata, credits, and keywords...")
        metadata = pd.read_csv(self.metadata_path, low_memory=False)
        credits = pd.read_csv(self.credits_path)
        keywords = pd.read_csv(self.keywords_path)

        metadata['id'] = metadata['id'].astype(str)
        credits['id'] = credits['id'].astype(str)
        keywords['id'] = keywords['id'].astype(str)

        metadata = metadata[["id", "title", "overview", "genres", "release_date"]]
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

        metadata["year"] = pd.to_datetime(metadata["release_date"], errors="coerce").dt.year

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
        if self.metadata is None:
            raise ValueError("Metadata not initialized.")

        title = title.strip().lower()

        self.metadata = self.metadata[self.metadata["title"].apply(lambda x: isinstance(x, str))].copy()
        self.metadata['title_lower'] = self.metadata['title'].str.lower().str.strip()

        result = self.metadata[self.metadata['title_lower'] == title]
        if not result.empty:
            return result['id'].values[0]

        all_titles = self.metadata["title_lower"].tolist()
        close_matches = difflib.get_close_matches(title, all_titles, n=3, cutoff=0.6)

        if close_matches:
            print("\n[INFO] Movie title not found.")
            print("[INFO] Did you mean:")
            for idx, match in enumerate(close_matches, 1):
                original_title = self.metadata[self.metadata["title_lower"] == match]["title"].values[0]
                print(f"{idx}. {original_title}")

            choice = input("Enter the number of the correct movie (or 0 to cancel): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(close_matches):
                selected_title = close_matches[int(choice) - 1]
                return self.metadata[self.metadata["title_lower"] == selected_title]["id"].values[0]
            else:
                raise ValueError("No valid movie selected.")
        else:
            raise ValueError(f"No similar movie titles found for '{title}'.")

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

    def hybrid_recommend(self, user_id, movie_id, top_n=10, alpha=0.3):
        content_recs = self.content_recommend(movie_id, top_n=top_n)

        user_rated_ids = pd.read_csv(self.ratings_path)
        user_rated_ids = user_rated_ids[user_rated_ids["userId"] == int(user_id)]["movieId"].astype(str).tolist()
        content_recs = content_recs[~content_recs["id"].isin(user_rated_ids)]

        content_recs = content_recs[content_recs["content_score"] > 0.2]
        if content_recs.empty:
            raise ValueError("No sufficiently similar movies found to recommend.")

        content_recs["cf_score"] = content_recs["id"].apply(lambda x: self.cf_model.predict(str(user_id), str(x)).est)
        content_recs["hybrid_score"] = alpha * content_recs["cf_score"] + (1 - alpha) * content_recs["content_score"]

        input_genres = set(self.metadata[self.metadata["id"] == movie_id]["genres"].values[0].split())
        content_recs["genre_boost"] = content_recs["genres"].apply(lambda g: self.genre_overlap(input_genres, g))
        content_recs["hybrid_score"] += 0.15 * content_recs["genre_boost"]

        content_recs["title_boost"] = content_recs["title"].apply(lambda x: self.title_overlap_boost(
            self.metadata[self.metadata["id"] == movie_id]["title"].values[0], x))
        content_recs["hybrid_score"] += 0.2 * content_recs["title_boost"]

        min_score = content_recs["hybrid_score"].min()
        max_score = content_recs["hybrid_score"].max()
        eps = 1e-5
        content_recs["match percentage"] = ((content_recs["hybrid_score"] - min_score + eps) / (max_score - min_score + eps)) * 100
        content_recs["match percentage"] = content_recs["match percentage"].round(1)

        input_movie = self.metadata[self.metadata["id"] == movie_id].iloc[0]
        input_director = input_movie["crew"]
        input_cast = input_movie["cast"].split()

        content_recs["same_director"] = content_recs["crew"].apply(lambda x: input_director in x)
        content_recs["shared_actor"] = content_recs["cast"].apply(lambda x: any(actor in x.split() for actor in input_cast))

        content_recs["shared_actor"] = content_recs["shared_actor"].apply(lambda x: "✔" if x else "✖")
        content_recs["same_director"] = content_recs["same_director"].apply(lambda x: "✔" if x else "✖")

        # Rename columns for cleaner output
        content_recs.rename(columns={
            "shared_actor": "Shared Actor",
            "same_director": "Same Director"
        }, inplace=True)

        return content_recs.sort_values("match percentage", ascending=False)[
            ["title", "genres", "year", "match percentage", "Shared Actor", "Same Director"]
        ].reset_index(drop=True)


# Run the recommender
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
        movie_id = recommender.get_movie_id_from_title(movie_title)
        recommendations = recommender.hybrid_recommend(int(user_id), movie_id, top_n=5)
        print("\nTop Recommendations:")
        print(recommendations)

        feedback = input("Mark any movies as NOT relevant (comma-separated numbers or 'none'): ")
        if feedback.lower() != "none":
            indices = [int(x.strip()) for x in feedback.split(",") if x.strip().isdigit()]
            if indices:
                not_liked_titles = recommendations.iloc[indices]["title"].tolist()
                print("Thanks! You didn't like:", not_liked_titles)

        save = input("Do you want to save recommendations to a CSV? (y/n): ").strip().lower()
        if save == "y":
            recommendations.to_csv(f"recommendations_for_{movie_title.replace(' ', '_')}.csv", index=False)
            print("Recommendations saved!")
    except Exception as e:
        print(f"Error: {e}")
