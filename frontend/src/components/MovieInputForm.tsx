import React, { useState } from "react";

type MovieInputFormProps = {
  onSubmit: (form: { movie_title: string; user_id: number; num_recommendations: number }) => void;
  loading?: boolean;
};

const MovieInputForm: React.FC<MovieInputFormProps> = ({ onSubmit, loading }) => {
  const [movieTitle, setMovieTitle] = useState("");
  const [userId, setUserId] = useState(1);
  const [numRecommendations, setNumRecommendations] = useState(5);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit({
      movie_title: movieTitle,
      user_id: userId,
      num_recommendations: numRecommendations,
    });
  };

  return (
    <div className="w-full flex flex-col items-center justify-start px-4 sm:px-6 lg:px-8 mt-28 mb-12">
      <form
        onSubmit={handleSubmit}
        className="bg-black/50 backdrop-blur-xl shadow-2xl border border-white/10 text-white px-10 py-12 rounded-3xl w-full max-w-2xl transition-all duration-300"
      >
        <h2 className="text-3xl md:text-4xl font-extrabold mb-8 text-center text-white drop-shadow-md tracking-wide">
          Movie<span className="text-red-500">Mate</span> AI Picks
        </h2>

        <div className="mb-6">
          <label className="block text-sm font-semibold text-gray-300 mb-2">
            Movie Title
          </label>
          <input
            type="text"
            className="w-full px-5 py-3 rounded-xl bg-white/10 border border-white/20 text-white placeholder-gray-400 focus:ring-2 focus:ring-red-500 focus:outline-none transition"
            placeholder="e.g. The Dark Knight"
            value={movieTitle}
            onChange={(e) => setMovieTitle(e.target.value)}
            required
          />
        </div>

        <div className="mb-6">
          <label className="block text-sm font-semibold text-gray-300 mb-2">User ID</label>
          <input
            type="number"
            min={1}
            className="w-full px-5 py-3 rounded-xl bg-white/10 border border-white/20 text-white placeholder-gray-400 focus:ring-2 focus:ring-red-500 focus:outline-none transition"
            value={userId}
            onChange={(e) => setUserId(Number(e.target.value))}
            required
          />
        </div>

        <div className="mb-8">
          <label className="block text-sm font-semibold text-gray-300 mb-2">
            Number of Recommendations:{" "}
            <span className="text-red-400 font-semibold">{numRecommendations}</span>
          </label>
          <input
            type="range"
            min={1}
            max={20}
            value={numRecommendations}
            onChange={(e) => setNumRecommendations(Number(e.target.value))}
            className="w-full accent-red-500"
          />
        </div>

        <div className="flex items-center justify-center">
          <button
            type="submit"
            disabled={loading}
            className="bg-red-600 hover:bg-red-700 text-white font-semibold px-8 py-3 rounded-xl text-lg shadow-md transition-all duration-300 disabled:opacity-50"
          >
            {loading ? (
              <span className="flex items-center gap-2">
                <span className="animate-spin h-5 w-5 border-2 border-white border-t-red-400 rounded-full inline-block"></span>
                Loading...
              </span>
            ) : (
              "Get Recommendations"
            )}
          </button>
        </div>
      </form>
    </div>
  );
};

export default MovieInputForm;
