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
    onSubmit({ movie_title: movieTitle, user_id: userId, num_recommendations: numRecommendations });
  };

  return (
    <div className="w-full flex flex-col items-center justify-start px-4 sm:px-6 lg:px-8 mt-40 mb-12">
   



      
      <form
        onSubmit={handleSubmit}
        className="bg-white/90 backdrop-blur-lg shadow-xl rounded-2xl px-8 py-10 w-full max-w-xl border border-blue-100"
      >
        <h2 className="text-2xl font-bold mb-6 text-center text-gray-800">
          Get Tailored Movie Picks
        </h2>

        
        <div className="mb-5">
          <label className="block text-gray-700 text-sm font-medium mb-2">Movie Title</label>
          <input
            type="text"
            className="block w-full rounded-lg border border-gray-300 py-3 px-4 text-base focus:ring-2 focus:ring-blue-500 focus:outline-none transition"
            value={movieTitle}
            onChange={(e) => setMovieTitle(e.target.value)}
            placeholder="e.g. Avatar"
            required
          />
        </div>

        
        <div className="mb-5">
          <label className="block text-gray-700 text-sm font-medium mb-2">User ID</label>
          <input
            type="number"
            min={1}
            className="block w-full rounded-lg border border-gray-300 py-3 px-4 text-base focus:ring-2 focus:ring-blue-500 focus:outline-none transition"
            value={userId}
            onChange={(e) => setUserId(Number(e.target.value))}
            required
          />
        </div>

       
        <div className="mb-8">
          <label className="block text-gray-700 text-sm font-medium mb-2">
            Number of Recommendations:{" "}
            <span className="font-semibold text-blue-600">{numRecommendations}</span>
          </label>
          <input
            type="range"
            min={1}
            max={20}
            value={numRecommendations}
            onChange={(e) => setNumRecommendations(Number(e.target.value))}
            className="w-full accent-blue-500"
          />
        </div>

       
        <div className="flex items-center justify-center">
          <button
            type="submit"
            className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-8 rounded-lg text-lg shadow-md transition disabled:opacity-50"
            disabled={loading}
          >
            {loading ? (
              <span className="flex items-center gap-2">
                <span className="animate-spin h-5 w-5 border-2 border-white border-t-blue-400 rounded-full inline-block"></span>
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
