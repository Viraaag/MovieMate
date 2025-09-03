import React from 'react';

interface Recommendation {
  title: string;
  genres: string[];
  release_year: number;
  vote_average: number;
  poster_url?: string;
  imdb_url?: string;
  why_recommended?: string;
}

interface RecommendationTableProps {
  recommendations: Recommendation[];
}

const RecommendationTable: React.FC<RecommendationTableProps> = ({ recommendations }) => {
  return (
    <div className="overflow-x-auto mt-6">
      <h2 className="text-2xl font-bold mb-4">Recommended Movies</h2>
      <table className="min-w-full table-auto border border-gray-300">
        <thead className="bg-gray-100">
          <tr>
            <th className="px-4 py-2">Poster</th>
            <th className="px-4 py-2">Title</th>
            <th className="px-4 py-2">Genres</th>
            <th className="px-4 py-2">Year</th>
            <th className="px-4 py-2">Rating</th>
            <th className="px-4 py-2">Why Recommended</th>
          </tr>
        </thead>
        <tbody>
          {recommendations.map((movie, index) => (
            <tr key={index} className="text-center border-t border-gray-200">
              <td className="px-4 py-2">
                {movie.poster_url ? (
                  <img
                    src={movie.poster_url}
                    alt={movie.title}
                    className="w-20 h-auto rounded"
                  />
                ) : (
                  'N/A'
                )}
              </td>
              <td className="px-4 py-2">
                {movie.imdb_url ? (
                  <a
                    href={movie.imdb_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-600 hover:underline"
                  >
                    {movie.title}
                  </a>
                ) : (
                  movie.title
                )}
              </td>
              <td className="px-4 py-2">
                {(movie.genres?.length && Array.isArray(movie.genres)) ? movie.genres.join(', ') : 'N/A'}
              </td>
              <td className="px-4 py-2">{movie.release_year}</td>
              <td className="px-4 py-2">{movie.vote_average}</td>
              <td className="px-4 py-2 whitespace-pre-wrap text-sm">{movie.why_recommended || '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default RecommendationTable;
