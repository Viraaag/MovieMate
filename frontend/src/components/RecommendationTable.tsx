import React from "react";
import { CSVLink } from "react-csv";

type Recommendation = {
  title: string;
  genres: string;
  year?: number;
  match_percentage: number;
  why_recommended: string;
};

type RecommendationTableProps = {
  recommendations: Recommendation[];
};

const RecommendationTable: React.FC<RecommendationTableProps> = ({ recommendations }) => {
  if (!recommendations || recommendations.length === 0) return null;

  const headers = [
    { label: "Title", key: "title" },
    { label: "Genres", key: "genres" },
    { label: "Year", key: "year" },
    { label: "Match %", key: "match_percentage" },
    { label: "Why Recommended", key: "why_recommended" },
  ];

  const csvData = recommendations.map((rec) => ({
    title: rec.title,
    genres: rec.genres,
    year: rec.year || "N/A",
    match_percentage: rec.match_percentage,
    why_recommended: rec.why_recommended,
  }));

  return (
    <div className="backdrop-blur-xl bg-white/5 border border-white/10 shadow-2xl rounded-3xl px-8 pt-8 pb-10 mb-16 w-full max-w-6xl mx-auto text-white transition-all duration-300">
      <div className="flex justify-between items-center mb-6 px-2">
        <h3 className="text-3xl font-bold text-white drop-shadow tracking-wide">
          Your Recommendations
        </h3>
        <CSVLink
          data={csvData}
          headers={headers}
          filename="recommendations.csv"
          className="bg-red-600 hover:bg-red-700 text-white font-semibold py-2 px-6 rounded-xl text-sm shadow-md transition duration-300"
        >
          ⬇ Download CSV
        </CSVLink>
      </div>

      <div className="overflow-x-auto">
        <table className="min-w-full text-left rounded-xl overflow-hidden text-white">
          <thead>
            <tr className="bg-white/10 border-b border-white/10">
              <th className="px-6 py-4 text-lg font-semibold tracking-wider">Title</th>
              <th className="px-6 py-4 text-lg font-semibold tracking-wider">Genres</th>
              <th className="px-6 py-4 text-lg font-semibold tracking-wider">Year</th>
              <th className="px-6 py-4 text-lg font-semibold tracking-wider text-right">Match %</th>
              <th className="px-6 py-4 text-lg font-semibold tracking-wider">Why Recommended</th>
            </tr>
          </thead>
          <tbody>
            {recommendations.map((rec, idx) => (
              <tr
                key={idx}
                className={`transition duration-300 ${
                  idx % 2 === 0 ? "bg-white/5" : "bg-white/10"
                } hover:bg-red-900/30`}
              >
                <td className="px-6 py-4 text-base font-medium">{rec.title}</td>
                <td className="px-6 py-4 text-sm text-gray-300">{rec.genres}</td>
                <td className="px-6 py-4 text-sm text-gray-300">
                  {rec.year || "N/A"}
                </td>
                <td className="px-6 py-4 text-base text-right font-semibold text-red-400">
                  {typeof rec.match_percentage === "number" ? `${rec.match_percentage.toFixed(1)}` : "N/A"}
                </td>
                <td className="px-6 py-4 text-sm text-gray-300 max-w-xs">
                  {rec.why_recommended}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default RecommendationTable;
