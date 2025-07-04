import React from "react";
import { CSVLink } from "react-csv";

type Recommendation = {
  title: string;
  match: number;
};

type RecommendationTableProps = {
  recommendations: Recommendation[];
};

const RecommendationTable: React.FC<RecommendationTableProps> = ({ recommendations }) => {
  if (!recommendations || recommendations.length === 0) return null;

  const headers = [
    { label: "Title", key: "title" },
    { label: "Match %", key: "match" },
  ];

  const csvData = recommendations.map((rec) => ({
    title: rec.title,
    match: rec.match,
  }));

  return (
    <div className="  rounded-2xl px-10 pt-8 pb-8 mb-8 w-full max-w-2xl mx-auto ">
      <div className="flex justify-between items-center mb-6">
        <h3 className="text-2xl font-bold text-white">Recommendations</h3>
        <CSVLink
          data={csvData}
          headers={headers}
          filename="recommendations.csv"
          className="bg-green-500 hover:bg-green-600 text-white font-semibold py-2 px-6 rounded-lg shadow-md transition text-base"
        >
          Download CSV
        </CSVLink>
      </div>
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200 rounded-lg overflow-hidden text-center">
          <thead className="bg-blue-50 ">
            <tr>
              <th className="px-6 py-3 font-bold text-blue-700  tracking-wider text-xl text-center">Title</th>
              <th className="px-6 py-3 font-bold text-blue-700 tracking-wider text-xl text-center">Match %</th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-100">
            {recommendations.map((rec, idx) => (
              <tr
                key={idx}
                className={
                  idx % 2 === 0
                    ? "bg-blue-50 hover:bg-blue-100 transition"
                    : "bg-white hover:bg-blue-50 transition"
                }
              >
                <td className="px-6 py-3 whitespace-nowrap text-lg text-gray-800">{rec.title}</td>
                <td className="px-6 py-3 whitespace-nowrap text-lg text-blue-700 font-semibold">
                  {typeof rec.match === "number" ? `${rec.match.toFixed(1)}` : "N/A"}
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
