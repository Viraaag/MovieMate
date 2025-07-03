import React, { useState } from "react";

interface Recommendation {
  title: string;
  genres: string;
  year: number;
  match_percentage: number;
  why_recommended: string;
}

const App: React.FC = () => {
  const [movieTitle, setMovieTitle] = useState("Avatar");
  const [userId, setUserId] = useState(1);
  const [topN, setTopN] = useState(5);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    setRecommendations([]);
    try {
      const res = await fetch("/api/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: userId,
          movie_title: movieTitle,
          top_n: topN,
        }),
      });
      if (!res.ok) throw new Error("Failed to fetch recommendations");
      const data = await res.json();
      setRecommendations(data.recommendations || []);
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message || "Unknown error");
      } else {
        setError("Unknown error");
      }
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = () => {
    const csvRows = [
      ["Title", "Genres", "Year", "Match %", "Why Recommended"],
      ...recommendations.map(r => [r.title, r.genres, r.year, r.match_percentage, r.why_recommended]),
    ];
    const csvContent = csvRows.map(row => row.join(",")).join("\n");
    const blob = new Blob([csvContent], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${movieTitle}_recommendations.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-gray-100 flex flex-col items-center justify-between">
      {/* Header */}
      <header className="w-full bg-white shadow-md py-6 px-4 flex flex-col items-center mb-8">
        <div className="flex items-center gap-3">
          <span className="text-4xl">🎬</span>
          <h1 className="text-3xl md:text-4xl font-extrabold text-blue-700 tracking-tight">MovieMate Recommender</h1>
        </div>
        <p className="text-gray-500 mt-2 text-center max-w-xl">Get personalized movie recommendations instantly with our smart hybrid engine. Enter your favorite movie and get suggestions tailored just for you!</p>
      </header>

      {/* Main Content */}
      <main className="w-full flex-1 flex flex-col items-center px-2">
        <form
          onSubmit={handleSubmit}
          className="bg-white shadow-lg rounded-xl p-8 w-full max-w-xl flex flex-col gap-6 mb-8"
        >
          <div className="flex flex-col gap-2">
            <label className="text-gray-700 font-semibold">Movie Title</label>
            <input
              type="text"
              className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-400 text-lg"
              value={movieTitle}
              onChange={e => setMovieTitle(e.target.value)}
              placeholder="e.g. Avatar"
              required
            />
          </div>
          <div className="flex gap-4 flex-col md:flex-row">
            <div className="flex-1 flex flex-col gap-2">
              <label className="text-gray-700 font-semibold">User ID</label>
              <input
                type="number"
                min={1}
                className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-400 text-lg"
                value={userId}
                onChange={e => setUserId(Number(e.target.value))}
                required
              />
            </div>
            <div className="flex-1 flex flex-col gap-2">
              <label className="text-gray-700 font-semibold"># of Recommendations</label>
              <input
                type="number"
                min={1}
                max={20}
                className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-400 text-lg"
                value={topN}
                onChange={e => setTopN(Number(e.target.value))}
                required
              />
            </div>
          </div>
          <button
            type="submit"
            className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-lg shadow transition disabled:opacity-60 text-lg"
            disabled={loading}
          >
            {loading ? (
              <span className="flex items-center gap-2"><svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"></path></svg>Loading...</span>
            ) : (
              "Get Recommendations"
            )}
          </button>
          {error && <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-2 rounded-lg font-medium text-center">{error}</div>}
        </form>

        {recommendations.length > 0 && (
          <section className="w-full max-w-3xl bg-white rounded-xl shadow-lg p-6 mb-8">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-2xl font-bold text-gray-800">Top {topN} Recommendations for <span className="text-blue-700">'{movieTitle}'</span></h2>
              <button
                onClick={handleDownload}
                className="bg-green-500 hover:bg-green-600 text-white px-4 py-2 rounded-lg shadow font-semibold"
              >
                Download CSV
              </button>
            </div>
            <div className="overflow-x-auto rounded-lg">
              <table className="min-w-full bg-white border border-gray-200 rounded-lg">
                <thead>
                  <tr className="bg-blue-50">
                    <th className="py-3 px-4 border-b text-left font-semibold">Title</th>
                    <th className="py-3 px-4 border-b text-left font-semibold">Genres</th>
                    <th className="py-3 px-4 border-b text-left font-semibold">Year</th>
                    <th className="py-3 px-4 border-b text-left font-semibold">Match %</th>
                    <th className="py-3 px-4 border-b text-left font-semibold">Why Recommended</th>
                  </tr>
                </thead>
                <tbody>
                  {recommendations.map((rec, idx) => (
                    <tr key={idx} className={idx % 2 === 0 ? "bg-gray-50 hover:bg-blue-50" : "bg-white hover:bg-blue-50"}>
                      <td className="py-2 px-4 border-b font-semibold text-gray-900">{rec.title}</td>
                      <td className="py-2 px-4 border-b text-gray-700">{rec.genres}</td>
                      <td className="py-2 px-4 border-b text-gray-700">{rec.year}</td>
                      <td className="py-2 px-4 border-b text-blue-700 font-bold">{rec.match_percentage}%</td>
                      <td className="py-2 px-4 border-b text-gray-700">{rec.why_recommended}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}
      </main>

      {/* Footer */}
      <footer className="w-full bg-white py-4 text-center text-gray-400 text-sm border-t mt-8">
        &copy; {new Date().getFullYear()} MovieMate &mdash; Smart Movie Recommendations
      </footer>
    </div>
  );
};

export default App; 