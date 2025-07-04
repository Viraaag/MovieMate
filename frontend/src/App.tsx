import { useState } from "react";
import MovieInputForm from "./components/MovieInputForm";
import RecommendationTable from "./components/RecommendationTable";
import ErrorMessage from "./components/ErrorMessage";
import axios from "axios";

import Hero from "./components/Hero";
import Navbar from "./components/Navbar";
import Footer from "./components/Footer";

interface FormData {
  movie_title: string;
  user_id: number;
  num_recommendations: number;
}

const DEV_MODE = true;

export default function App() {
  const [recommendations, setRecommendations] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [lastForm, setLastForm] = useState<FormData | null>(null);

  const handleSubmit = async (formData: FormData) => {
    setLoading(true);
    setError("");
    setSuggestions([]);
    setRecommendations([]);
    setLastForm(formData);

    try {
      if (DEV_MODE) {

        await new Promise((resolve) => setTimeout(resolve, 800));

        const demoRecommendations = [
          { title: "The Matrix", match: 9.2 },
          { title: "Inception", match: 9.0 },
          { title: "Interstellar", match: 8.7 },
          { title: "Blade Runner 2049", match: 8.6 },
          { title: "Tenet", match: 8.1 }
        ];


        setRecommendations(demoRecommendations);
      } else {
        const res = await axios.post("http://localhost:5000/recommend", formData);
        setRecommendations(res.data.recommendations);
      }
    } catch (err: any) {
      if (!DEV_MODE && err.response && err.response.data) {
        setError(err.response.data.error || "An error occurred.");
        setSuggestions(err.response.data.suggestions || []);
      } else {
        setError("Could not connect to backend or demo error occurred.");
      }
    } finally {
      setLoading(false);
    }
  };

  const handleSuggestionClick = (suggestion: string) => {
    if (lastForm) {
      handleSubmit({ ...lastForm, movie_title: suggestion });
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-900 via-violet-950 to-black overflow-auto">
      < Navbar />
      <Hero />
      
        <MovieInputForm onSubmit={handleSubmit} loading={loading} />
        <ErrorMessage
          error={error}
          suggestions={suggestions}
          onSuggestionClick={handleSuggestionClick}
        />

        <RecommendationTable recommendations={recommendations} />
        <Footer />
        </div>
      

      );
}
