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

interface Recommendation {
  title: string;
  genres: string;
  year?: number;
  match_percentage: number;
  why_recommended: string;
}

const DEV_MODE = false; // Set to false to use real API

export default function App() {
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
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
        // Demo mode for testing
        await new Promise((resolve) => setTimeout(resolve, 800));

        const demoRecommendations: Recommendation[] = [
          { 
            title: "The Matrix", 
            genres: "Action, Sci-Fi",
            year: 1999,
            match_percentage: 92.5,
            why_recommended: "🧠 Content Similarity, 🎯 Genre Match"
          },
          { 
            title: "Inception", 
            genres: "Action, Sci-Fi, Thriller",
            year: 2010,
            match_percentage: 90.2,
            why_recommended: "🧠 Content Similarity, 🎯 Genre Match"
          },
          { 
            title: "Interstellar", 
            genres: "Adventure, Drama, Sci-Fi",
            year: 2014,
            match_percentage: 87.8,
            why_recommended: "🧠 Content Similarity, 🎯 Genre Match"
          },
          { 
            title: "Blade Runner 2049", 
            genres: "Action, Drama, Sci-Fi",
            year: 2017,
            match_percentage: 86.4,
            why_recommended: "🧠 Content Similarity, 🎯 Genre Match"
          },
          { 
            title: "Tenet", 
            genres: "Action, Thriller",
            year: 2020,
            match_percentage: 81.3,
            why_recommended: "🧠 Content Similarity, 🎯 Genre Match"
          }
        ];

        setRecommendations(demoRecommendations);
      } else {
        // Real API call
        const response = await axios.post("http://localhost:5000/recommend", formData);
        
        if (response.data.error) {
          setError(response.data.error);
          if (response.data.suggestions) {
            setSuggestions(response.data.suggestions);
          }
        } else {
          setRecommendations(response.data.recommendations);
        }
      }
    } catch (err: any) {
      if (!DEV_MODE && err.response && err.response.data) {
        setError(err.response.data.error || "An error occurred.");
        setSuggestions(err.response.data.suggestions || []);
      } else {
        setError("Could not connect to backend. Please make sure the server is running on port 5000.");
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
      <Navbar />
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
