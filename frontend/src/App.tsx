import { useState } from "react";
import MovieInputForm from "./components/MovieInputForm";
import RecommendationTable from "./components/RecommendationTable";
import ErrorMessage from "./components/ErrorMessage";
import { ApiService, type Recommendation } from "./services/api";

import Hero from "./components/Hero";
import Navbar from "./components/Navbar";
import Footer from "./components/Footer";

interface FormData {
  movie_title: string;
  user_id: number;
  num_recommendations: number;
}

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
      // Always use real API call now
      try {
        const response = await ApiService.getRecommendations(formData);
        
        if (response.error) {
          setError(response.error);
          if (response.suggestions) {
            setSuggestions(response.suggestions);
          }
        } else {
          setRecommendations(response.recommendations);
        }
      } catch (err: any) {
        setError(err.message || "An unexpected error occurred.");
      }
    } catch (err: any) {
      setError(err.message || "An error occurred.");
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
