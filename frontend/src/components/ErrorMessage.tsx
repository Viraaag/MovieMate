import React from "react";

type ErrorMessageProps = {
  error?: string;
  suggestions?: string[];
  onSuggestionClick?: (suggestion: string) => void;
};

const ErrorMessage: React.FC<ErrorMessageProps> = ({ error, suggestions = [], onSuggestionClick }) => {
  if (!error && (!suggestions || suggestions.length === 0)) return null;

  return (
    <div className="bg-white/90 border-l-8 border-red-500 shadow-lg rounded-xl px-8 py-6 max-w-lg mx-auto mb-8 mt-4">
      {error && <div className="font-bold text-lg text-red-700 mb-3">{error}</div>}
      {suggestions && suggestions.length > 0 && (
        <div>
          <div className="mb-2 text-gray-700 font-semibold">Suggestions:</div>
          <div className="flex flex-wrap gap-3">
            {suggestions.map((s, idx) => (
              <button
                key={idx}
                className="bg-blue-100 hover:bg-blue-300 text-blue-800 font-semibold py-2 px-5 rounded-lg shadow-sm transition border border-blue-200"
                onClick={() => onSuggestionClick && onSuggestionClick(s)}
                type="button"
              >
                {s}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default ErrorMessage; 