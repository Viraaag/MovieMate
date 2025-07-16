# 🎬 MovieMate - AI Movie Recommendation System

A hybrid movie recommendation system that combines collaborative filtering and content-based filtering to provide personalized movie recommendations. Features a modern React frontend with a FastAPI backend.

## ✨ Features

- **Hybrid Recommendation Engine**: Combines collaborative filtering and content-based filtering
- **AI-Powered Fallback**: Uses Groq API to fetch movie data for titles not in the dataset
- **Modern Web Interface**: Beautiful React frontend with Tailwind CSS
- **Real-time Recommendations**: Fast API responses with detailed explanations
- **Export Functionality**: Download recommendations as CSV
- **Smart Suggestions**: Provides similar movie titles when exact matches aren't found

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- npm or yarn

### 1. Clone the Repository

```bash
git clone <repository-url>
cd MovieMate
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv310

# Activate virtual environment
# On Windows:
venv310\Scripts\activate
# On macOS/Linux:
source venv310/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

### 3. Set Up Frontend

```bash
cd frontend
npm install
cd ..
```

### 4. Configure API Keys (Optional)

If you want to use the AI fallback feature, create an `api.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Start the Application

#### Option A: Using the Startup Script (Recommended)

```bash
# On Windows:
start.bat

# On macOS/Linux:
python start.py
```

#### Option B: Manual Start

**Terminal 1 - Backend:**
```bash
# Activate virtual environment first
python -m uvicorn src.api_server:app --host 0.0.0.0 --port 5000 --reload
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

### 6. Access the Application

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:5000
- **API Documentation**: http://localhost:5000/docs

## 📁 Project Structure

```
MovieMate/
├── data/                          # Movie datasets
│   ├── ratings_small.csv         # User ratings
│   ├── movies_metadata.csv       # Movie metadata
│   ├── credits.csv               # Cast and crew data
│   └── keywords.csv              # Movie keywords
├── frontend/                      # React frontend
│   ├── src/
│   │   ├── components/           # React components
│   │   └── App.tsx              # Main app component
│   └── package.json
├── src/                          # Python backend
│   ├── api_server.py            # FastAPI server
│   ├── recommender.py           # Recommendation engine
│   └── dataloader.py            # Data loading utilities
├── start.py                      # Python startup script
├── start.bat                     # Windows startup script
└── requirements.txt              # Python dependencies
```

## 🔧 API Endpoints

### POST /recommend
Get movie recommendations based on a movie title.

**Request Body:**
```json
{
  "movie_title": "The Dark Knight",
  "user_id": 1,
  "num_recommendations": 5
}
```

**Response:**
```json
{
  "recommendations": [
    {
      "title": "Inception",
      "genres": "Action, Sci-Fi, Thriller",
      "year": 2010,
      "match_percentage": 92.5,
      "why_recommended": "🧠 Content Similarity, 🎯 Genre Match"
    }
  ],
  "error": null,
  "suggestions": null
}
```

### GET /health
Check API health status.

### GET /api/recommend (Legacy)
Legacy endpoint for backward compatibility.

## 🧠 How It Works

### Recommendation Algorithm

1. **Content-Based Filtering**: Uses TF-IDF vectorization of movie features (overview, genres, keywords, cast, crew)
2. **Collaborative Filtering**: Uses SVD (Singular Value Decomposition) on user ratings
3. **Hybrid Combination**: Combines both approaches with configurable weights
4. **Smart Boosting**: Applies genre overlap, title similarity, and director/actor matching boosts

### AI Fallback

When a movie title isn't found in the dataset, the system:
1. Uses Groq API to fetch movie metadata
2. Adds the movie to the dataset
3. Rebuilds the recommendation models
4. Provides recommendations based on the new data

## 🎨 Frontend Features

- **Modern UI**: Beautiful gradient design with glassmorphism effects
- **Responsive Design**: Works on desktop and mobile devices
- **Real-time Feedback**: Loading states and error handling
- **Smart Suggestions**: Click on suggested movie titles
- **Export Functionality**: Download recommendations as CSV
- **Detailed Information**: Shows genres, year, and recommendation reasons

## 🔍 Usage Examples

1. **Basic Recommendation**: Enter "The Matrix" to get sci-fi action recommendations
2. **Genre Exploration**: Try "Titanic" for romance and drama suggestions
3. **Director Matching**: Enter "Christopher Nolan" movies to find similar films
4. **Actor Discovery**: Search for movies with specific actors to find similar performances

## 🛠️ Development

### Backend Development

```bash
# Run with auto-reload
python -m uvicorn src.api_server:app --reload

# Run tests
python -m pytest tests/
```

### Frontend Development

```bash
cd frontend
npm run dev
npm run build
npm run test
```

### Adding New Features

1. **New Recommendation Algorithm**: Extend `HybridRecommender` class
2. **Frontend Components**: Add new React components in `frontend/src/components/`
3. **API Endpoints**: Add new routes in `src/api_server.py`

## 📊 Performance

- **Model Loading**: ~30 seconds on first startup
- **Recommendation Generation**: <1 second per request
- **AI Fallback**: ~3-5 seconds for new movie fetching
- **Frontend Response**: <100ms for UI updates

## 🐛 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Kill process on port 5000
   lsof -ti:5000 | xargs kill -9
   ```

2. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   cd frontend && npm install
   ```

3. **Data Files Missing**
   - Ensure all CSV files are in the `data/` directory
   - Check file permissions

4. **Frontend Not Loading**
   - Verify Node.js version (16+)
   - Clear npm cache: `npm cache clean --force`

### Debug Mode

Enable debug mode in `frontend/src/App.tsx`:
```typescript
const DEV_MODE = true; // Set to true for demo data
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- MovieLens dataset for training data
- Groq for AI-powered movie data fetching
- FastAPI for the backend framework
- React and Tailwind CSS for the frontend