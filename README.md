# MovieMate

## Production Frontend (React + Tailwind)

A new production-ready frontend is being added using React and Tailwind CSS. The frontend will:
- Provide a single-page, user-friendly interface for movie recommendations
- Allow users to input a movie title, user ID, and number of recommendations
- Display recommendations in a styled, responsive table
- Allow CSV download of results

The frontend will communicate with a REST API endpoint (to be exposed by the Python backend) for recommendations.

---

## REST API Endpoint (for Frontend Integration)

- **POST** `/api/recommend`
  - **Request Body:**
    ```json
    {
      "user_id": 1,
      "movie_title": "Avatar",
      "top_n": 5
    }
    ```
  - **Response:**
    ```json
    {
      "recommendations": [
        {
          "title": "Movie Title",
          "genres": "Action Adventure",
          "year": 2009,
          "match_percentage": 97.5,
          "why_recommended": "✅ Same Director, 🎭 Shared Actor, 🎯 Genre Match"
        },
        ...
      ]
    }
    ```

---