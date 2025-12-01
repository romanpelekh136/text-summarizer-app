# Text Summarizer App

A modern web application for text summarization using multiple algorithms including machine learning.

## Features

- **Three Summarization Engines**:
  - **LexRank (Local)**: Graph-based extractive summarization. Fast and offline.
  - **Random Forest (ML)**: Machine learning-based extractive summarization trained on CNN/DailyMail dataset with SMOTE balancing.
  - **Google Gemini (Cloud)**: AI-powered abstractive summarization (requires API Key).
- **Adjustable Length**: Control the number of sentences in your summary.
- **Multi-language Support**: Auto-detects language for all models.
- **Clean UI**: Simple, minimal design with responsive layout.
- **Docker Support**: Easy deployment with Docker Compose.
- **Secure**: Server-side API key configuration via `.env`.

## Tech Stack

- **Frontend**: React, Vite, Vanilla CSS
- **Backend**: Python, FastAPI, Sumy (LexRank), Scikit-learn (Random Forest), Google Generative AI
- **ML**: Random Forest with SMOTE, TF-IDF, 12 engineered features
- **Infrastructure**: Docker, Docker Compose

## Random Forest Model

The Random Forest model uses advanced machine learning techniques:
- **Dataset**: 10,000 CNN/DailyMail articles
- **Features**: 12 engineered features including TF-IDF, position, length, proper nouns, quotes, centrality, and news-specific keywords
- **Class Balancing**: SMOTE (Synthetic Minority Over-sampling) to handle imbalanced data
- **Optimization**: News-specific tuning with strong lead sentence weighting

## Getting Started

### Prerequisites

- **Docker** (Recommended)
- OR **Node.js** (v18+) and **Python** (v3.10+)

### Method 1: Running with Docker (Recommended)

1. Clone the repository.
2. Create a `.env` file in the `backend/` directory (optional, see Configuration).
3. Run the application:
   ```bash
   docker-compose up --build
   ```
4. Open your browser:
   - **Frontend**: [http://localhost:5173](http://localhost:5173)
   - **Backend API**: [http://localhost:8000](http://localhost:8000)

### Method 2: Running Manually

#### Backend

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux/Mac
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. **(Optional)** Train the Random Forest model:
   ```bash
   python train_model.py
   ```
   This will download the CNN/DailyMail dataset and train the model (~10-15 minutes).
5. Run the server:
   ```bash
   python -m uvicorn main:app --reload
   ```

#### Frontend

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```

## Configuration

### Gemini API Key

To use the **Google Gemini** summarizer:

1. Create a file named `.env` in the `backend/` folder.
2. Add your API key:
   ```env
   GEMINI_API_KEY=your_actual_api_key_here
   ```
3. Restart the backend.

### Random Forest Model

The Random Forest model file (`summarizer_model.pkl`) is included if you've trained it locally. To retrain:

```bash
cd backend
python train_model.py
```

## Project Structure

```
text-summarizer-app/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── summarizer.py        # Summarization logic
│   ├── train_model.py       # Random Forest training script
│   ├── requirements.txt     # Python dependencies
│   ├── .env                 # Environment variables (create this)
│   └── Dockerfile           # Backend Docker config
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main React component
│   │   └── index.css        # Styles
│   ├── package.json         # Node dependencies
│   └── Dockerfile           # Frontend Docker config
├── docker-compose.yml       # Docker orchestration
└── README.md                # This file
```

## Usage

1. Open the app at `http://localhost:5173`
2. Select a summarization model from the dropdown:
   - **LexRank**: Fast, local, extractive
   - **Random Forest**: ML-based, trained on news articles
   - **Gemini**: Cloud-based, abstractive (requires API key)
3. Paste your text
4. Adjust the number of sentences
5. Click "Summarize"

## Development

The app uses hot-reloading for both frontend and backend:
- Frontend: Vite dev server automatically reloads on changes
- Backend: Uvicorn `--reload` flag restarts on Python file changes

## License

This project is for educational purposes.