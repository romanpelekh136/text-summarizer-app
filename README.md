# Text Summarizer App

A modern, premium web application for summarizing text using advanced algorithms.

## Features

- **Dual Summarization Engines**:
  - **LexRank (Local)**: Extractive summarization using graph-based algorithms. Fast, offline, and accurate for picking key sentences.
  - **Google Gemini (Cloud)**: Abstractive summarization using AI. Rewrites text for a more natural, human-like summary (requires API Key).
- **Adjustable Length**: Control the number of sentences in your summary.
- **Premium UI**: Beautiful glassmorphism design with smooth animations and responsive layout.
- **Docker Support**: Easy deployment with Docker Compose.
- **Secure**: Server-side API key configuration support.

## Tech Stack

- **Frontend**: React, Vite, Vanilla CSS (Premium Design)
- **Backend**: Python, FastAPI, Sumy (LexRank), Google Generative AI
- **Infrastructure**: Docker, Docker Compose

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
4. Run the server:
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

To use the **Google Gemini** summarizer without entering your API key in the frontend every time, you can configure it on the server.

1. Create a file named `.env` in the `backend/` folder.
2. Add your API key:
   ```env
   GEMINI_API_KEY=your_actual_api_key_here
   ```
3. Restart the backend.