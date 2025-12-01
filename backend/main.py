from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from summarizer import summarize_text
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SummarizeRequest(BaseModel):
    text: str
    sentences_count: int = 3
    model_type: str = "lexrank" # "lexrank" or "gemini"

class SummarizeResponse(BaseModel):
    summary: str

@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest):
    print(f"Received request: {request}")
    if not request.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    # Use API key from env var only
    api_key = os.getenv("GEMINI_API_KEY")

    try:
        print("Calling summarize_text...")
        summary = summarize_text(
            request.text, 
            request.sentences_count, 
            model_type=request.model_type, 
            api_key=api_key
        )
        print("Summarization complete.")
        return SummarizeResponse(summary=summary)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Text Summarizer API is running"}
