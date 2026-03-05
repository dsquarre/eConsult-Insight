from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import asyncio
from typing import Dict, Any

from transformer import Analysis
from tfidf import Extract

app = FastAPI()

# CORS middleware configuration
headers = {
    'Access-Control-Expose-Headers': 'Content-Disposition',
    'Access-Control-Allow-Origin': 'http://localhost:3000'
}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Global state to store uploaded file data
uploaded_data = {
    "sentences": [],
    "sentiments": [],
    "important_rare": []
}


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    global uploaded_data
    uploaded_data = {
        "sentences": [],
        "sentiments": [],
        "important_rare": []
    }


@app.post("/upload/")
def upload_file(file: UploadFile = File(...)) -> Dict[str, str]:
    """
    Upload and process a file for analysis
    """
    global uploaded_data
    
    try:
        # Perform analysis on uploaded file
        sentences, sentiments, important = Analysis(file)
        
        if not sentences:
            return JSONResponse(
                status_code=400,
                content={"error": "Could not extract text from file. Please ensure the file contains a comments column and is a valid CSV or Excel file."}
            )
        
        # Extract word cloud and sentiment counts
        word_cloud, sentiment_counts = Extract(sentences, sentiments)
        
        # Store results
        uploaded_data = {
            "sentences": sentences,
            "sentiments": sentiments,
            "important_rare": important,
            "wordcount": word_cloud,
            "sentiment": sentiment_counts
        }
        
        return {"message": "File received and processed"}
        
    except Exception as e:
        print(f"Upload error: {e}")
        return JSONResponse(
            status_code=400,
            content={"error": f"Processing failed: {str(e)}"}
        )


@app.get("/result/")
async def get_result() -> Dict[str, Any]:
    """
    Get processed analysis results
    """
    global uploaded_data
    
    if not uploaded_data["sentences"]:
        return {
            "wordcount": {},
            "sentiment": {"positive": 0, "negative": 0, "neutral": 0},
            "important_rare": []
        }
    
    return {
        "wordcount": uploaded_data.get("wordcount", {}),
        "sentiment": uploaded_data.get("sentiment", {"positive": 0, "negative": 0, "neutral": 0}),
        "important_rare": uploaded_data.get("important_rare", [])
    }


@app.get("/")
def read_index() -> Dict[str, str]:
    """Root endpoint"""
    return {"message": "eConsult Insight API is running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
