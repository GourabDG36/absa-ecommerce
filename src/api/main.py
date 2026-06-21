import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import PredictRequest, PredictResponse, AspectPrediction
from src.api.model_loader import predict

app = FastAPI(
    title="ABSA API",
    description="Aspect-Based Sentiment Analysis using fine-tuned RoBERTa",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "ABSA API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict_sentiment(request: PredictRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    if not request.aspect.strip():
        raise HTTPException(status_code=400, detail="Aspect cannot be empty")

    try:
        result = predict(request.text, request.aspect)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return PredictResponse(
        prediction=AspectPrediction(
            text=request.text,
            aspect=request.aspect,
            predicted_sentiment=result["predicted_sentiment"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
        )
    )