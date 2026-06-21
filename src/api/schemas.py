from pydantic import BaseModel


class PredictRequest(BaseModel):
    text: str
    aspect: str


class AspectPrediction(BaseModel):
    text: str
    aspect: str
    predicted_sentiment: str
    confidence: float
    probabilities: dict[str, float]


class PredictResponse(BaseModel):
    prediction: AspectPrediction
    shap_explanation: dict[str, float] | None = None