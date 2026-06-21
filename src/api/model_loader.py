import torch
import numpy as np
from transformers import RobertaForSequenceClassification, RobertaTokenizerFast
from src.model.dataset import ID2LABEL

MODEL_DIR = "outputs/model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = None
_tokenizer = None


def get_model():
    global _model, _tokenizer
    if _model is None:
        _tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        _model = RobertaForSequenceClassification.from_pretrained(MODEL_DIR)
        _model.to(DEVICE)
        _model.eval()
    return _model, _tokenizer


def predict(text: str, aspect: str) -> dict:
    model, tokenizer = get_model()
    input_text = f"[ASPECT] {aspect} [SEP] {text}"

    inputs = tokenizer(
        input_text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    predicted_id = int(np.argmax(probs))

    return {
        "predicted_sentiment": ID2LABEL[predicted_id],
        "confidence": float(probs[predicted_id]),
        "probabilities": {
            "positive": float(probs[0]),
            "negative": float(probs[1]),
            "neutral": float(probs[2]),
        }
    }