# Aspect-Based Sentiment Analysis for E-commerce Reviews

Fine-tuned RoBERTa model for aspect-based sentiment classification, with SHAP-based explainability, deployed via FastAPI + Docker and demoed through Streamlit.

---

## Overview

Traditional sentiment analysis classifies an entire review as positive, negative, or neutral. This project goes a step further — it predicts sentiment **for a specific aspect** mentioned within a review. For example:

> "The battery life is amazing, but the keyboard feels cheap."

- Aspect: `battery life` → **Positive**
- Aspect: `keyboard` → **Negative**

This is **Aspect-Based Sentiment Analysis (ABSA)** — a more fine-grained and practically useful task than standard sentiment classification, especially for e-commerce review analysis where a single review often covers multiple product aspects with different (sometimes conflicting) sentiments.

---

## Key Features

- **Fine-tuned RoBERTa** on the SemEval 2014 Task 4 dataset (Laptop + Restaurant reviews)
- **Class-weighted loss** to handle label imbalance across sentiment classes
- **MLflow** experiment tracking (loss, accuracy, F1 across training runs)
- **Optuna** hyperparameter tuning (learning rate, batch size)
- **SHAP explainability** — token-level attribution showing *why* the model predicted a given sentiment
- **FastAPI** backend with a clean, documented `/predict` REST endpoint
- **Docker + Docker Compose** for reproducible, portable deployment
- **Streamlit** frontend for interactive demos

---

## Architecture

```
                                                          
   Review +         FastAPI            Fine-tuned         Sentiment +
   Aspect      →     /predict     →     RoBERTa      →     Confidence
                                                          
                                                
                                                ↓
                                        SHAP Explainer
                                       (token attribution)
                                                
                                                ↓
                                       Streamlit Frontend
                                       (interactive demo)
```

---

## Results

| Metric                | Value                                |
|------------------------|---------------------------------------|
| Macro F1               | 0.5485                               |
| Best learning rate     | 2.28e-05                             |
| Best batch size        | 32                                   |
| Training dataset       | SemEval 2014 Task 4 (Laptops + Restaurants) |
| Total samples          | 5,915 (after removing `conflict` labels) |

**Class distribution:**

| Sentiment | Count | Share  |
|-----------|-------|--------|
| Positive  | 3,151 | 53.3%  |
| Negative  | 1,671 | 28.3%  |
| Neutral   | 1,093 | 18.5%  |

The model performs strongly on clear-cut cases but struggles more on the minority neutral class and on sentences involving negation — both expected limitations given the class imbalance and the relatively small dataset size. SHAP explanations make these failure modes visible at the token level rather than leaving them as a black box, which is the core motivation behind this project.

---

## Tech Stack

`Python` · `PyTorch` · `HuggingFace Transformers (RoBERTa)` · `SHAP` · `MLflow` · `Optuna` · `FastAPI` · `Docker` · `Streamlit` · `scikit-learn`

---

## Project Structure

```
absa-ecommerce/
├── data/
│   ├── raw/                    # SemEval 2014 XML files
│   └── processed/              # cleaned, preprocessed CSVs
├── src/
│   ├── data/                   # XML parsing + preprocessing
│   ├── model/                  # dataset class + training loop
│   ├── explainability/         # SHAP explanation logic
│   └── api/                    # FastAPI app
├── scripts/
│   └── run_eda.py              # EDA script
├── outputs/
│   ├── model/                  # trained model weights
│   ├── eda/                    # EDA plots
│   └── shap/                   # SHAP explanation outputs
├── app.py                      # Streamlit frontend
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Setup & Usage

### 1. Clone and install dependencies

```bash
git clone https://github.com/GourabDG36/absa-ecommerce.git
cd absa-ecommerce
pip install -r requirements.txt
```

### 2. Prepare the data

Download the SemEval 2014 Task 4 XML files into `data/raw/`, then run:

```bash
python src/data/parse_semeval.py
python src/data/preprocess.py
```

### 3. Run exploratory data analysis

```bash
python scripts/run_eda.py
```

Plots are saved to `outputs/eda/`.

### 4. Train the model

```bash
python src/model/train.py
```

Tracks experiments via MLflow and tunes hyperparameters via Optuna automatically. The best model is saved to `outputs/model/`.

### 5. Generate SHAP explanations

```bash
python src/explainability/explain.py
```

Explanation plots and a results summary are saved to `outputs/shap/`.

### 6. Run the API

```bash
uvicorn src.api.main:app --reload --port 8000
```

### 7. Run the Streamlit demo

```bash
streamlit run app.py
```

### 8. Or run everything via Docker

```bash
docker-compose up --build
```

---

## API Reference

### `POST /predict`

**Request:**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "The battery life is amazing", "aspect": "battery life"}'
```

**Response:**

```json
{
  "prediction": {
    "text": "The battery life is amazing",
    "aspect": "battery life",
    "predicted_sentiment": "positive",
    "confidence": 0.573,
    "probabilities": {
      "positive": 0.573,
      "negative": 0.196,
      "neutral": 0.231
    }
  }
}
```

### `GET /health`

Simple health check endpoint, returns `{"status": "ok"}`.

---

## Future Improvements

- Expand training data beyond SemEval 2014 to improve neutral-class recall
- Add LIME as a secondary explainer for comparison against SHAP
- Auto-extract aspect terms instead of requiring manual input
- Add a GitHub Actions CI/CD pipeline for automated testing
- Experiment with larger backbone models (e.g. `roberta-large`, `deberta-v3`)

---

## Author

**Gourab** — B.Tech CSE (AI & Data Science)
GitHub: [@GourabDG36](https://github.com/GourabDG36)
