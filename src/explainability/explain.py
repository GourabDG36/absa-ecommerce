import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from transformers import RobertaForSequenceClassification, RobertaTokenizerFast
from src.model.dataset import LABEL2ID, ID2LABEL

MODEL_DIR = "outputs/model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    model = RobertaForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(DEVICE)
    model.eval()
    return model, tokenizer


def predict_proba(texts, model, tokenizer):
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    ).to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return probs


def build_explainer(model, tokenizer):
    def f(texts):
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        return predict_proba(texts, model, tokenizer)

    masker = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(f, masker, output_names=list(ID2LABEL.values()))
    return explainer


def explain_single(text, aspect, model, tokenizer, explainer, save_path=None):
    input_text = f"[ASPECT] {aspect} [SEP] {text}"
    probs = predict_proba([input_text], model, tokenizer)[0]
    predicted_label = ID2LABEL[np.argmax(probs)]

    print(f"\nText     : {text}")
    print(f"Aspect   : {aspect}")
    print(f"Predicted: {predicted_label}")
    print(f"Probs    → positive: {probs[0]:.3f} | negative: {probs[1]:.3f} | neutral: {probs[2]:.3f}")

    shap_values = explainer([input_text])

    plt.figure()
    shap.plots.text(shap_values[0], display=False)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved SHAP plot → {save_path}")
    plt.close()

    return predicted_label, probs, shap_values


def run_batch_explanations(df_sample, model, tokenizer, explainer, output_dir="outputs/shap"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = []

    for idx, row in df_sample.iterrows():
        print(f"\n[{idx}] Processing: {row['aspect_term']}")
        label, probs, _ = explain_single(
            text=row["text"],
            aspect=row["aspect_term"],
            model=model,
            tokenizer=tokenizer,
            explainer=explainer,
            save_path=f"{output_dir}/shap_{idx}.png",
        )
        results.append({
            "text": row["text"],
            "aspect_term": row["aspect_term"],
            "true_label": row["polarity"],
            "predicted_label": label,
            "prob_positive": probs[0],
            "prob_negative": probs[1],
            "prob_neutral": probs[2],
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{output_dir}/shap_results.csv", index=False)
    print(f"\nSaved results → {output_dir}/shap_results.csv")
    return results_df


if __name__ == "__main__":
    print("Loading model...")
    model, tokenizer = load_model()
    explainer = build_explainer(model, tokenizer)

    df = pd.read_csv("data/processed/semeval14_preprocessed.csv")

    # one sample from each class for explanation report
    sample = pd.concat([
        df[df["polarity"] == "positive"].sample(2, random_state=42),
        df[df["polarity"] == "negative"].sample(2, random_state=42),
        df[df["polarity"] == "neutral"].sample(2, random_state=42),
    ]).reset_index(drop=True)

    results = run_batch_explanations(sample, model, tokenizer, explainer)
    print("\n=== SHAP Explanation Summary ===")
    print(results[["aspect_term", "true_label", "predicted_label"]].to_string())