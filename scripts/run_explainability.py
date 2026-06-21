import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.explainability.explain import load_model, build_explainer, explain_single

if __name__ == "__main__":
    model, tokenizer = load_model()
    explainer = build_explainer(model, tokenizer)

    # test with a custom review
    explain_single(
        text="The battery life on this laptop is absolutely amazing.",
        aspect="battery life",
        model=model,
        tokenizer=tokenizer,
        explainer=explainer,
        save_path="outputs/shap/custom_example.png",
    )