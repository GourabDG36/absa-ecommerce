import re
import pandas as pd
from transformers import RobertaTokenizerFast


LABEL2ID = {"positive": 0, "negative": 1, "neutral": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")


def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def build_input(text: str, aspect: str) -> str:
    """ABSA input format: [ASPECT] <aspect> [SEP] <sentence>"""
    return f"[ASPECT] {aspect} [SEP] {text}"


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["text"].apply(clean_text)
    df["input_text"] = df.apply(
        lambda r: build_input(r["text"], r["aspect_term"]), axis=1
    )
    df["label"] = df["polarity"].map(LABEL2ID)
    df = df.dropna(subset=["label"])
    return df[["sentence_id", "domain", "text", "aspect_term", "input_text", "label", "polarity"]]


def tokenize(texts: list[str], max_length: int = 128) -> dict:
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


if __name__ == "__main__":
    df = pd.read_csv("data/processed/semeval14_combined.csv")
    df = preprocess_df(df)
    df.to_csv("data/processed/semeval14_preprocessed.csv", index=False)
    print(df[["aspect_term", "polarity", "input_text"]].head(5))
    print("\nLabel distribution:\n", df["polarity"].value_counts())