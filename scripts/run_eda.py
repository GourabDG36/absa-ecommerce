import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.parse_semeval import load_all
from src.data.preprocess import preprocess_df

Path("outputs/eda").mkdir(parents=True, exist_ok=True)


def plot_polarity_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, domain in zip(axes, ["laptops", "restaurants"]):
        sub = df[df["domain"] == domain]
        sub["polarity"].value_counts().plot(
            kind="bar", ax=ax,
            color=["#2ecc71", "#e74c3c", "#95a5a6"]
        )
        ax.set_title(f"{domain.capitalize()} — polarity distribution")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.savefig("outputs/eda/polarity_distribution.png", dpi=150)
    print("Saved → outputs/eda/polarity_distribution.png")
    plt.close()


def plot_top_aspects(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, domain in zip(axes, ["laptops", "restaurants"]):
        top = (
            df[df["domain"] == domain]["aspect_term"]
            .str.lower()
            .value_counts()
            .head(20)
        )
        top.plot(kind="barh", ax=ax)
        ax.invert_yaxis()
        ax.set_title(f"Top 20 aspect terms — {domain}")
    plt.tight_layout()
    plt.savefig("outputs/eda/top_aspects.png", dpi=150)
    print("Saved → outputs/eda/top_aspects.png")
    plt.close()


def plot_text_length(df):
    df["text_len"] = df["text"].str.split().str.len()
    plt.figure(figsize=(8, 4))
    sns.histplot(df["text_len"], bins=30, kde=True, color="#3498db")
    plt.title("Review text — word count distribution")
    plt.xlabel("Word count")
    plt.tight_layout()
    plt.savefig("outputs/eda/text_length.png", dpi=150)
    print("Saved → outputs/eda/text_length.png")
    plt.close()
    return df


def print_class_balance(df):
    print("\n=== Label balance per domain ===")
    print(df.groupby(["domain", "polarity"]).size().unstack(fill_value=0))
    print("\n=== Overall polarity split ===")
    print(
        df["polarity"]
        .value_counts(normalize=True)
        .mul(100).round(1)
        .astype(str)
        .add("%")
    )


def main():
    print("Loading and parsing SemEval 2014...")
    raw_df = load_all("data/raw")

    print("Preprocessing...")
    df = preprocess_df(raw_df)
    print(f"Total samples after preprocessing: {len(df)}")

    plot_polarity_distribution(df)
    plot_top_aspects(df)
    df = plot_text_length(df)
    print_class_balance(df)

    df.to_csv("data/processed/semeval14_preprocessed.csv", index=False)
    print("\nSaved → data/processed/semeval14_preprocessed.csv")
    print("\nPhase 1 complete. Check outputs/eda/ for plots.")


if __name__ == "__main__":
    main()