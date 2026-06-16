import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path


def parse_semeval_xml(filepath: str, domain: str) -> pd.DataFrame:
    tree = ET.parse(filepath)
    root = tree.getroot()
    records = []

    for sentence in root.findall("sentence"):
        sid = sentence.get("id")
        text = sentence.findtext("text", default="").strip()

        aspect_terms = sentence.find("aspectTerms")
        if aspect_terms is None:
            continue

        for aspect in aspect_terms.findall("aspectTerm"):
            term = aspect.get("term")
            polarity = aspect.get("polarity")
            from_idx = int(aspect.get("from", 0))
            to_idx = int(aspect.get("to", 0))

            if polarity == "conflict":   # skip ambiguous labels
                continue

            records.append({
                "sentence_id": sid,
                "domain": domain,
                "text": text,
                "aspect_term": term,
                "polarity": polarity,
                "from": from_idx,
                "to": to_idx,
            })

    return pd.DataFrame(records)


def load_all(data_dir: str = "data/raw") -> pd.DataFrame:
    base = Path(data_dir)
    laptops = parse_semeval_xml(base / "Laptop_Train_v2.xml", "laptops")
    restaurants = parse_semeval_xml(base / "Restaurants_Train_v2.xml", "restaurants")
    df = pd.concat([laptops, restaurants], ignore_index=True)
    df.to_csv("data/processed/semeval14_combined.csv", index=False)
    print(f"Saved {len(df)} records → data/processed/semeval14_combined.csv")
    return df


if __name__ == "__main__":
    df = load_all()
    print(df.head())
    print(df["polarity"].value_counts())