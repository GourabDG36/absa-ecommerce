import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizerFast

LABEL2ID = {"positive": 0, "negative": 1, "neutral": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")


class ABSADataset(Dataset):
    def __init__(self, df, max_length=128):
        self.texts = df["input_text"].tolist()
        self.labels = df["label"].tolist()
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }