import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
import numpy as np
import pandas as pd
import mlflow
import mlflow.pytorch
import optuna

from torch import nn
from torch.utils.data import DataLoader
from transformers import RobertaForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight

from src.model.dataset import ABSADataset, ID2LABEL, LABEL2ID


DATA_PATH = "data/processed/semeval14_preprocessed.csv"
MODEL_DIR = "outputs/model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_LABELS = 3


def get_class_weights(labels):
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1, 2]),
        y=labels,
    )
    return torch.tensor(weights, dtype=torch.float).to(DEVICE)


def train_epoch(model, loader, optimizer, scheduler, loss_fn):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


def eval_epoch(model, loader, loss_fn):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            total_loss += loss.item()
            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average="macro")
    report = classification_report(
        all_labels, all_preds,
        target_names=list(ID2LABEL.values())
    )
    return total_loss / len(loader), f1, report


def run_training(lr, batch_size, epochs, trial_number=0):
    df = pd.read_csv(DATA_PATH)
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )

    train_ds = ABSADataset(train_df)
    val_ds = ABSADataset(val_df)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    class_weights = get_class_weights(train_df["label"].tolist())
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    with mlflow.start_run(run_name=f"trial_{trial_number}"):
        mlflow.log_params({"lr": lr, "batch_size": batch_size, "epochs": epochs})

        best_f1 = 0
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, loss_fn)
            val_loss, val_f1, report = eval_epoch(model, val_loader, loss_fn)

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_f1_macro": val_f1,
            }, step=epoch)

            print(f"Epoch {epoch}/{epochs} | train_loss: {train_loss:.4f} | val_f1: {val_f1:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
                model.save_pretrained(MODEL_DIR)
                print(f"  ✓ Best model saved (f1={best_f1:.4f})")

        print("\nClassification Report:\n", report)
        mlflow.log_metric("best_val_f1", best_f1)

    return best_f1


def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 5e-5, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32])
    return run_training(lr, batch_size, epochs=4, trial_number=trial.number)


if __name__ == "__main__":
    mlflow.set_experiment("absa-roberta")
    print(f"Training on: {DEVICE}")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5)

    print("\n=== Best trial ===")
    print(f"  F1: {study.best_value:.4f}")
    print(f"  Params: {study.best_params}")