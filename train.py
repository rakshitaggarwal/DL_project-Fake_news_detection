import os
import json
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split

from dataset import FakeNewsDataset
from model import RoBERTaFakeNewsClassifier
from eval import evaluate

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    df = pd.read_csv("train.csv")

    train_df, val_df = train_test_split(df, test_size=0.1, stratify=df["label"], random_state=42)

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    train_dataset = FakeNewsDataset(train_df, tokenizer)
    val_dataset = FakeNewsDataset(val_df, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    model = RoBERTaFakeNewsClassifier()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    epochs = 2  # Increase to 3–5 for better accuracy
    total_steps = len(train_loader) * epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    best_f1 = 0

    for epoch in range(epochs):
        print(f"\n---- Epoch {epoch+1}/{epochs} ----")
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch)["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        metrics = evaluate(model, val_loader, device)
        print(metrics)

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            os.makedirs("outputs", exist_ok=True)
            torch.save(model.state_dict(), "outputs/best_model.pt")
            with open("outputs/metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

    print("Training complete. Best F1:", best_f1)

if __name__ == "__main__":
    train()
