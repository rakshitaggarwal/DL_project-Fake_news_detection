#  Fake News Detection using RoBERTa  
Transformer-based Deep Learning Model for Classifying Fake vs Real News  

This project fine-tunes **RoBERTa-base**, a state-of-the-art Transformer architecture, for Fake News Detection using the Fake/True news dataset.  
RoBERTa achieves exceptional performance due to its deep contextual understanding.


---

#  **Training Results**

| Metric | Score |
|--------|--------|
| Accuracy | **1.00** |
| Precision | **1.00** |
| Recall | **1.00** |
| F1-Score | **1.00** |

Your RoBERTa model achieved **perfect performance** on the validation set.

---

# 🛠 Install Requirements

```bash
pip install torch transformers pandas scikit-learn numpy tqdm

python train.py
```
# dataset.py
```
import torch
from torch.utils.data import Dataset

class FakeNewsDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=256):
        self.texts = (df['title'].fillna('') + ' ' +
                      df['text'].fillna('')).tolist()
        self.labels = df['label'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(label, dtype=torch.long)
        return item
```
# model.py
```
import torch
import torch.nn as nn
from transformers import AutoModel

class RoBERTaFakeNewsClassifier(nn.Module):
    def __init__(self, model_name="roberta-base", dropout=0.3):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        hidden_size = self.roberta.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_emb = outputs.last_hidden_state[:, 0, :]  # CLS token
        logits = self.classifier(self.dropout(cls_emb))

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {"loss": loss, "logits": logits}

```
# eval.py
```
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def evaluate(model, dataloader, device):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            labels = batch["labels"].numpy()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs["logits"], dim=1).cpu().numpy()

            all_labels.extend(labels)
            all_preds.extend(preds)

    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds),
        "recall": recall_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds)
    }
```
# train.py
```
import os
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from dataset import FakeNewsDataset
from model import RoBERTaFakeNewsClassifier
from eval import evaluate

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_one_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Training"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs["loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    df = pd.read_csv("train.csv")

    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42,
                                        stratify=df['label'])

    model_name = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = FakeNewsDataset(train_df, tokenizer)
    val_dataset = FakeNewsDataset(val_df, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = RoBERTaFakeNewsClassifier(model_name=model_name)
    model.to(device)

    epochs = 3
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    best_val_f1 = 0.0
    metrics_history = []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        val_metrics = evaluate(model, val_loader, device)

        print(f"Train loss: {train_loss:.4f}")
        print(f"Val accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Val F1: {val_metrics['f1']:.4f}")

        metrics_history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            **val_metrics
        })

        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            os.makedirs("outputs", exist_ok=True)
            torch.save(model.state_dict(), "outputs/best_model.pt")
            print(">> Saved new best model")

    with open("outputs/metrics.json", "w") as f:
        json.dump(metrics_history, f, indent=2)

if __name__ == "__main__":
    main()

```
Why RoBERTa?
```
Modern transformer deep learning architecture

Strong contextual understanding

Outperforms LSTM/CNN

Achieved 100% accuracy & F1 on this dataset
