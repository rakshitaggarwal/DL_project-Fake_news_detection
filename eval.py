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
