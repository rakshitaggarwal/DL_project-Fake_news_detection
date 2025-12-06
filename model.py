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
        cls_emb = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(cls_emb))

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {"loss": loss, "logits": logits}
