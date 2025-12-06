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

        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item['labels'] = torch.tensor(label, dtype=torch.long)
        return item
