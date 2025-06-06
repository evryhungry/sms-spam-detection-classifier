import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class SpamHamDataset(Dataset):
    """
    CSV 파일을 읽어서, 각 행에 대해
    - input_ids (토큰 ID)
    - attention_mask
    - labels (0=ham, 1=spam)
    을 반환하는 PyTorch Dataset
    """
    def __init__(
        self,
        csv_path: str,
        tokenizer_name: str = "bert-base-uncased",
        max_length: int = 128,
    ):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        text = str(row["text"])
        label = int(row["label_id"])

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),         # 0 또는 1
        }
        return item


from torch.utils.data import DataLoader

def get_dataloader(
    csv_path: str,
    batch_size: int,
    shuffle: bool = False,
    tokenizer_name: str = "bert-base-uncased",
    max_length: int = 128,
):
    dataset = SpamHamDataset(csv_path, tokenizer_name=tokenizer_name, max_length=max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)