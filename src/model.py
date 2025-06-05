import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

class SpamHamClassifier(nn.Module):
    """
    HuggingFace의 AutoModelForSequenceClassification을 사용해
    입력 문장이 ham(0)인지 spam(1)인지 분류하는 모델.
    """

    def __init__(self, model_name: str = "bert-base-uncased", num_labels: int = 2):
        super().__init__()
        # pretrained 모델 불러오기 (classification head 포함)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )

    def forward(self, input_ids, attention_mask, labels=None):
        """
        labels를 넣으면 (loss, logits, …) 를 반환.
        labels를 None으로 두면 (logits, …) 를 반환.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs