import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    """
    재현성을 위해 Python, NumPy, Torch의 랜덤 시드를 고정
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True   # 필요시 해제
    # torch.backends.cudnn.benchmark = False

def compute_accuracy(preds, labels):
    """
    preds: numpy array of shape (N, num_labels) → logits
    labels: numpy array of shape (N,)
    반환: 정확도 (float)
    """
    pred_labels = np.argmax(preds, axis=1)
    return (pred_labels == labels).mean()