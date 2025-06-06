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
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def compute_accuracy(preds, labels):
    pred_labels = np.argmax(preds, axis=1)
    return (pred_labels == labels).mean()