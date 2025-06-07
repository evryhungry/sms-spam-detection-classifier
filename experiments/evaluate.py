import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

def compute_metrics(y_true, y_pred):
    return(
        'precision', precision_score(y_true, y_pred),
        'recall', recall_score(y_true, y_pred),
        'f1', f1_score(y_true, y_pred),
    )

def save_to_csv(path,  results_dict):
    df = pd.DataFrame(results_dict)
    df.to_csv(path, index=True)