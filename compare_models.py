import os
import time
import pandas as pd
import numpy as np
import torch

from src.utils import set_seed, compute_accuracy
from src.model import SpamHamClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CSV_PATH = 'data/processed/test.csv'
MY_MODEL_CHECKPOINT = 'checkpoints/best_model_epoch1.pt'

# 평가 지표 계산 함수
def compute_metrics(y_true, y_pred):
    return {
        'precision': precision_score(y_true, y_pred),
        'recall':    recall_score(y_true, y_pred),
        'f1':        f1_score(y_true, y_pred),
        'accuracy':  compute_accuracy(y_true, y_pred)
    }

# 1) 나의 모델(SpamHamClassifier) 추론
def run_my_model(test_texts):
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = SpamHamClassifier(model_name='bert-base-uncased', num_labels=2)
    # 체크포인트 로드
    model.load_state_dict(torch.load(MY_MODEL_CHECKPOINT, map_location=DEVICE))
    model.to(DEVICE).eval()

    # 토크나이즈 및 텐서 변환
    enc = tokenizer(
        test_texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    input_ids      = enc['input_ids'].to(DEVICE)
    attention_mask = enc['attention_mask'].to(DEVICE)

    # 예측 및 시간 측정
    start = time.time()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits  = outputs.logits
        preds   = logits.argmax(dim=1).cpu().numpy()
    elapsed_ms = (time.time() - start) * 1000
    return preds, elapsed_ms

# 2) TF-IDF + SVM 추론
def run_tfidf_svm(train_texts, train_labels, test_texts):
    set_seed(42)
    tfidf = TfidfVectorizer(max_features=10000, lowercase=True, token_pattern=r"\b\w+\b")
    X_train = tfidf.fit_transform(train_texts)
    X_test  = tfidf.transform(test_texts)

    clf = LinearSVC(C=1.0, max_iter=10000)
    clf.fit(X_train, train_labels)

    start = time.time()
    preds = clf.predict(X_test)
    elapsed_ms = (time.time() - start) * 1000
    return preds, elapsed_ms

# 3) Transformer(BERT) Zero-shot 추론
def run_transformer(test_texts):
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model     = AutoModelForSequenceClassification.from_pretrained(
                    'distilbert-base-uncased', num_labels=2
                ).to(DEVICE).eval()

    enc = tokenizer(
        test_texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    input_ids      = enc['input_ids'].to(DEVICE)
    attention_mask = enc['attention_mask'].to(DEVICE)

    start = time.time()
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask).logits
        preds  = logits.argmax(dim=1).cpu().numpy()
    elapsed_ms = (time.time() - start) * 1000
    return preds, elapsed_ms

# 메인: 데이터 로드, 분할, 비교 실행, 결과 저장
if __name__ == '__main__':
    set_seed(42)
    # CSV 읽기
    df = pd.read_csv(CSV_PATH)
    texts  = df['text'].astype(str).tolist()
    labels = df['label_id'].astype(int).tolist()
    # train/test 분할
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=42
    )

    # 비교 모델 맵
    experiments = {
        'MyModel (BERT-ft)':    lambda: run_my_model(test_texts),
        'TFIDF + SVM':          lambda: run_tfidf_svm(train_texts, train_labels, test_texts),
        'Zero-shot BERT':       lambda: run_transformer(test_texts),
    }

    results = {}
    for name, fn in experiments.items():
        set_seed(42)
        preds, elapsed_ms = fn()
        metrics = compute_metrics(test_labels, preds)
        metrics['time_ms_per_sample'] = elapsed_ms / len(test_texts)
        results[name] = metrics
        print(f"{name}: {metrics}")

    # 결과 저장
    os.makedirs('experiments/results', exist_ok=True)
    df_res = pd.DataFrame(results).T
    df_res.to_csv('experiments/results/metrics.csv', index=True)
    print('Saved to experiments/results/metrics.csv')
