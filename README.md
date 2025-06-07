# sms-spam-detection-classifier

한 줄 요약: 문자 메시지 텍스트를 스팸(Spam) 또는 정상(Ham)으로 분류하는 딥러닝 기반 분류 모델을 만들어 보자

## 목차
- [소개](#소개)
- [기능(Features)](#기능features)
- [사용 기술(Tech Stack)](#사용-기술tech-stack)
- [파일 구조(Project Structure)](#파일-구조project-structure)
- [설치 및 실행(Installation & Usage)](#설치-및-실행installation--usage)
- [예시(Examples)](#예시examples)
- [향후 개선 사항(Contributing)](#향후-개선-사항)

---

## 소개
- SMS 데이터를 토큰화하고, LSTM 기반 모델을 학습시켜 스팸 메시지를 자동으로 분류  
- 데이터 전처리 → 데이터셋 로딩 → 모델 학습 → 추론(inference) 순서의 파이프라인 구성
- 정확도 및 손실 그래프를 위한 data_exploration.ipynb를 구성

## 기능Features
- **데이터 전처리** (`data_preprocessing.py`)  
  - Raw CSV/텍스트 파일 로드  
  - 중복 제거, 결측치 처리  
  - 토큰화 & 패딩 → 정수 인덱스 시퀀스 변환  
- **커스텀 데이터셋** (`dataset.py`)  
  - `torch.utils.data.Dataset` 상속  
  - 배치 단위로 텍스트 + 레이블 반환  
- **모델 정의** (`model.py`)  
  - 임베딩 → LSTM → Fully-Connected 분류기  
  - 드롭아웃, 배치 정규화 등 옵션 제공  
- **학습 스크립트** (`train.py`)  
  - 학습/검증 루프  
  - 손실(loss) 및 정확도(accuracy) 모니터링  
  - 체크포인트 저장/로드  
- **추론 모듈** (`inference.py`)  
  - 학습된 모델 불러오기  
  - 새로운 SMS 텍스트를 입력 받아 스팸 여부 예측  
- **유틸리티** (`utils.py`)  
  - 토큰 사전(vocabulary) 저장/불러오기  
  - 학습 곡선 시각화(정확도, 손실 그래프)

## 사용-기술tech-stack
- **언어 & 프레임워크**: Python, PyTorch  
- **데이터 처리**: pandas, NumPy, scikit-learn  
- **토크나이저**: NLTK (또는 HuggingFace Tokenizers)  
- **시각화**: Matplotlib, Seaborn  
- **실험 관리(선택)**: TensorBoard, Weights & Biases


## 파일-구조project-structure
```
sms-spam-detection-classifier
├── checkpoints
│   ├── best_model_epochX.pt
├── data
│   ├── processed
│   │   ├── test.csv
│   │   ├── train.csv
│   │   └── valid.csv
│   └── raw
│       ├── spamhamdata.xls
│       └── spamhamdata.xlsx
├── notebooks
│   └── data_exploration.ipynb
├── README.md
├── requirements.txt
└── src
    ├── __init__.py
    ├── __pycache__
    │   ├── dataset.cpython-310.pyc
    │   ├── model.cpython-310.pyc
    │   └── utils.cpython-310.pyc
    ├── data_preprocessing.py
    ├── dataset.py
    ├── inference.py
    ├── model.py
    ├── train.py
    └── utils.py
```

## 설치-및-실행installation--usage
1. 프로젝트 클론 & 디렉터리 이동  
   ```bash
   git clone https://github.com/username/spam-sms-classifier.git
   cd spam-sms-classifier
2. 의존 패키지 설치
   ```bash
   pip install -r requirements.txt
3. 데이터 전처리
   ```bash
   python data_preprocessing.py \
    --input ./raw_data/sms.csv \
    --output ./processed_data/ \
    --vocab_size 10000 \
    --max_len 50
4. 모델 학습
   ```bash
   python train.py \
    --data_dir ./processed_data/ \
    --epochs 10 \
    --batch_size 64 \
    --lr 1e-3 \
    --save_dir ./checkpoints/
5. 추론 실행 &rightarrow; data_exploration.ipynb (권장)

## 예시examples
- jupyter 사용 로그 그래프 확인해서 넣어둘 것.

## 향후-개선-사항
- 다국어 지원이 현제 실험이 안되어 있는 상황
- 모은 정보량이 5000개의 문장으로 더 많은 데이터가 필요하다고 생각
  
