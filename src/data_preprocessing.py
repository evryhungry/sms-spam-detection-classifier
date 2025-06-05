import os
import pandas as pd
from sklearn.model_selection import train_test_split

input_path  = "data/raw/spamhamdata.xls"
output_dir  = "data/processed"

def load_and_split_excel(
    input_path: str,
    output_dir: str,
    test_size: float = 0.1,
    valid_size: float = 0.1,
    random_state: int = 42,
):

    # 1) Excel 로드 (openpyxl 엔진 사용)
    df = pd.read_excel(input_path, engine='xlrd')

    # 컬럼명 통일: ['label', 'text'] 형태라고 가정
    # 실제 엑셀 칼럼명이 다르면 수정 필요
    df.columns = ['label', 'text']

    # 2) 결측치 제거 & 소문자 변환
    df = df.dropna(subset=['label', 'text']).copy()
    df['text'] = df['text'].astype(str).str.strip().str.lower()

    # 3) 레이블 라벨링: ham→0, spam→1
    df['label_id'] = df['label'].map({'ham': 0, 'spam': 1})

    # 4) train / temp(valid+test) 분리
    train_df, temp_df = train_test_split(
        df,
        test_size=(test_size + valid_size),
        stratify=df['label_id'],
        random_state=random_state,
    )

    # temp → valid / test 분리
    relative_test_size = test_size / (test_size + valid_size)
    valid_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        stratify=temp_df['label_id'],
        random_state=random_state,
    )

    # 5) output_dir 생성 (없으면)
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, 'train.csv')
    valid_path = os.path.join(output_dir, 'valid.csv')
    test_path  = os.path.join(output_dir, 'test.csv')

    # CSV로 저장 (UTF-8-sig: Excel에서 한글 깨짐 방지용)
    train_df.to_csv(train_path, index=False, encoding='utf-8-sig')
    valid_df.to_csv(valid_path, index=False, encoding='utf-8-sig')
    test_df.to_csv(test_path, index=False, encoding='utf-8-sig')

    print(f"▶ Train size   : {len(train_df)}")
    print(f"▶ Valid size   : {len(valid_df)}")
    print(f"▶ Test size    : {len(test_df)}")
    print(f"저장 위치 ▶ {output_dir}")


if __name__ == "__main__":
    # 실행 시점을 프로젝트 루트로 가정
    # data/raw/spamhamdata.xls → data/processed/ 하위에 CSV 저장
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_excel_path = os.path.join(project_root, "data", "raw", "spamhamdata.xls")
    processed_dir   = os.path.join(project_root, "data", "processed")

    load_and_split_excel(raw_excel_path, processed_dir)