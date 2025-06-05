import os
import torch
from transformers import AutoTokenizer

from model import SpamHamClassifier

def predict_single_text(text: str, model, tokenizer, device="cpu", max_length: int = 128):
    """
    단일 문장(text)을 입력받아,
    - 토크나이징 → tensor 변환 → 모델 추론 → 0 또는 1(ham/spam) 반환
    """
    encoding = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.cpu().numpy()
        pred_label = int(logits.argmax(axis=1)[0])
    return pred_label  # 0=ham, 1=spam


if __name__ == "__main__":
    # 1) 장치 설정, 토크나이저, 모델 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoint_dir = os.path.join(project_root, "checkpoints")

    # 가장 최근에 저장된 best_model을 자동으로 찾기 위해, 확장자가 .pt인 파일 중 이름이 가장 뒤인 것을 가져오도록 해도 되고
    # 예시에서는 epoch3로 저장했다고 가정
    model_path = os.path.join(checkpoint_dir, "best_model_epoch3.pt")

    # 모델 초기화 & 가중치 로드
    model = SpamHamClassifier(model_name=model_name, num_labels=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    print("Loaded model from:", model_path)

    # 2) 예측할 샘플 문장 리스트 작성
    sample_texts = [
        "Congrats! You have won a free ticket to Bahamas. Reply YES to claim.",
    ]

    # 3) 예측 수행
    for txt in sample_texts:
        pred = predict_single_text(txt, model, tokenizer, device, max_length=128)
        label_str = "spam" if pred == 1 else "ham"
        print("────────────────────────────────────────")
        print(f"문장: {txt}")
        print(f"예측: {label_str}  (0=ham, 1=spam)")

    print("Inference 완료.")