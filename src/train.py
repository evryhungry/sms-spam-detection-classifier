import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm

from src.dataset import SpamHamDataset
from src.model import SpamHamClassifier
from src.utils import set_seed, compute_accuracy


def train_loop(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Train", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def valid_loop(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Valid", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            all_preds.append(logits.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    accuracy = compute_accuracy(all_preds, all_labels)

    return avg_loss, accuracy


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model_name = "bert-base-uncased"
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 학습 데이터 경로
    train_csv = os.path.join(project_root, "data", "processed", "train.csv")
    valid_csv = os.path.join(project_root, "data", "processed", "valid.csv")

    # 하이퍼파라미터
    epochs = 3
    train_batch_size = 16
    valid_batch_size = 32
    learning_rate = 2e-5
    max_length = 128

    train_dataset = SpamHamDataset(
        csv_path=train_csv,
        tokenizer_name=model_name,
        max_length=max_length,
    )
    valid_dataset = SpamHamDataset(
        csv_path=valid_csv,
        tokenizer_name=model_name,
        max_length=max_length,
    )

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=False)

    # 모델, 옵티마이저, 스케줄러 정의
    model = SpamHamClassifier(model_name=model_name, num_labels=2)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    checkpoint_dir = os.path.join(project_root, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 학습 및 검증 루프
    best_valid_acc = 0.0
    for epoch in range(1, epochs + 1):
        print(f"\n========== Epoch {epoch}/{epochs} ==========")
        train_loss = train_loop(model, train_loader, optimizer, scheduler, device)
        valid_loss, valid_acc = valid_loop(model, valid_loader, device)

        print(f"▶ Train Loss : {train_loss:.4f}")
        print(f"▶ Valid Loss : {valid_loss:.4f} | Valid Acc : {valid_acc:.4f}")

        # 모델 저장
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
        save_path = os.path.join(checkpoint_dir, f"best_model_epoch{epoch}.pt")
        torch.save(model.state_dict(), save_path)
        print(f"▶ New model saved ▶ {save_path}")

    print("\nTraining finished.")
    print(f"Best Valid Accuracy = {best_valid_acc:.4f}")


if __name__ == "__main__":
    main()