"""
Скрипт обучения CNN-классификатора патчей на датасете EuroSAT.

Запуск из корня проекта:
    python -m src.train
"""
import time

import torch
import torch.nn as nn
from torch.optim import Adam

from src.config import DEVICE, NUM_EPOCHS, LEARNING_RATE, MODEL_PATH
from src.dataset import get_dataloaders
from src.model import PatchClassifierCNN


def evaluate(model: nn.Module, loader, criterion) -> tuple[float, float]:
    """Считает среднюю потерю и точность на переданном DataLoader'е."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train():
    print(f"Используемое устройство: {DEVICE}")

    train_loader, val_loader, test_loader, class_names = get_dataloaders()
    print(f"Классы: {class_names}")
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    model = PatchClassifierCNN(num_classes=len(class_names)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        start = time.time()
        model.train()

        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            predicted = outputs.argmax(dim=1)
            running_correct += (predicted == labels).sum().item()
            running_total += labels.size(0)

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total

        val_loss, val_acc = evaluate(model, val_loader, criterion)
        elapsed = time.time() - start

        print(
            f"Эпоха {epoch:02d}/{NUM_EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"{elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "class_names": class_names,
            }, MODEL_PATH)
            print(f"  -> Новая лучшая модель сохранена (val_acc={val_acc:.4f})")

    # финальная проверка на test-выборке лучшей сохраненной модели
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"\nИтог на тестовой выборке: loss={test_loss:.4f} acc={test_acc:.4f}")
    print(f"Модель сохранена в: {MODEL_PATH}")


if __name__ == "__main__":
    train()