"""
Архитектура CNN для патчевой классификации спектральных/RGB снимков.

Простая, но эффективная сверточная сеть под патчи 64x64 — такого размера
достаточно для EuroSAT и не требует часов обучения даже на небольшой GPU.
"""
import torch
import torch.nn as nn

from src.config import NUM_CLASSES, IMAGE_CHANNELS


class PatchClassifierCNN(nn.Module):
    """
    4 сверточных блока (conv -> batchnorm -> relu -> maxpool) + классификатор.
    Вход: (batch, 3, 64, 64) -> Выход: (batch, NUM_CLASSES) логиты.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, in_channels: int = IMAGE_CHANNELS):
        super().__init__()

        self.features = nn.Sequential(
            # Блок 1: 64x64 -> 32x32
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Блок 2: 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Блок 3: 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Блок 4: 8x8 -> 4x4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x