"""
Загрузка датасета EuroSAT (RGB) и подготовка DataLoader'ов.

При первом запуске torchvision сам скачает архив EuroSAT (~90 МБ)
в папку data/ и разложит изображения по подпапкам классов.
"""
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import EuroSAT

from src.config import DATA_DIR, BATCH_SIZE, VAL_SPLIT, TEST_SPLIT, PATCH_SIZE


def get_transforms():
    """Трансформации: ресайз под размер патча + нормализация ImageNet-стайл."""
    return transforms.Compose([
        transforms.Resize((PATCH_SIZE, PATCH_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])


def load_eurosat_dataset():
    """
    Скачивает (если нужно) и загружает EuroSAT RGB датасет.
    Возвращает полный Dataset.
    """
    transform = get_transforms()
    dataset = EuroSAT(
        root=str(DATA_DIR),
        download=True,
        transform=transform,
    )
    return dataset


def get_dataloaders():
    """
    Делит датасет на train/val/test и возвращает DataLoader'ы.

    Returns:
        train_loader, val_loader, test_loader, class_names (list[str])
    """
    dataset = load_eurosat_dataset()

    n_total = len(dataset)
    n_val = int(n_total * VAL_SPLIT)
    n_test = int(n_total * TEST_SPLIT)
    n_train = n_total - n_val - n_test

    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, dataset.classes