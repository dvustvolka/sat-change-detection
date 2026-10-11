"""
Центральный конфиг проекта: пути, классы, гиперпараметры.
"""
from pathlib import Path
import torch

# --- Пути ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "patch_classifier.pt"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# --- Классы EuroSAT (RGB-версия, 10 классов земной поверхности) ---
CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]
NUM_CLASSES = len(CLASS_NAMES)

# Цвета для визуализации классов на карте изменений (RGB, 0-255)
CLASS_COLORS = {
    "AnnualCrop": (255, 211, 0),
    "Forest": (34, 139, 34),
    "HerbaceousVegetation": (154, 205, 50),
    "Highway": (105, 105, 105),
    "Industrial": (178, 34, 34),
    "Pasture": (173, 255, 47),
    "PermanentCrop": (218, 165, 32),
    "Residential": (255, 0, 0),
    "River": (0, 191, 255),
    "SeaLake": (0, 0, 139),
}

# --- Параметры патчей и модели ---
PATCH_SIZE = 64          # размер патча в пикселях (как в EuroSAT)
IMAGE_CHANNELS = 3       # RGB
BATCH_SIZE = 64
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# --- Device ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"