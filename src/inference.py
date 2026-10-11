"""
Инференс: классификация целого снимка патчами фиксированного размера.

Снимок режется на сетку патчей PATCH_SIZE x PATCH_SIZE (без overlap).
Каждый патч классифицируется моделью, итоговая маска того же размера,
что исходный снимок, но "блочная" — все пиксели одного патча получают
один класс.
"""
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

from src.config import DEVICE, MODEL_PATH, PATCH_SIZE
from src.model import PatchClassifierCNN

_NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])


class PatchClassifier:
    """
    Обертка над обученной моделью: загружает веса один раз,
    предоставляет метод classify_image для целого снимка.
    """

    def __init__(self, model_path: str | Path = MODEL_PATH):
        checkpoint = torch.load(model_path, map_location=DEVICE)
        self.class_names: list[str] = checkpoint["class_names"]

        self.model = PatchClassifierCNN(num_classes=len(self.class_names))
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(DEVICE)
        self.model.eval()

    def classify_image(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Классифицирует снимок патчами.

        Args:
            image: numpy-массив (H, W, 3), dtype=uint8

        Returns:
            class_grid: numpy-массив (grid_h, grid_w) с индексами классов,
                        по одному значению на патч
            class_mask_full: numpy-массив (H_cropped, W_cropped) с индексами
                        классов, растянутыми на полный размер снимка
                        (H_cropped/W_cropped — размер, обрезанный до кратности PATCH_SIZE)
        """
        h, w = image.shape[0], image.shape[1]
        grid_h = h // PATCH_SIZE
        grid_w = w // PATCH_SIZE

        if grid_h == 0 or grid_w == 0:
            raise ValueError(
                f"Снимок слишком мал ({h}x{w}) для патчей размера {PATCH_SIZE}x{PATCH_SIZE}"
            )

        # обрезаем снимок до размера, кратного PATCH_SIZE
        cropped = image[: grid_h * PATCH_SIZE, : grid_w * PATCH_SIZE]

        patches = []
        for i in range(grid_h):
            for j in range(grid_w):
                patch = cropped[
                    i * PATCH_SIZE:(i + 1) * PATCH_SIZE,
                    j * PATCH_SIZE:(j + 1) * PATCH_SIZE,
                ]
                patches.append(self._preprocess_patch(patch))

        batch = torch.stack(patches).to(DEVICE)  # (grid_h*grid_w, 3, PATCH_SIZE, PATCH_SIZE)

        with torch.no_grad():
            outputs = self.model(batch)
            predictions = outputs.argmax(dim=1).cpu().numpy()

        class_grid = predictions.reshape(grid_h, grid_w)

        # растягиваем сетку классов на полный размер снимка (по пикселям)
        class_mask_full = np.repeat(
            np.repeat(class_grid, PATCH_SIZE, axis=0), PATCH_SIZE, axis=1
        )

        return class_grid, class_mask_full

    @staticmethod
    def _preprocess_patch(patch: np.ndarray) -> torch.Tensor:
        """uint8 (H, W, 3) -> нормализованный тензор (3, H, W)."""
        tensor = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
        tensor = _NORMALIZE(tensor)
        return tensor
