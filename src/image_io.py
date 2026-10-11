"""
Универсальная загрузка снимков разных форматов: PNG, JPG, JPEG, TIFF
(включая многоканальные спектральные TIFF).

Для текущей модели (обучена на RGB EuroSAT) многоканальные снимки
приводятся к RGB через первые 3 канала или явный выбор каналов.
Это сделано с запасом на будущее расширение под спектральные индексы.
"""
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


def load_image_as_array(path: str | Path) -> np.ndarray:
    """
    Загружает изображение любого поддерживаемого формата в numpy-массив
    формата (H, W, 3), dtype=uint8, значения 0-255.

    Поддерживает: .png, .jpg, .jpeg, .tif, .tiff
    Для многоканальных TIFF берутся первые 3 канала как RGB.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".png", ".jpg", ".jpeg"):
        return _load_with_pillow(path)

    if suffix in (".tif", ".tiff"):
        if RASTERIO_AVAILABLE:
            try:
                return _load_with_rasterio(path)
            except Exception:
                # Если rasterio не смог прочитать (например обычный RGB-tiff
                # без геопривязки, который он иногда не любит) — пробуем Pillow
                return _load_with_pillow(path)
        return _load_with_pillow(path)

    raise ValueError(
        f"Неподдерживаемый формат файла: {suffix}. "
        f"Поддерживаются: .png, .jpg, .jpeg, .tif, .tiff"
    )


def _load_with_pillow(path: Path) -> np.ndarray:
    """Загрузка через Pillow, с приведением к RGB uint8."""
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def _load_with_rasterio(path: Path) -> np.ndarray:
    """
    Загрузка через rasterio. Работает с многоканальными спектральными TIFF.
    Берет первые 3 канала (или меньше, если их меньше 3) как RGB.

    Нормализация через percentile stretch (2%-98%) вместо абсолютного min/max —
    это стандартная практика в remote sensing. Абсолютный min/max очень
    чувствителен к выбросам (яркие облака, блики на воде, засветки сенсора):
    один яркий пиксель растягивает диапазон и давит контраст всей остальной
    сцены, из-за чего изображение выглядит тусклым и плохо различимым для
    модели классификации.
    """
    with rasterio.open(path) as src:
        num_bands = src.count
        bands_to_read = min(3, num_bands)
        # rasterio нумерует каналы с 1
        data = src.read(list(range(1, bands_to_read + 1)))  # (bands, H, W)

    # (bands, H, W) -> (H, W, bands)
    data = np.transpose(data, (1, 2, 0)).astype(np.float32)

    # если каналов меньше 3 (например ЧБ снимок) — дублируем в RGB
    if data.shape[2] == 1:
        data = np.repeat(data, 3, axis=2)
    elif data.shape[2] == 2:
        extra = data.mean(axis=2, keepdims=True)
        data = np.concatenate([data, extra], axis=2)

    data = _percentile_stretch(data, lower_percentile=2.0, upper_percentile=98.0)

    return data.astype(np.uint8)


def _percentile_stretch(
    data: np.ndarray,
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0,
) -> np.ndarray:
    """
    Растягивает каждый канал изображения отдельно по процентилям в диапазон 0-255.

    Нормализация делается покально по каждому каналу (а не по всему изображению
    сразу), потому что разные спектральные каналы могут иметь разную шкалу
    значений (например Red и NIR не обязаны совпадать по диапазону яркости).

    Args:
        data: (H, W, C) float32
        lower_percentile: нижний процентиль, значения ниже обрезаются до 0
        upper_percentile: верхний процентиль, значения выше обрезаются до 255

    Returns:
        (H, W, C) float32, значения в диапазоне 0-255
    """
    result = np.zeros_like(data, dtype=np.float32)

    for c in range(data.shape[2]):
        channel = data[:, :, c]
        low = np.percentile(channel, lower_percentile)
        high = np.percentile(channel, upper_percentile)

        if high > low:
            stretched = (channel - low) / (high - low) * 255.0
        else:
            stretched = np.zeros_like(channel)

        result[:, :, c] = np.clip(stretched, 0, 255)

    return result

def ensure_same_size(img1: np.ndarray, img2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Приводит два изображения к одному размеру (по минимальным H и W),
    обрезая по центру. Нужно для корректного сравнения T1/T2.
    """
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])

    def center_crop(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        start_h = (img.shape[0] - target_h) // 2
        start_w = (img.shape[1] - target_w) // 2
        return img[start_h:start_h + target_h, start_w:start_w + target_w]

    return center_crop(img1, h, w), center_crop(img2, h, w)
