"""
Детекция изменений: сравнение классифицированных сеток T1 и T2.

Строит:
- матрицу переходов (сколько патчей перешло из класса A в класс B)
- булеву маску "где произошло изменение класса"
- RGB-визуализацию карты изменений
"""
from dataclasses import dataclass, field

import numpy as np

from src.config import CLASS_COLORS, PATCH_SIZE


@dataclass
class ChangeResult:
    """Результат сравнения двух классифицированных снимков."""
    transition_matrix: np.ndarray          # (num_classes, num_classes), [i, j] = кол-во патчей класс_i -> класс_j
    changed_mask_grid: np.ndarray          # (grid_h, grid_w) bool, True где класс изменился
    change_percentage: float               # доля изменившихся патчей, 0-100
    transitions_list: list[dict] = field(default_factory=list)  # человекочитаемый список переходов


def compute_change(
    class_grid_t1: np.ndarray,
    class_grid_t2: np.ndarray,
    class_names: list[str],
) -> ChangeResult:
    """
    Сравнивает две сетки классов одинакового размера.

    Args:
        class_grid_t1: (grid_h, grid_w) индексы классов до
        class_grid_t2: (grid_h, grid_w) индексы классов после
        class_names: список имен классов по индексу

    Returns:
        ChangeResult с матрицей переходов и статистикой
    """
    if class_grid_t1.shape != class_grid_t2.shape:
        raise ValueError(
            f"Сетки классов разного размера: {class_grid_t1.shape} vs {class_grid_t2.shape}. "
            f"Снимки должны быть приведены к одному размеру перед классификацией."
        )

    num_classes = len(class_names)
    transition_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    for i in range(num_classes):
        for j in range(num_classes):
            transition_matrix[i, j] = np.sum(
                (class_grid_t1 == i) & (class_grid_t2 == j)
            )

    changed_mask_grid = class_grid_t1 != class_grid_t2
    total_patches = class_grid_t1.size
    changed_patches = int(np.sum(changed_mask_grid))
    change_percentage = (changed_patches / total_patches) * 100.0

    transitions_list = _build_transitions_list(transition_matrix, class_names)

    return ChangeResult(
        transition_matrix=transition_matrix,
        changed_mask_grid=changed_mask_grid,
        change_percentage=change_percentage,
        transitions_list=transitions_list,
    )


def _build_transitions_list(transition_matrix: np.ndarray, class_names: list[str]) -> list[dict]:
    """
    Превращает матрицу переходов в список словарей вида:
    {"from": "Forest", "to": "Residential", "count": 5}
    Только реальные переходы (count > 0), без диагонали "класс не изменился".
    """
    result = []
    num_classes = len(class_names)
    for i in range(num_classes):
        for j in range(num_classes):
            if i == j:
                continue
            count = int(transition_matrix[i, j])
            if count > 0:
                result.append({
                    "from": class_names[i],
                    "to": class_names[j],
                    "count": count,
                })
    result.sort(key=lambda x: x["count"], reverse=True)
    return result


def render_change_map(
    changed_mask_grid: np.ndarray,
    class_grid_t2: np.ndarray,
    class_names: list[str],
) -> np.ndarray:
    """
    Строит RGB-изображение карты изменений.

    Патчи без изменений — приглушенный серый.
    Патчи с изменением класса — закрашены цветом НОВОГО класса (из CLASS_COLORS),
    чтобы наглядно показать, во что превратилась территория.

    Returns:
        numpy-массив (grid_h * PATCH_SIZE, grid_w * PATCH_SIZE, 3), dtype=uint8
    """
    grid_h, grid_w = changed_mask_grid.shape
    pixel_map = np.full((grid_h, grid_w, 3), fill_value=200, dtype=np.uint8)  # серый фон

    for i in range(grid_h):
        for j in range(grid_w):
            if changed_mask_grid[i, j]:
                new_class = class_names[class_grid_t2[i, j]]
                color = CLASS_COLORS.get(new_class, (255, 255, 255))
                pixel_map[i, j] = color

    # растягиваем до полного размера изображения в пикселях
    full_map = np.repeat(
        np.repeat(pixel_map, PATCH_SIZE, axis=0), PATCH_SIZE, axis=1
    )
    return full_map


def render_class_mask(class_grid: np.ndarray, class_names: list[str]) -> np.ndarray:
    """
    Строит RGB-визуализацию обычной (не change-) маски классов —
    полезно показать, как классифицирован T1 и T2 по отдельности.

    Returns:
        numpy-массив (grid_h * PATCH_SIZE, grid_w * PATCH_SIZE, 3), dtype=uint8
    """
    grid_h, grid_w = class_grid.shape
    pixel_map = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    for i in range(grid_h):
        for j in range(grid_w):
            class_name = class_names[class_grid[i, j]]
            pixel_map[i, j] = CLASS_COLORS.get(class_name, (255, 255, 255))

    full_map = np.repeat(
        np.repeat(pixel_map, PATCH_SIZE, axis=0), PATCH_SIZE, axis=1
    )
    return full_map
def render_overlay_on_image(
    image: np.ndarray,
    changed_mask_grid: np.ndarray,
    class_grid_t2: np.ndarray,
    class_names: list[str],
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Накладывает полупрозначную цветную подсветку изменений прямо на реальный снимок,
    плюс рамку вокруг каждого изменившегося патча. Так видно и сам снимок,
    и где конкретно произошли изменения.

    Args:
        image: исходный снимок (H, W, 3), uint8, тот же размер что подавался
               в classify_image (после ensure_same_size)
        changed_mask_grid: (grid_h, grid_w) bool, True где класс изменился
        class_grid_t2: (grid_h, grid_w) индексы классов "после",
                       определяет цвет подсветки (во что превратилась территория)
        class_names: список имен классов
        alpha: прозрachность подсветки, 0 = не видна, 1 = сплошной цвет

    Returns:
        numpy-массив (H_cropped, W_cropped, 3), uint8 — снимок с оверлеем,
        обрезанный до размера, кратного PATCH_SIZE (как и маски классов)
    """
    grid_h, grid_w = changed_mask_grid.shape
    cropped = image[: grid_h * PATCH_SIZE, : grid_w * PATCH_SIZE].copy()
    overlay = cropped.astype(np.float32)

    border_thickness = max(2, PATCH_SIZE // 20)

    for i in range(grid_h):
        for j in range(grid_w):
            if not changed_mask_grid[i, j]:
                continue

            new_class = class_names[class_grid_t2[i, j]]
            color = np.array(CLASS_COLORS.get(new_class, (255, 255, 255)), dtype=np.float32)

            y0, y1 = i * PATCH_SIZE, (i + 1) * PATCH_SIZE
            x0, x1 = j * PATCH_SIZE, (j + 1) * PATCH_SIZE

            # полупрозрачная подсветка всего патча
            patch_region = overlay[y0:y1, x0:x1]
            overlay[y0:y1, x0:x1] = patch_region * (1 - alpha) + color * alpha

            # сплошная рамка по контуру патча для четкой границы
            overlay[y0:y0 + border_thickness, x0:x1] = color
            overlay[y1 - border_thickness:y1, x0:x1] = color
            overlay[y0:y1, x0:x0 + border_thickness] = color
            overlay[y0:y1, x1 - border_thickness:x1] = color

    return np.clip(overlay, 0, 255).astype(np.uint8)
def generate_text_report(
    result: "ChangeResult",
    class_names: list[str],
    patch_area_m2: float | None = None,
) -> str:
    """
    Строит человекочитаемое текстовое описание изменений территории
    на основе найденных переходов между классами.

    Args:
        result: результат compute_change
        class_names: список имен классов
        patch_area_m2: площадь одного патча в квадратных метрах, если известна
                       (для Sentinel-2 10м/пиксель и патча 64x64: 640*640 = 409600 м2).
                       Если None — площадь в отчете не указывается, только количество патчей.

    Returns:
        Строка с текстовым отчетом на русском языке.
    """
    lines: list[str] = []

    total_patches = result.changed_mask_grid.size
    changed_patches = int(np.sum(result.changed_mask_grid))

    if changed_patches == 0:
        return (
            "Значимых изменений территории между снимками не обнаружено. "
            "Классификация объектов на обоих снимках совпадает."
        )

    lines.append(
        f"Обнаружены изменения на {result.change_percentage:.1f}% территории "
        f"({changed_patches} из {total_patches} участков снимка)."
    )
    lines.append("")
    lines.append("Основные выявленные изменения:")

    # Группируем переходы по целевому классу (во что превратилось),
    # чтобы текст звучал как связный рассказ, а не просто список пар
    by_target: dict[str, list[dict]] = {}
    for t in result.transitions_list:
        by_target.setdefault(t["to"], []).append(t)

    # сортируем группы по суммарному количеству патчей, чтобы самые
    # значимые изменения были в начале отчета
    sorted_targets = sorted(
        by_target.items(),
        key=lambda kv: sum(t["count"] for t in kv[1]),
        reverse=True,
    )

    class_name_ru = {
        "AnnualCrop": "сельскохозяйственные культуры",
        "Forest": "лес",
        "HerbaceousVegetation": "травянистая растительность",
        "Highway": "автомагистраль/дорога",
        "Industrial": "промышленная застройка",
        "Pasture": "пастбище",
        "PermanentCrop": "многолетние насаждения",
        "Residential": "жилая застройка",
        "River": "река",
        "SeaLake": "водоем (море/озеро)",
    }

    for target_class, transitions in sorted_targets:
        total_count = sum(t["count"] for t in transitions)
        target_ru = class_name_ru.get(target_class, target_class)

        sources_text = ", ".join(
            f"{class_name_ru.get(t['from'], t['from'])} ({t['count']} уч.)"
            for t in sorted(transitions, key=lambda x: x["count"], reverse=True)
        )

        area_text = ""
        if patch_area_m2 is not None:
            area_ha = (total_count * patch_area_m2) / 10_000  # м2 -> га
            area_text = f", площадь ≈ {area_ha:.2f} га"

        lines.append(
            f"— В «{target_ru}» превратилось {total_count} уч.{area_text}, "
            f"ранее это было: {sources_text}."
        )

    return "\n".join(lines)
