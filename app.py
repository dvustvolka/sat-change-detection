"""
Streamlit-приложение: классификация объектов спутниковых снимков
и детекция изменений территории между двумя датами.

Запуск из корня проекта:
    streamlit run app.py
"""
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from src.config import MODEL_PATH
from src.image_io import load_image_as_array, ensure_same_size
from src.inference import PatchClassifier
from src.change_detection import (
    compute_change,
    render_change_map,
    render_class_mask,
    render_overlay_on_image,
    generate_text_report,
)
st.set_page_config(page_title="Детекция изменений территории", layout="wide")


@st.cache_resource
def get_classifier() -> PatchClassifier:
    """Загружает модель один раз и кеширует между запусками UI."""
    return PatchClassifier(MODEL_PATH)


def save_uploaded_file(uploaded_file) -> Path:
    """Сохраняет загруженный через Streamlit файл во временный путь на диске."""
    suffix = Path(uploaded_file.name).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    return Path(tmp.name)


def main():
    st.title("Классификация и детекция изменений по спутниковым снимкам")
    st.markdown(
        "Загрузите два снимка одной территории (до / после). "
        "Поддерживаемые форматы: **PNG, JPG, JPEG, TIFF** (включая многоканальный спектральный TIFF)."
    )

    if not MODEL_PATH.exists():
        st.error(
            f"Модель не найдена по пути {MODEL_PATH}. "
            f"Сначала запусти обучение: python -m src.train"
        )
        return

    col1, col2 = st.columns(2)
    with col1:
        file_before = st.file_uploader("Снимок ДО", type=["png", "jpg", "jpeg", "tif", "tiff"], key="before")
    with col2:
        file_after = st.file_uploader("Снимок ПОСЛЕ", type=["png", "jpg", "jpeg", "tif", "tiff"], key="after")

    if file_before is None or file_after is None:
        st.info("Загрузите оба снимка, чтобы начать анализ.")
        return

    run_button = st.button("Запустить анализ", type="primary")
    if not run_button:
        return

    with st.spinner("Загрузка изображений..."):
        path_before = save_uploaded_file(file_before)
        path_after = save_uploaded_file(file_after)

        try:
            img_before = load_image_as_array(path_before)
            img_after = load_image_as_array(path_after)
        except Exception as e:
            st.error(f"Не удалось загрузить одно из изображений: {e}")
            return
        finally:
            path_before.unlink(missing_ok=True)
            path_after.unlink(missing_ok=True)

    img_before, img_after = ensure_same_size(img_before, img_after)

    st.subheader("Исходные снимки")
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_before, caption="До", use_container_width=True)
    with col2:
        st.image(img_after, caption="После", use_container_width=True)

    with st.spinner("Классификация снимков..."):
        classifier = get_classifier()
        class_names = classifier.class_names

        try:
            grid_before, _ = classifier.classify_image(img_before)
            grid_after, _ = classifier.classify_image(img_after)
        except ValueError as e:
            st.error(str(e))
            return

    st.subheader("Классификация объектов")
    col1, col2 = st.columns(2)
    with col1:
        mask_before = render_class_mask(grid_before, class_names)
        st.image(mask_before, caption="Классы ДО", use_container_width=True)
    with col2:
        mask_after = render_class_mask(grid_after, class_names)
        st.image(mask_after, caption="Классы ПОСЛЕ", use_container_width=True)

    _render_legend(class_names)

    with st.spinner("Анализ изменений..."):
        result = compute_change(grid_before, grid_after, class_names)

    st.subheader("Карта изменений")

    tab1, tab2 = st.tabs(["Изменения на снимке", "Абстрактная карта"])

    with tab1:
        overlay_image = render_overlay_on_image(img_after, result.changed_mask_grid, grid_after, class_names)
        st.image(
            overlay_image,
            caption="Снимок ПОСЛЕ с подсветкой изменившихся участков (цвет = новый класс территории)",
            use_container_width=True,
        )
        _render_legend(class_names)

    with tab2:
        change_map = render_change_map(result.changed_mask_grid, grid_after, class_names)
        st.image(change_map, caption="Серый — без изменений, цвет — новый класс территории", use_container_width=True)

    st.metric("Доля изменившейся территории", f"{result.change_percentage:.1f}%")
    st.subheader("Переходы между классами")
    if result.transitions_list:
        st.table(result.transitions_list)
    else:
        st.success("Изменений класса территории не обнаружено.")
    st.metric("Доля изменившейся территории", f"{result.change_percentage:.1f}%")

    st.subheader("Текстовое описание изменений")
    report_text = generate_text_report(result, class_names)
    st.markdown(report_text.replace("\n", "  \n"))  # два пробела перед \n — перенос строки в markdown

    st.subheader("Переходы между классами")


def _render_legend(class_names: list[str]) -> None:
    """Отображает цветовую легенду классов."""
    from src.config import CLASS_COLORS

    st.caption("Легенда классов:")
    cols = st.columns(len(class_names))
    for col, name in zip(cols, class_names):
        color = CLASS_COLORS.get(name, (255, 255, 255))
        hex_color = "#%02x%02x%02x" % color
        col.markdown(
            f'<div style="background-color:{hex_color};padding:4px;border-radius:4px;'
            f'text-align:center;color:white;font-size:11px;">{name}</div>',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
