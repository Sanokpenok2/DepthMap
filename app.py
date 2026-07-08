"""
Веб-интерфейс DepthMap: карта глубины по стереопаре.

Запуск:
    python app.py
"""

from __future__ import annotations

import cv2
import gradio as gr
import numpy as np

from depth_map import compute_stereo_disparity, measure_distance, load_calibration

COLORMAPS = ["JET", "TURBO", "MAGMA", "INFERNO", "VIRIDIS", "PLASMA", "BONE"]


def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _file_path(upload) -> str | None:
    if upload is None:
        return None
    if isinstance(upload, str):
        return upload
    return getattr(upload, "name", None) or str(upload)


def run_depth_map(
    left_img,
    right_img,
    calib_file,
    method,
    num_disparities,
    block_size,
    min_disparity,
    use_wls,
    wls_lambda,
    wls_sigma,
    colormap,
):
    left_path = _file_path(left_img)
    right_path = _file_path(right_img)
    if not left_path or not right_path:
        return None, None, None, "Загрузите левое и правое изображения.", None

    calib_path = _file_path(calib_file)
    try:
        result = compute_stereo_disparity(
            left_path,
            right_path,
            method=method,
            num_disparities=int(num_disparities),
            block_size=int(block_size),
            min_disparity=int(min_disparity),
            wls=use_wls,
            wls_lambda=float(wls_lambda),
            wls_sigma=float(wls_sigma),
            colormap=colormap,
            calib_path=calib_path,
        )
    except ValueError as exc:
        return None, None, None, f"Ошибка: {exc}", None

    status = "\n".join(result.log)
    if result.rectified:
        status += "\nРектификация применена."

    Q = load_calibration(calib_path)["Q"] if calib_path else None
    measure_cache = {
        "disparity_float": result.disparity_float,
        "Q": Q,
    }

    return (
        _bgr_to_rgb(result.disparity_color),
        _bgr_to_rgb(result.left_gray),
        _bgr_to_rgb(result.right_gray),
        status,
        measure_cache,
    )


def measure_at_point(
    measure_cache,
    measure_window,
    focal,
    baseline,
    evt: gr.SelectData,
):
    if measure_cache is None:
        return "Сначала постройте карту глубины."

    x, y = evt.index
    Q = measure_cache.get("Q")
    disp_float = measure_cache["disparity_float"]

    focal_val = float(focal) if focal and focal > 0 else None
    baseline_val = float(baseline) if baseline and baseline > 0 else None

    dist, disp_val = measure_distance(
        disp_float,
        int(x),
        int(y),
        int(measure_window),
        Q,
        focal_val,
        baseline_val,
    )
    if dist is None:
        if disp_val is not None:
            return (
                f"({x}, {y}): диспаритет {disp_val:.2f} px. "
                "Для расстояния укажите калибровку или focal/baseline."
            )
        return f"({x}, {y}): нет данных о диспаритете в этой точке."

    unit = "ед. (square-size)" if Q is not None else "мм"
    return f"({x}, {y}): {dist:.1f} {unit} (диспаритет {disp_val:.2f} px)"


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="DepthMap") as demo:
        gr.Markdown("# DepthMap\nПостроение карты глубины по стереопаре.")

        with gr.Row():
            with gr.Column(scale=1):
                left_input = gr.Image(label="Левое изображение", type="filepath")
                right_input = gr.Image(label="Правое изображение", type="filepath")
                calib_input = gr.File(
                    label="Файл калибровки (.npz, опционально)",
                    file_types=[".npz"],
                )

                with gr.Accordion("Параметры сопоставления", open=True):
                    method = gr.Radio(
                        ["sgbm", "bm"],
                        value="sgbm",
                        label="Алгоритм",
                    )
                    num_disparities = gr.Slider(
                        16, 1024, value=128, step=16, label="Диапазон диспаритетов"
                    )
                    block_size = gr.Slider(
                        3, 21, value=5, step=2, label="Размер блока"
                    )
                    min_disparity = gr.Slider(
                        0, 64, value=0, step=1, label="Минимальный диспаритет"
                    )
                    colormap = gr.Dropdown(
                        COLORMAPS, value="JET", label="Цветовая палитра"
                    )

                with gr.Accordion("WLS-фильтр", open=False):
                    use_wls = gr.Checkbox(value=False, label="Включить WLS")
                    wls_lambda = gr.Slider(
                        1000, 20000, value=8000, step=500, label="Lambda"
                    )
                    wls_sigma = gr.Slider(
                        0.5, 5.0, value=1.5, step=0.1, label="Sigma"
                    )

                with gr.Accordion("Измерение расстояния", open=False):
                    measure_window = gr.Slider(
                        3, 21, value=5, step=2, label="Окно усреднения (пикс.)"
                    )
                    focal = gr.Number(
                        value=None,
                        label="Фокусное расстояние (пикс., без калибровки)",
                    )
                    baseline = gr.Number(
                        value=None,
                        label="База камер (мм, без калибровки)",
                    )

                run_btn = gr.Button("Построить карту глубины", variant="primary")

            with gr.Column(scale=2):
                disparity_out = gr.Image(label="Карта диспаритета")
                with gr.Row():
                    left_rect_out = gr.Image(label="Левое (после ректификации)")
                    right_rect_out = gr.Image(label="Правое (после ректификации)")
                depth_status = gr.Textbox(label="Статус", lines=4)
                measure_result = gr.Textbox(
                    label="Измерение по клику",
                    lines=2,
                    interactive=False,
                )
                measure_cache = gr.State(None)

        run_btn.click(
            run_depth_map,
            inputs=[
                left_input,
                right_input,
                calib_input,
                method,
                num_disparities,
                block_size,
                min_disparity,
                use_wls,
                wls_lambda,
                wls_sigma,
                colormap,
            ],
            outputs=[
                disparity_out,
                left_rect_out,
                right_rect_out,
                depth_status,
                measure_cache,
            ],
        )

        disparity_out.select(
            measure_at_point,
            inputs=[measure_cache, measure_window, focal, baseline],
            outputs=measure_result,
        )

    return demo


def main() -> None:
    demo = build_ui()
    demo.launch()


if __name__ == "__main__":
    main()
