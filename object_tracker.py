"""
Захват и трекинг объекта в кадре — отдельный модуль.

Здесь собрана вся логика выбора области объекта (ROI) и её сопровождения
между кадрами. Модуль не зависит от стерео/диспаритета, поэтому его можно
тестировать изолированно (в т.ч. на одном видео) или переиспользовать.

Быстрая проверка из командной строки (трекинг по одному видео):
    python object_tracker.py --video left.mp4 --tracker csrt

Программный вызов:
    from object_tracker import ObjectTracker
    trk = ObjectTracker(kind="csrt")
    trk.init(frame_bgr, roi)          # roi = (x, y, w, h)
    ok, roi = trk.update(next_frame)  # ok: bool, roi: (x, y, w, h) | None
"""

from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

from depth_map import display_scale, fit_for_display

Roi = tuple[int, int, int, int]

TRACKER_KINDS = ("csrt", "kcf", "mosse")


def create_raw_tracker(kind: str):
    """Создаёт «сырой» OpenCV-трекер с учётом разных сборок (cv2 / cv2.legacy)."""
    kind = kind.lower()
    if kind not in TRACKER_KINDS:
        raise ValueError(
            f"Неизвестный трекер '{kind}'. Доступны: {', '.join(TRACKER_KINDS)}."
        )
    name = {
        "csrt": "TrackerCSRT_create",
        "kcf": "TrackerKCF_create",
        "mosse": "TrackerMOSSE_create",
    }[kind]
    for mod in (cv2, getattr(cv2, "legacy", None)):
        if mod is None:
            continue
        factory = getattr(mod, name, None)
        if factory is not None:
            return factory()
    raise RuntimeError(
        f"Трекер '{kind}' недоступен в этой сборке OpenCV. "
        "Установите opencv-contrib-python."
    )


def clamp_roi(roi: tuple[float, float, float, float], width: int, height: int) -> Roi:
    """Приводит ROI к целым и обрезает по границам кадра."""
    x, y, rw, rh = roi
    x = int(round(x))
    y = int(round(y))
    rw = int(round(rw))
    rh = int(round(rh))
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    rw = max(1, min(rw, width - x))
    rh = max(1, min(rh, height - y))
    return x, y, rw, rh


def roi_center(roi: Roi) -> tuple[int, int]:
    x, y, rw, rh = roi
    return x + rw // 2, y + rh // 2


def select_object_roi(frame_bgr: np.ndarray, max_display: int = 1200) -> Roi | None:
    """Интерактивный выбор ROI мышью. Возвращает ROI в координатах полного кадра."""
    scale = display_scale(frame_bgr.shape, max_display)
    preview = fit_for_display(frame_bgr, scale)
    window = "Select object (Enter/Space = OK, c = cancel)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    roi = cv2.selectROI(window, preview, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window)
    x, y, rw, rh = roi
    if rw <= 0 or rh <= 0:
        return None
    if scale < 1.0:
        inv = 1.0 / scale
        x, y, rw, rh = int(x * inv), int(y * inv), int(rw * inv), int(rh * inv)
    return clamp_roi((x, y, rw, rh), frame_bgr.shape[1], frame_bgr.shape[0])


def estimate_roi_from_point(
    frame_bgr: np.ndarray,
    point: tuple[int, int],
    *,
    tolerance: int = 16,
    grabcut_refine: bool = True,
    max_fraction: float = 0.6,
) -> Roi | None:
    """Оценивает границы объекта по одной точке (клику).

    Шаг 1: floodFill от точки — связная область похожего цвета → грубая рамка.
    Шаг 2 (опц.): GrabCut внутри расширенной рамки уточняет границы объекта.
    """
    frame_bgr = _as_bgr(frame_bgr)
    h, w = frame_bgr.shape[:2]
    px, py = int(point[0]), int(point[1])
    if not (0 <= px < w and 0 <= py < h):
        return None

    blurred = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
    mask = np.zeros((h + 2, w + 2), np.uint8)
    lo = (tolerance,) * 3
    hi = (tolerance,) * 3
    flags = 4 | cv2.FLOODFILL_FIXED_RANGE | (255 << 8)
    cv2.floodFill(blurred, mask, (px, py), 0, lo, hi, flags)
    region = mask[1:-1, 1:-1]

    ys, xs = np.where(region > 0)
    if xs.size < 20:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    rw, rh = x1 - x0 + 1, y1 - y0 + 1

    # Слишком большая область — вероятно, залило фон: откат к боксу вокруг точки.
    if rw > w * max_fraction and rh > h * max_fraction:
        return None

    roi = (x0, y0, rw, rh)
    if grabcut_refine:
        refined = _grabcut_refine(frame_bgr, roi)
        if refined is not None:
            roi = refined
    return clamp_roi(roi, w, h)


def _grabcut_refine(frame_bgr: np.ndarray, roi: Roi) -> Roi | None:
    """Уточняет рамку объекта GrabCut'ом внутри расширенной области."""
    h, w = frame_bgr.shape[:2]
    x, y, rw, rh = roi
    mx, my = int(rw * 0.25) + 5, int(rh * 0.25) + 5
    ex0 = max(0, x - mx)
    ey0 = max(0, y - my)
    ex1 = min(w, x + rw + mx)
    ey1 = min(h, y + rh + my)
    sub = frame_bgr[ey0:ey1, ex0:ex1]
    if sub.shape[0] < 10 or sub.shape[1] < 10:
        return None
    rect = (x - ex0, y - ey0, rw, rh)
    mask = np.zeros(sub.shape[:2], np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(sub, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
    except Exception:
        return None
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    ys, xs = np.where(fg > 0)
    if xs.size < 20:
        return None
    nx0, nx1 = int(xs.min()), int(xs.max())
    ny0, ny1 = int(ys.min()), int(ys.max())
    return (ex0 + nx0, ey0 + ny0, nx1 - nx0 + 1, ny1 - ny0 + 1)


def select_object_by_click(
    frame_bgr: np.ndarray,
    max_display: int = 1200,
    *,
    tolerance: int = 16,
    grabcut_refine: bool = True,
) -> Roi | None:
    """Клик по объекту → авто-определение рамки. Enter — принять, C/Esc — отмена."""
    frame_bgr = _as_bgr(frame_bgr)
    scale = display_scale(frame_bgr.shape, max_display)
    preview = fit_for_display(frame_bgr, scale)
    inv = 1.0 / scale if scale > 0 else 1.0
    window = "Click object (click=detect, Enter=OK, C/Esc=cancel)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    state: dict[str, Roi | None] = {"roi": None}

    def on_mouse(event: int, mx: int, my: int, flags: int, userdata) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        fx, fy = int(mx * inv), int(my * inv)
        roi = estimate_roi_from_point(
            frame_bgr, (fx, fy), tolerance=tolerance, grabcut_refine=grabcut_refine
        )
        if roi is None:
            # Фолбэк: небольшой бокс вокруг клика.
            side = max(20, int(min(frame_bgr.shape[:2]) * 0.08))
            roi = clamp_roi(
                (fx - side // 2, fy - side // 2, side, side),
                frame_bgr.shape[1],
                frame_bgr.shape[0],
            )
        state["roi"] = roi

    cv2.setMouseCallback(window, on_mouse)

    while True:
        vis = preview.copy()
        roi = state["roi"]
        if roi is not None:
            x, y, rw, rh = roi
            cv2.rectangle(
                vis,
                (int(x * scale), int(y * scale)),
                (int((x + rw) * scale), int((y + rh) * scale)),
                (0, 220, 0),
                2,
            )
        cv2.imshow(window, vis)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 32):  # Enter / Space
            cv2.destroyWindow(window)
            return state["roi"]
        if key in (ord("c"), ord("C"), 27):
            cv2.destroyWindow(window)
            return None


def _as_bgr(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame


def _as_tracker_input(frame: np.ndarray) -> np.ndarray:
    """Трекеры OpenCV ожидают 3-канальное изображение."""
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame


class ObjectTracker:
    """Сессия захвата и сопровождения одного объекта.

    Инкапсулирует тип трекера, текущий ROI и состояние захвата, чтобы логика
    трекинга не была размазана по основному циклу и легко тестировалась.
    """

    def __init__(
        self,
        kind: str = "csrt",
        *,
        smooth: float = 0.6,
        lock_size: bool = True,
        keep_aspect: bool = True,
        max_scale_step: float = 0.05,
        verify: bool = True,
        verify_threshold: float = 0.45,
        max_jump: float = 0.7,
        min_visible: float = 0.4,
        lost_patience: int = 2,
        verify_rel: float = 0.75,
        min_iou: float = 0.5,
        max_size_ratio: float = 1.6,
        reacquire: bool = True,
        reacquire_threshold: float = 0.62,
        reacquire_radius: float = 2.5,
        reacquire_global: bool = False,
        reacquire_interval: int = 5,
        reacquire_scale_min: float = 0.35,
        reacquire_scale_max: float = 1.4,
    ) -> None:
        self.kind = kind.lower()
        if self.kind not in TRACKER_KINDS:
            raise ValueError(
                f"Неизвестный трекер '{kind}'. Доступны: {', '.join(TRACKER_KINDS)}."
            )
        if not 0.0 <= smooth < 1.0:
            raise ValueError("smooth должен быть в диапазоне [0.0, 1.0).")
        # smooth — сила сглаживания рамки: 0 = «как есть», ближе к 1 = плавнее (но с задержкой).
        self.smooth = float(smooth)
        self.lock_size = bool(lock_size)
        # keep_aspect — масштабировать рамку равномерно, сохраняя исходные пропорции
        # (иначе CSRT «схлопывает» стороны неравномерно и обрезает объект).
        self.keep_aspect = bool(keep_aspect)
        # max_scale_step — максимальное относительное изменение масштаба за кадр
        # (ограничивает резкое схлопывание рамки). 0 = без ограничения.
        self.max_scale_step = float(max_scale_step)
        # verify — отклонять «ложный» трекинг: прыжок рамки / слабое совпадение с эталоном.
        self.verify = bool(verify)
        self.verify_threshold = float(verify_threshold)
        # max_jump — макс. смещение центра за кадр в долях диагонали ROI (0 = без лимита).
        self.max_jump = float(max_jump)
        # min_visible — минимальная доля рамки, которая должна оставаться внутри кадра.
        self.min_visible = float(min_visible)
        # lost_patience — сколько подряд «плохих» кадров нужно, чтобы объявить потерю.
        self.lost_patience = max(1, int(lost_patience))
        # verify_rel — отклонять кадр, если score упал ниже EMA*verify_rel (дрейф на соседний объект).
        self.verify_rel = float(verify_rel)
        # min_iou — мин. IoU с предыдущей рамкой (отсекает «перескок» на пересекающийся объект).
        self.min_iou = float(min_iou)
        # max_size_ratio — макс. изменение площади за кадр; сильнее → LOST, а не сжатие рамки.
        self.max_size_ratio = float(max_size_ratio)
        # reacquire — повторный захват объекта после потери по эталонному шаблону.
        self.reacquire = bool(reacquire)
        self.reacquire_threshold = float(reacquire_threshold)
        # reacquire_radius — окно поиска вокруг последней позиции (в долях max(w,h) ROI).
        # По умолчанию ищем только рядом, чтобы не хватать похожий фон в другом конце кадра.
        self.reacquire_radius = float(reacquire_radius)
        # reacquire_global — разрешить поиск по всему кадру (осторожно: ложные срабатывания).
        self.reacquire_global = bool(reacquire_global)
        # reacquire_interval — искать объект не каждый кадр (иначе FPS сильно падает).
        self.reacquire_interval = max(1, int(reacquire_interval))
        # Диапазон масштабов эталона при поиске (объект мог стать меньше/больше).
        self.reacquire_scale_min = float(reacquire_scale_min)
        self.reacquire_scale_max = float(reacquire_scale_max)
        if not 0.05 <= self.reacquire_scale_min <= self.reacquire_scale_max <= 4.0:
            raise ValueError("reacquire_scale_min/max должны быть в (0.05..4] и min<=max.")
        self._tracker = None
        self.roi: Roi | None = None
        # Дробное (несглаженное к целым) состояние рамки для EMA-фильтра.
        self._roi_f: tuple[float, float, float, float] | None = None
        self._locked_size: tuple[float, float] | None = None
        # Исходный размер рамки и текущий сглаженный масштаб относительно него.
        self._init_size: tuple[float, float] | None = None
        self._scale: float = 1.0
        # Эталон объекта (grayscale) для повторного захвата после потери.
        self._template: np.ndarray | None = None
        self._template_size: tuple[int, int] | None = None
        self._fail_streak: int = 0
        self._since_reacquire: int = 0
        self._lost_frames: int = 0
        self._score_ema: float | None = None
        self.ok: bool = False
        self.initialized: bool = False
        self.reacquired: bool = False
        self.last_score: float | None = None

    def init(self, frame: np.ndarray, roi: Roi) -> Roi:
        """Инициализирует трекер по кадру и ROI (x, y, w, h)."""
        img = _as_tracker_input(frame)
        roi = clamp_roi(roi, img.shape[1], img.shape[0])
        self._tracker = create_raw_tracker(self.kind)
        self._tracker.init(img, roi)
        self.roi = roi
        self._roi_f = (float(roi[0]), float(roi[1]), float(roi[2]), float(roi[3]))
        self._locked_size = (float(roi[2]), float(roi[3]))
        self._init_size = (float(roi[2]), float(roi[3]))
        self._scale = 1.0
        self._save_template(img, roi)
        self._fail_streak = 0
        self._since_reacquire = 0
        self._lost_frames = 0
        self.ok = True
        self.initialized = True
        self.reacquired = False
        self.last_score = self._score_roi(img, roi)
        self._score_ema = self.last_score
        return roi

    def _save_template(self, img_bgr: np.ndarray, roi: Roi) -> None:
        x, y, rw, rh = roi
        patch = img_bgr[y : y + rh, x : x + rw]
        if patch.size == 0:
            return
        self._template = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        self._template_size = (rw, rh)

    def _score_roi(self, img_bgr: np.ndarray, box: tuple[float, float, float, float]) -> float:
        """Сходство текущего патча с эталоном (примерно [-1..1])."""
        if self._template is None:
            return 1.0
        x, y, rw, rh = clamp_roi(box, img_bgr.shape[1], img_bgr.shape[0])
        patch = img_bgr[y : y + rh, x : x + rw]
        if patch.size == 0:
            return -1.0
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        templ = cv2.resize(self._template, (gray.shape[1], gray.shape[0]))
        pvar = float(np.var(gray))
        tvar = float(np.var(templ))
        # Однотонный патч: matchTemplate даёт ложные 1.0. Сравниваем среднюю яркость.
        # (однотонный объект тоже имеет var≈0 — его нельзя отсекать как «фон».)
        if pvar < 1.0 or tvar < 1.0:
            diff = abs(float(gray.mean()) - float(templ.mean()))
            return 1.0 - min(1.0, diff / 40.0)
        res = cv2.matchTemplate(gray, templ, cv2.TM_CCOEFF_NORMED)
        score = float(res[0, 0])
        return score if np.isfinite(score) else -1.0

    def _appearance_ok(
        self, img_bgr: np.ndarray, box: tuple[float, float, float, float]
    ) -> bool:
        """Отсекает кусты/тёмные пятна по яркости и контрасту эталона."""
        if self._template is None:
            return True
        x, y, rw, rh = clamp_roi(box, img_bgr.shape[1], img_bgr.shape[0])
        patch = img_bgr[y : y + rh, x : x + rw]
        if patch.size == 0:
            return False
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        templ = cv2.resize(
            self._template, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_AREA
        )
        mean_diff = abs(float(gray.mean()) - float(templ.mean()))
        if mean_diff > 55.0:
            return False
        pstd = float(np.std(gray))
        tstd = float(np.std(templ))
        if tstd < 1.0:
            return pstd < 8.0
        ratio = pstd / tstd
        return 0.35 <= ratio <= 2.8

    def _aspect_ok(self, box: tuple[float, float, float, float]) -> bool:
        """Найденная рамка должна сохранять пропорции эталона."""
        if self._template_size is None:
            return True
        tw, th = self._template_size
        if tw < 1 or th < 1:
            return True
        _, _, w, h = box
        if w < 1 or h < 1:
            return False
        ratio = (float(w) / float(h)) / (float(tw) / float(th))
        return 1.0 / 1.35 <= ratio <= 1.35

    def _visible_fraction(
        self, box: tuple[float, float, float, float], width: int, height: int
    ) -> float:
        x, y, rw, rh = box
        x0, y0 = max(0.0, x), max(0.0, y)
        x1, y1 = min(float(width), x + rw), min(float(height), y + rh)
        inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        area = max(rw * rh, 1.0)
        return inter / area

    def _size_ok(self, box: tuple[float, float, float, float]) -> bool:
        """False, если площадь рамки резко изменилась (CSRT сжался на случайный патч).

        Сравниваем с последней принятой рамкой (не с «вечным» lock), чтобы
        постепенное удаление объекта не блокировало трекинг жёстче нужного.
        """
        if self.max_size_ratio <= 1.0:
            return True
        _, _, w, h = box
        new_a = max(float(w) * float(h), 1.0)
        ref_w = ref_h = None
        if self._roi_f is not None:
            ref_w, ref_h = self._roi_f[2], self._roi_f[3]
        elif self.lock_size and self._locked_size is not None:
            ref_w, ref_h = self._locked_size
        elif self._init_size is not None:
            ref_w, ref_h = self._init_size
        if ref_w is None or ref_h is None:
            return True
        old_a = max(float(ref_w) * float(ref_h), 1.0)
        ratio = new_a / old_a
        return (1.0 / self.max_size_ratio) <= ratio <= self.max_size_ratio


    def _jump_ok(self, box: tuple[float, float, float, float]) -> bool:
        """False, если центр «прыгнул» слишком далеко за один кадр."""
        if self.max_jump <= 0.0 or self._roi_f is None:
            return True
        px, py, pw, ph = self._roi_f
        pcx, pcy = px + pw / 2.0, py + ph / 2.0
        x, y, w, h = box
        cx, cy = x + w / 2.0, y + h / 2.0
        dist = float(np.hypot(cx - pcx, cy - pcy))
        diag = float(np.hypot(max(pw, 1.0), max(ph, 1.0)))
        return dist <= self.max_jump * diag

    @staticmethod
    def _iou(
        a: tuple[float, float, float, float], b: tuple[float, float, float, float]
    ) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x0, y0 = max(ax, bx), max(ay, by)
        x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        union = aw * ah + bw * bh - inter
        return float(inter / union) if union > 0 else 0.0

    def _validate_box(
        self, img_bgr: np.ndarray, box: tuple[float, float, float, float]
    ) -> tuple[bool, float]:
        """Проверяет, что новая рамка — всё ещё наш объект, а не соседний куст/фон.

        Отсекает типичный сбой CSRT: при лёгком пересечении границ рамка
        «переезжает» на более текстурированный соседний объект.
        """
        score = self._score_roi(img_bgr, box)
        self.last_score = score
        if not self.verify:
            return True, score
        # Резкое сжатие/раздувание рамки = потеря, а не «новый объект».
        if not self._size_ok(box):
            return False, score
        if not np.isfinite(score) or score < self.verify_threshold:
            return False, score
        if not self._jump_ok(box):
            return False, score
        if self._visible_fraction(box, img_bgr.shape[1], img_bgr.shape[0]) < self.min_visible:
            return False, score
        if not self._appearance_ok(img_bgr, box):
            return False, score
        if not self._aspect_ok(box):
            return False, score

        # Относительное падение score (постепенный дрейф на другой объект).
        if (
            self._score_ema is not None
            and self.verify_rel > 0.0
            and score < self._score_ema * self.verify_rel
        ):
            return False, score

        if self._roi_f is not None:
            iou = self._iou(self._roi_f, box)
            # Резкий сдвиг рамки на пересекающийся объект при среднем score.
            if self.min_iou > 0.0 and iou < self.min_iou and score < 0.88:
                return False, score
            # Если на старой позиции объект всё ещё «узнаётся» лучше — не перескакиваем.
            # (только при заметном уезде, иначе 5px сдвиг ломает NCC)
            px, py, pw, ph = self._roi_f
            x, y, w, h = box
            dist = float(
                np.hypot((x + w / 2.0) - (px + pw / 2.0), (y + h / 2.0) - (py + ph / 2.0))
            )
            diag = float(np.hypot(max(pw, 1.0), max(ph, 1.0)))
            if dist > 0.2 * diag or iou < max(self.min_iou, 0.55):
                old_score = self._score_roi(img_bgr, self._roi_f)
                if (
                    np.isfinite(old_score)
                    and old_score >= self.verify_threshold
                    and score + 0.03 < old_score
                ):
                    return False, score

        return True, score

    def _search_window(
        self, width: int, height: int
    ) -> tuple[int, int, int, int] | None:
        """Окно поиска для перезахвата: локально вокруг последней ROI или весь кадр.

        Чем дольше цель потеряна, тем шире окно (человек мог отойти с объектом).
        """
        if self.reacquire_global or self._roi_f is None:
            return (0, 0, width, height)
        px, py, pw, ph = self._roi_f
        cx, cy = px + pw / 2.0, py + ph / 2.0
        # Расширение окна: +100% за ~60 кадров потери, максимум ×3.
        expand = 1.0 + min(2.0, self._lost_frames / 60.0)
        rad = self.reacquire_radius * max(pw, ph, 1.0) * expand
        x0 = max(0, int(cx - rad))
        y0 = max(0, int(cy - rad))
        x1 = min(width, int(cx + rad))
        y1 = min(height, int(cy + rad))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        return (x0, y0, x1 - x0, y1 - y0)

    def _reacquire_scales(self) -> list[float]:
        """Масштабы эталона — широкий диапазон, чтобы найти объект дальше (мельче)."""
        lo = self.reacquire_scale_min
        hi = self.reacquire_scale_max
        n = 9
        scales = [float(s) for s in np.geomspace(lo, hi, num=n)]
        if lo <= self._scale <= hi:
            scales.append(float(self._scale))
        return sorted(set(round(s, 4) for s in scales))

    def _try_reacquire(self, img_bgr: np.ndarray) -> Roi | None:
        """Ищет эталон объекта. None — не найден.

        Берёт лучший кандидат с учётом близости к последней позиции (штраф за
        дальность), иначе matchTemplate часто цепляет похожий фон в стороне.
        """
        if self._template is None or self._template_size is None:
            return None
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape[:2]
        win = self._search_window(W, H)
        if win is None:
            return None
        wx, wy, ww, wh = win
        region = gray[wy : wy + wh, wx : wx + ww]

        # Ускорение: matchTemplate на 1/2 разрешения (~4× дешевле).
        scale_down = 0.5
        rh, rw = region.shape[:2]
        small_w, small_h = max(8, int(rw * scale_down)), max(8, int(rh * scale_down))
        region_s = cv2.resize(region, (small_w, small_h), interpolation=cv2.INTER_AREA)
        inv = 1.0 / scale_down
        tw0, th0 = self._template_size

        solid = float(np.var(self._template)) < 1.0
        method = cv2.TM_SQDIFF_NORMED if solid else cv2.TM_CCOEFF_NORMED

        pcx = pcy = diag = None
        if self._roi_f is not None:
            px, py, pw, ph = self._roi_f
            pcx, pcy = px + pw / 2.0, py + ph / 2.0
            diag = float(np.hypot(max(pw, 1.0), max(ph, 1.0)))

        # (adj, raw_score, fx, fy, fw, fh)
        cands: list[tuple[float, float, int, int, int, int]] = []
        for s in self._reacquire_scales():
            tw = max(8, int(round(tw0 * s * scale_down)))
            th = max(8, int(round(th0 * s * scale_down)))
            if tw >= region_s.shape[1] or th >= region_s.shape[0]:
                continue
            templ = cv2.resize(self._template, (tw, th), interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(region_s, templ, method)
            if solid:
                minv, _, minloc, _ = cv2.minMaxLoc(res)
                score = 1.0 - float(minv)
                maxloc = minloc
            else:
                _, maxv, _, maxloc = cv2.minMaxLoc(res)
                score = float(maxv)
            if not np.isfinite(score) or score < self.reacquire_threshold:
                continue
            fx = wx + int(round(maxloc[0] * inv))
            fy = wy + int(round(maxloc[1] * inv))
            fw = max(8, int(round(tw * inv)))
            fh = max(8, int(round(th * inv)))
            cx, cy = fx + fw / 2.0, fy + fh / 2.0
            pen = 0.0
            if pcx is not None and diag is not None and diag > 0:
                dist = float(np.hypot(cx - pcx, cy - pcy))
                pen += min(0.4, 0.18 * (dist / diag))
            # Предпочитаем масштаб ближе к последнему известному.
            cur = max(self._scale, 0.05)
            pen += min(0.12, 0.1 * abs(float(np.log(s / cur))))
            cands.append((score - pen, score, fx, fy, fw, fh))

        if not cands:
            return None
        cands.sort(key=lambda t: t[0], reverse=True)
        best_adj, best_raw, fx, fy, fw, fh = cands[0]

        # Неоднозначность: два похожих пика → не угадываем.
        if len(cands) >= 2:
            second_adj = cands[1][0]
            if best_adj - second_adj < 0.04 and best_raw < self.reacquire_threshold + 0.12:
                return None

        cand = (fx, fy, fw, fh)
        if not self._aspect_ok(cand):
            return None
        if not self._appearance_ok(img_bgr, cand):
            return None

        # Дальний прыжок требует более сильного совпадения.
        if pcx is not None and diag is not None and diag > 0:
            cx, cy = fx + fw / 2.0, fy + fh / 2.0
            dist = float(np.hypot(cx - pcx, cy - pcy))
            need = self.reacquire_threshold
            if dist > 1.2 * diag:
                need = max(need + 0.1, 0.7)
            if dist > 2.0 * diag:
                need = max(need + 0.15, 0.75)
            if best_raw < need:
                return None

        full_score = self._score_roi(img_bgr, cand)
        if not np.isfinite(full_score) or full_score < self.reacquire_threshold:
            return None
        if not self._appearance_ok(img_bgr, cand):
            return None
        self.last_score = full_score
        return cand

    def _smooth_box(
        self, box: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        """Сглаживает рамку: центр по EMA, размер — равномерным масштабом.

        Размер меняется единым масштабом относительно исходной рамки, поэтому
        пропорции объекта сохраняются (нет неравномерного «схлопывания» сторон),
        а сам масштаб сглаживается и ограничивается по скорости изменения.
        """
        x, y, w, h = box
        cx, cy = x + w / 2.0, y + h / 2.0
        a = 1.0 - self.smooth if self.smooth > 0.0 else 1.0

        # --- Центр ---
        if self._roi_f is None:
            ncx, ncy = cx, cy
        else:
            px, py, pw, ph = self._roi_f
            pcx, pcy = px + pw / 2.0, py + ph / 2.0
            ncx = pcx + a * (cx - pcx)
            ncy = pcy + a * (cy - pcy)

        # --- Размер ---
        if self.lock_size and self._locked_size is not None:
            nw, nh = self._locked_size
            return (ncx - nw / 2.0, ncy - nh / 2.0, nw, nh)

        if not self.keep_aspect or self._init_size is None:
            # «Сырое» поведение: стороны сглаживаются независимо (как у трекера).
            if self._roi_f is None:
                return (ncx - w / 2.0, ncy - h / 2.0, w, h)
            _, _, pw, ph = self._roi_f
            nw = pw + a * (w - pw)
            nh = ph + a * (h - ph)
            return (ncx - nw / 2.0, ncy - nh / 2.0, nw, nh)

        # Равномерный масштаб от площади рамки трекера — сохраняет пропорции.
        w0, h0 = self._init_size
        target_scale = float(np.sqrt(max(w * h, 1.0) / max(w0 * h0, 1.0)))

        # Сглаживаем масштаб и ограничиваем резкое изменение за кадр.
        new_scale = self._scale + a * (target_scale - self._scale)
        if self.max_scale_step > 0.0:
            lo = self._scale * (1.0 - self.max_scale_step)
            hi = self._scale * (1.0 + self.max_scale_step)
            new_scale = max(lo, min(hi, new_scale))
        new_scale = max(0.1, new_scale)
        self._scale = new_scale

        nw = w0 * new_scale
        nh = h0 * new_scale
        return (ncx - nw / 2.0, ncy - nh / 2.0, nw, nh)

    def init_interactive(
        self, frame: np.ndarray, max_display: int = 1200
    ) -> Roi | None:
        """Даёт выбрать объект мышью (рамкой) и инициализирует трекер. None — отмена."""
        roi = select_object_roi(_as_tracker_input(frame), max_display)
        if roi is None:
            return None
        return self.init(frame, roi)

    def init_by_click(
        self,
        frame: np.ndarray,
        max_display: int = 1200,
        *,
        tolerance: int = 16,
        grabcut_refine: bool = True,
    ) -> Roi | None:
        """Выбор объекта кликом с авто-определением границ. None — отмена."""
        roi = select_object_by_click(
            _as_tracker_input(frame),
            max_display,
            tolerance=tolerance,
            grabcut_refine=grabcut_refine,
        )
        if roi is None:
            return None
        return self.init(frame, roi)

    def _accept_box(self, img_bgr: np.ndarray, box: tuple[float, float, float, float]) -> None:
        """Принимает новую рамку (сглаживание + обновление состояния)."""
        smoothed = self._smooth_box(
            (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        )
        self._roi_f = smoothed
        self.roi = clamp_roi(smoothed, img_bgr.shape[1], img_bgr.shape[0])
        self.ok = True
        self._fail_streak = 0
        self._lost_frames = 0
        if self.last_score is not None and np.isfinite(self.last_score):
            if self._score_ema is None:
                self._score_ema = float(self.last_score)
            else:
                self._score_ema = 0.9 * self._score_ema + 0.1 * float(self.last_score)

    def _reinit_on(self, img_bgr: np.ndarray, box: Roi) -> None:
        """Пересоздаёт OpenCV-трекер на заданной рамке.

        Размер/эталон обновляем только при уверенном match; иначе при lock_size
        двигаем только центр — чтобы слабый ложный пик не «прилипал» навсегда.
        """
        h, w = img_bgr.shape[:2]
        found = clamp_roi(
            (float(box[0]), float(box[1]), float(box[2]), float(box[3])), w, h
        )
        score = (
            float(self.last_score)
            if self.last_score is not None and np.isfinite(self.last_score)
            else 0.0
        )
        adopt_size = score >= max(self.reacquire_threshold + 0.08, 0.7)

        if self.lock_size and self._locked_size is not None and not adopt_size:
            lw, lh = self._locked_size
            cx = found[0] + found[2] / 2.0
            cy = found[1] + found[3] / 2.0
            box = clamp_roi((cx - lw / 2.0, cy - lh / 2.0, lw, lh), w, h)
        else:
            box = found

        self._tracker = create_raw_tracker(self.kind)
        self._tracker.init(img_bgr, box)
        self._roi_f = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        if self._init_size is not None and self._init_size[0] > 0:
            self._scale = float(box[2]) / self._init_size[0]
        self.roi = box
        self.ok = True
        self._fail_streak = 0
        self._lost_frames = 0
        if adopt_size:
            self._save_template(img_bgr, box)
            self._locked_size = (float(box[2]), float(box[3]))
        if self.last_score is not None and np.isfinite(self.last_score):
            self._score_ema = float(self.last_score)

    def _attempt_reacquire(self, img_bgr: np.ndarray) -> bool:
        """Пробует найти объект. True — успешно переинициализирован."""
        if not self.reacquire:
            return False
        found = self._try_reacquire(img_bgr)
        if found is None:
            return False
        if (
            self.last_score is None
            or not np.isfinite(self.last_score)
            or self.last_score < self.reacquire_threshold
        ):
            return False

        # Сильное изменение размера — только при очень уверенном match.
        if self._roi_f is not None or self._locked_size is not None:
            if self._locked_size is not None:
                rw, rh = self._locked_size
            else:
                rw, rh = self._roi_f[2], self._roi_f[3]
            old_a = max(float(rw) * float(rh), 1.0)
            new_a = max(float(found[2]) * float(found[3]), 1.0)
            ratio = new_a / old_a
            if ratio < 0.55 or ratio > 1.9:
                if self.last_score < max(0.72, self.reacquire_threshold + 0.12):
                    return False

        self._reinit_on(img_bgr, found)
        self.reacquired = True
        return True

    def update(self, frame: np.ndarray) -> tuple[bool, Roi | None]:
        """Обновляет положение объекта на новом кадре.

        Если трекер «перепрыгнул» на другой объект/фон (слабое сходство с эталоном
        или слишком большой прыжок рамки), кадр считается неудачным. После
        lost_patience подряд неудач объявляется потеря цели; затем возможен
        повторный захват по эталону (не каждый кадр — см. reacquire_interval).
        """
        if not self.initialized or self._tracker is None:
            raise RuntimeError("ObjectTracker не инициализирован: вызовите init().")
        img = _as_tracker_input(frame)
        self.reacquired = False

        # Быстрый путь при уже объявленной потере: CSRT не гоняем каждый кадр.
        if not self.ok:
            self._lost_frames += 1
            self._since_reacquire += 1
            if self._since_reacquire >= self.reacquire_interval:
                self._since_reacquire = 0
                if self._attempt_reacquire(img):
                    return self.ok, self.roi
            return self.ok, self.roi

        raw_ok, box = self._tracker.update(img)
        accepted = False
        if raw_ok:
            valid, _score = self._validate_box(
                img, (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
            )
            if valid:
                self._accept_box(
                    img, (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
                )
                accepted = True
            else:
                # Не двигаем рамку к ложной цели. Откат CSRT — только один раз
                # на серию сбоев (пересоздание трекера дорогое).
                self._fail_streak += 1
                if self.roi is not None and self._fail_streak == 1:
                    self._tracker = create_raw_tracker(self.kind)
                    self._tracker.init(img, self.roi)
        else:
            self._fail_streak += 1
            self.last_score = None

        if accepted:
            return self.ok, self.roi

        # Пока не набрали patience — рамка остаётся на последней хорошей позиции.
        if self._fail_streak < self.lost_patience and self.roi is not None:
            return True, self.roi

        # Объявляем потерю.
        self.ok = False
        self._since_reacquire = 0
        self._lost_frames = max(self._lost_frames, 1)
        # Один поиск сразу при потере; дальше — по interval.
        self._attempt_reacquire(img)
        return self.ok, self.roi

    def reset(self) -> None:
        """Сбрасывает состояние (для повторного выбора объекта)."""
        self._tracker = None
        self.roi = None
        self._roi_f = None
        self._locked_size = None
        self._init_size = None
        self._scale = 1.0
        self._template = None
        self._template_size = None
        self._fail_streak = 0
        self._since_reacquire = 0
        self._lost_frames = 0
        self._score_ema = None
        self.ok = False
        self.initialized = False
        self.reacquired = False
        self.last_score = None


def draw_tracking(
    frame_bgr: np.ndarray,
    roi: Roi | None,
    ok: bool,
    frame_idx: int = 0,
) -> np.ndarray:
    """Рисует рамку ROI и статус трекинга (для отладки/тестов)."""
    out = frame_bgr.copy() if frame_bgr.ndim == 3 else cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)
    color = (0, 220, 0) if ok else (0, 0, 255)
    if roi is not None:
        x, y, rw, rh = roi
        cv2.rectangle(out, (x, y), (x + rw, y + rh), color, 2)
        cx, cy = roi_center(roi)
        cv2.drawMarker(out, (cx, cy), color, cv2.MARKER_CROSS, 14, 2)
    if roi is None:
        state = "SELECT (R)"
    else:
        state = "OK" if ok else "LOST"
    status = f"frame {frame_idx}  {state}"
    cv2.putText(out, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Изолированный тест захвата и трекинга объекта по одному видео.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video", required=True, help="Видео для теста трекинга.")
    p.add_argument(
        "--tracker", choices=list(TRACKER_KINDS), default="csrt", help="Тип трекера."
    )
    p.add_argument(
        "--roi",
        type=int,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        default=None,
        help="ROI без интерактивного выбора (иначе выбор мышью).",
    )
    p.add_argument(
        "--smooth",
        type=float,
        default=0.6,
        help="Сглаживание рамки [0..1): 0 = без сглаживания, ближе к 1 = плавнее.",
    )
    p.add_argument(
        "--lock-size",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Фиксировать размер рамки (двигается только центр). Резкое сжатие → LOST.",
    )
    p.add_argument(
        "--keep-aspect",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Масштабировать рамку равномерно, сохраняя исходные пропорции.",
    )
    p.add_argument(
        "--max-scale-step",
        type=float,
        default=0.05,
        help="Макс. относительное изменение масштаба рамки за кадр (0 = без лимита).",
    )
    p.add_argument(
        "--max-size-ratio",
        type=float,
        default=1.6,
        help="Макс. изменение площади рамки за кадр (1.6 ≈ ±60%%); иначе LOST.",
    )
    p.add_argument(
        "--verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Отклонять дрейф рамки на другой объект/фон (сходство с эталоном + прыжок).",
    )
    p.add_argument(
        "--verify-threshold",
        type=float,
        default=0.45,
        help="Мин. сходство с эталоном [0..1], ниже — считаем кадр плохим.",
    )
    p.add_argument(
        "--max-jump",
        type=float,
        default=0.7,
        help="Макс. прыжок центра за кадр в долях диагонали ROI (0 = без лимита).",
    )
    p.add_argument(
        "--verify-rel",
        type=float,
        default=0.75,
        help="Отклонять кадр, если score < EMA*verify-rel (дрейф на соседний объект).",
    )
    p.add_argument(
        "--min-iou",
        type=float,
        default=0.5,
        help="Мин. IoU с предыдущей рамкой (отсекает перескок на пересекающийся объект).",
    )
    p.add_argument(
        "--lost-patience",
        type=int,
        default=2,
        help="Сколько подряд плохих кадров нужно, чтобы объявить потерю цели.",
    )
    p.add_argument(
        "--reacquire",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Повторно захватывать объект после потери по эталону.",
    )
    p.add_argument(
        "--reacquire-threshold",
        type=float,
        default=0.62,
        help="Порог совпадения для повторного захвата [0..1].",
    )
    p.add_argument(
        "--reacquire-radius",
        type=float,
        default=2.5,
        help="Окно поиска при перезахвате (доли max(w,h) ROI вокруг последней позиции).",
    )
    p.add_argument(
        "--reacquire-global",
        action="store_true",
        help="Искать объект по всему кадру при перезахвате (может хватать похожий фон).",
    )
    p.add_argument(
        "--reacquire-interval",
        type=int,
        default=5,
        help="Искать объект при потере каждые N кадров (1 = каждый кадр, тяжелее).",
    )
    p.add_argument(
        "--reacquire-scale-min",
        type=float,
        default=0.35,
        help="Мин. масштаб эталона при перезахвате (меньше = искать более далёкий/мелкий объект).",
    )
    p.add_argument(
        "--reacquire-scale-max",
        type=float,
        default=1.4,
        help="Макс. масштаб эталона при перезахвате.",
    )
    p.add_argument(
        "--click-tolerance",
        type=int,
        default=16,
        help="Допуск цвета при авто-выделении кликом (клавиша C).",
    )
    p.add_argument(
        "--no-grabcut",
        action="store_true",
        help="Не уточнять границы объекта GrabCut'ом при выборе кликом.",
    )
    p.add_argument("--max-display", type=int, default=1200, help="Макс. сторона окна.")
    p.add_argument(
        "--max-frames", type=int, default=0, help="Ограничить число кадров (0 = все)."
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="Без окна (для нагрузочных/CI тестов): печатает ROI по кадрам.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Ошибка: не удалось открыть видео '{args.video}'.")

    ok, frame = cap.read()
    if not ok:
        sys.exit("Ошибка: не удалось прочитать первый кадр.")

    tracker = ObjectTracker(
        kind=args.tracker,
        smooth=args.smooth,
        lock_size=args.lock_size,
        keep_aspect=args.keep_aspect,
        max_scale_step=args.max_scale_step,
        max_size_ratio=args.max_size_ratio,
        verify=args.verify,
        verify_threshold=args.verify_threshold,
        max_jump=args.max_jump,
        lost_patience=args.lost_patience,
        verify_rel=args.verify_rel,
        min_iou=args.min_iou,
        reacquire=args.reacquire,
        reacquire_threshold=args.reacquire_threshold,
        reacquire_radius=args.reacquire_radius,
        reacquire_global=args.reacquire_global,
        reacquire_interval=args.reacquire_interval,
        reacquire_scale_min=args.reacquire_scale_min,
        reacquire_scale_max=args.reacquire_scale_max,
    )
    if args.roi is not None:
        roi = tracker.init(frame, tuple(args.roi))
        print(f"Старт трекинга ({args.tracker}), ROI={roi}")
    elif args.no_show:
        sys.exit("В режиме --no-show укажите --roi (интерактивный выбор недоступен).")
    else:
        print("R — выбор рамкой, C — выбор кликом (авто-границы), Q — выход.")

    window = "Object tracking test (R=box, C=click, Q=quit)"
    if not args.no_show:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    frame_idx = 0
    lost_count = 0
    while True:
        if frame_idx > 0:
            ok, frame = cap.read()
            if not ok:
                print("Конец видео.")
                break
            if args.max_frames > 0 and frame_idx >= args.max_frames:
                print("Достигнут --max-frames.")
                break
            if tracker.initialized:
                track_ok, roi = tracker.update(frame)
                if not track_ok:
                    lost_count += 1

        if args.no_show:
            print(f"frame {frame_idx}: ok={tracker.ok} roi={tracker.roi}")
        else:
            vis = draw_tracking(frame, tracker.roi, tracker.ok, frame_idx)
            scale = display_scale(vis.shape, args.max_display)
            cv2.imshow(window, fit_for_display(vis, scale))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("r"), ord("R")):
                print("Выбор объекта рамкой на текущем кадре...")
                tracker.init_interactive(frame, args.max_display)
            if key in (ord("c"), ord("C")):
                print("Выбор объекта кликом на текущем кадре...")
                tracker.init_by_click(
                    frame,
                    args.max_display,
                    tolerance=args.click_tolerance,
                    grabcut_refine=not args.no_grabcut,
                )
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"Готово: обработано {frame_idx} кадров, потерь трекинга: {lost_count}.")


if __name__ == "__main__":
    main()
