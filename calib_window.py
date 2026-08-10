#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Режим калибровки для video_record: захват пар доски и запуск калибровки."""
from __future__ import annotations

import socket
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QCloseEvent, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from calibrate_stereo import (
    draw_board_corners,
    find_board_corners,
    run_mono_calibration_from_folder,
    run_stereo_calibration_from_folders,
    to_gray,
)
from depth_map import split_sbs
from video_record import (
    FRAME_HEIGHT,
    FRAME_WIDTH,
    STREAM_MODE_SPECS,
    Bridge,
    CameraInfo,
    CameraManager,
    LatestFrameStore,
    StreamMode,
    VideoLabel,
    _enable_windows_capture_performance,
    _restore_windows_execution_state,
)


class VideoPairSource:
    """Покадровый источник: два MP4 (L/R) или один SBS."""

    def __init__(self) -> None:
        self._cap_l: cv2.VideoCapture | None = None
        self._cap_r: cv2.VideoCapture | None = None
        self._cap_sbs: cv2.VideoCapture | None = None
        self._swap_lr = False
        self._mode: str = "none"  # none | pair | sbs
        self._frame_l: np.ndarray | None = None
        self._frame_r: np.ndarray | None = None
        self._index = 0
        self._count = 0

    @property
    def count(self) -> int:
        return self._count

    @property
    def index(self) -> int:
        return self._index

    @property
    def open(self) -> bool:
        return self._mode != "none" and self._count > 0

    def close(self) -> None:
        for cap in (self._cap_l, self._cap_r, self._cap_sbs):
            if cap is not None:
                cap.release()
        self._cap_l = self._cap_r = self._cap_sbs = None
        self._mode = "none"
        self._frame_l = self._frame_r = None
        self._index = 0
        self._count = 0

    def open_pair(self, left_path: str, right_path: str) -> None:
        self.close()
        cap_l = cv2.VideoCapture(left_path)
        cap_r = cv2.VideoCapture(right_path)
        if not cap_l.isOpened() or not cap_r.isOpened():
            cap_l.release()
            cap_r.release()
            raise ValueError("Не удалось открыть left/right видео.")
        n_l = int(cap_l.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        n_r = int(cap_r.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self._count = max(0, min(n_l, n_r))
        if self._count <= 0:
            cap_l.release()
            cap_r.release()
            raise ValueError("В видео нет кадров.")
        self._cap_l, self._cap_r = cap_l, cap_r
        self._mode = "pair"
        self.seek(0)

    def open_sbs(self, path: str, *, swap_lr: bool = False) -> None:
        self.close()
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError("Не удалось открыть SBS-видео.")
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if n <= 0:
            cap.release()
            raise ValueError("В видео нет кадров.")
        self._cap_sbs = cap
        self._swap_lr = bool(swap_lr)
        self._mode = "sbs"
        self._count = n
        self.seek(0)

    def seek(self, index: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        if not self.open:
            return None, None
        index = int(np.clip(index, 0, self._count - 1))
        self._index = index
        if self._mode == "pair":
            assert self._cap_l is not None and self._cap_r is not None
            self._cap_l.set(cv2.CAP_PROP_POS_FRAMES, index)
            self._cap_r.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok_l, fr_l = self._cap_l.read()
            ok_r, fr_r = self._cap_r.read()
            self._frame_l = fr_l if ok_l else None
            self._frame_r = fr_r if ok_r else None
        else:
            assert self._cap_sbs is not None
            self._cap_sbs.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, fr = self._cap_sbs.read()
            if not ok or fr is None:
                self._frame_l = self._frame_r = None
            else:
                left, right = split_sbs(fr, swap_lr=self._swap_lr)
                self._frame_l, self._frame_r = left, right
        return self._frame_l, self._frame_r

    def step(self, delta: int = 1) -> tuple[np.ndarray | None, np.ndarray | None]:
        return self.seek(self._index + delta)

    def current(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        return self._frame_l, self._frame_r


class CornerAcceptDialog(QDialog):
    """Показ кадров с углами: Принять / Браковать."""

    def __init__(
        self,
        parent: QWidget | None,
        *,
        vis_l: np.ndarray | None,
        vis_r: np.ndarray | None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Подтверждение кадра калибровки")
        self.setModal(True)
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        if vis_l is not None:
            lab = VideoLabel("Left")
            lab.set_frame(vis_l)
            lab.setMinimumSize(420, 336)
            row.addWidget(lab, 1)
        if vis_r is not None:
            lab = VideoLabel("Right")
            lab.set_frame(vis_r)
            lab.setMinimumSize(420, 336)
            row.addWidget(lab, 1)
        layout.addLayout(row)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Принять (Enter)")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Браковать (Esc)")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        QShortcut(QKeySequence(Qt.Key.Key_Return), self, activated=self.accept)
        QShortcut(QKeySequence(Qt.Key.Key_Enter), self, activated=self.accept)
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, activated=self.reject)
        self.resize(960, 520)


class CalibWorker(QThread):
    finished_ok = Signal(str, list)
    finished_err = Signal(str)

    def __init__(self, kind: str, kwargs: dict) -> None:
        super().__init__()
        self.kind = kind
        self.kwargs = kwargs

    def run(self) -> None:  # type: ignore[override]
        try:
            if self.kind == "stereo":
                path, log = run_stereo_calibration_from_folders(**self.kwargs)
            else:
                path, log = run_mono_calibration_from_folder(**self.kwargs)
            self.finished_ok.emit(path, log)
        except Exception as exc:  # noqa: BLE001 — показать в UI
            self.finished_err.emit(str(exc))


class CalibMainWindow(QMainWindow):
    """Окно режима калибровки: live/video захват + расчёт .npz."""

    COL_USE, COL_ROLE, COL_IP, COL_REPORTED_IP, COL_PORT, COL_TEMP, COL_LAST, COL_STATE = (
        range(8)
    )

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("П139Н-1 — калибровка стереопары")
        self.resize(1280, 920)

        self.bridge = Bridge()
        self.bridge.camera_seen.connect(self._on_camera_seen)
        self.bridge.camera_status.connect(self._on_camera_status)
        self.bridge.error.connect(self._show_error)

        self.manager: Optional[CameraManager] = None
        self.latest_store = LatestFrameStore()
        self.camera_rows: dict[str, int] = {}
        self.camera_info: dict[str, CameraInfo] = {}
        self.active_slots: dict[str, int] = {}
        self.role_order: list[str] = []
        self.latest_frame_time: dict[str, float] = {}
        self.displayed_sequence: dict[str, int] = {}

        self.video_src = VideoPairSource()
        self._live_preview_l: np.ndarray | None = None
        self._live_preview_r: np.ndarray | None = None
        self._session_dir: Path | None = None
        self._accepted_count = 0
        self._calib_worker: CalibWorker | None = None

        self._offline_timer = QTimer(self)
        self._offline_timer.timeout.connect(self._mark_offline_cameras)
        self._display_timer = QTimer(self)
        self._display_timer.timeout.connect(self._on_display_tick)
        self._video_play_timer = QTimer(self)
        self._video_play_timer.timeout.connect(self._video_play_step)

        self._build_ui()
        self._offline_timer.start(1000)
        self._display_timer.start(40)

        QShortcut(QKeySequence(Qt.Key.Key_Space), self, activated=self._capture_current)
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, activated=lambda: self._step_video(-1))
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, activated=lambda: self._step_video(1))
        QShortcut(QKeySequence(Qt.Key.Key_Comma), self, activated=lambda: self._step_video(-1))
        QShortcut(QKeySequence(Qt.Key.Key_Period), self, activated=lambda: self._step_video(1))
        self._start_discovery()

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setSizeConstraint(QVBoxLayout.SizeConstraint.SetMinimumSize)
        root.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        # --- Сеть ---
        network_box = QGroupBox("Сеть")
        net = QGridLayout(network_box)
        net.addWidget(QLabel("Ethernet-IP ПК:"), 0, 0)
        self.bind_ip = QComboBox()
        self.bind_ip.setEditable(True)
        self.bind_ip.addItems(self._local_ipv4_addresses())
        self.bind_ip.setCurrentText("0.0.0.0")
        net.addWidget(self.bind_ip, 0, 1)
        self.discovery_btn = QPushButton("Перезапустить обнаружение")
        self.discovery_btn.clicked.connect(self._restart_discovery)
        net.addWidget(self.discovery_btn, 0, 2)
        net.addWidget(QLabel("Тип видео (0x0101):"), 1, 0)
        self.stream_mode_box = QComboBox()
        for mode in (
            StreamMode.PROCESSED_MONO16,
            StreamMode.RAW_MONO16,
            StreamMode.OVERLAY_RGB888,
        ):
            _, _, _, label = STREAM_MODE_SPECS[mode]
            self.stream_mode_box.addItem(f"0b{int(mode):02b} — {label}", int(mode))
        net.addWidget(self.stream_mode_box, 1, 1, 1, 2)
        layout.addWidget(network_box)

        # --- Камеры ---
        cameras_box = QGroupBox("Обнаруженные тепловизоры")
        cameras_layout = QVBoxLayout(cameras_box)
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            [
                "Исп.",
                "Роль",
                "IP источника",
                "IP в телеметрии",
                "Видео UDP",
                "Темп. код",
                "Последний пакет",
                "Состояние",
            ]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.itemChanged.connect(self._on_table_item_changed)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self.COL_STATE, QHeaderView.ResizeMode.Stretch)
        cameras_layout.addWidget(self.table)
        cam_btns = QHBoxLayout()
        self.start_btn = QPushButton("Запустить выбранные")
        self.start_btn.clicked.connect(self._start_selected)
        self.stop_btn = QPushButton("Остановить потоки")
        self.stop_btn.clicked.connect(self._stop_streams)
        cam_btns.addWidget(self.start_btn)
        cam_btns.addWidget(self.stop_btn)
        cam_btns.addStretch(1)
        cameras_layout.addLayout(cam_btns)
        layout.addWidget(cameras_box)

        # --- Превью ---
        previews = QHBoxLayout()
        left_group = QGroupBox("Левый / единственный")
        left_layout = QVBoxLayout(left_group)
        self.left_video = VideoLabel("Нет видеопотока")
        self.left_stats = QLabel("—")
        left_layout.addWidget(self.left_video, 1)
        left_layout.addWidget(self.left_stats)
        previews.addWidget(left_group, 1)
        swap_col = QVBoxLayout()
        swap_col.addStretch(1)
        self.swap_roles_btn = QPushButton("⇄\nL / R")
        self.swap_roles_btn.setFixedSize(72, 72)
        self.swap_roles_btn.setEnabled(False)
        self.swap_roles_btn.clicked.connect(self._swap_roles)
        swap_col.addWidget(self.swap_roles_btn)
        swap_col.addStretch(1)
        previews.addLayout(swap_col)
        right_group = QGroupBox("Правый")
        right_layout = QVBoxLayout(right_group)
        self.right_video = VideoLabel("Нет второго видеопотока")
        self.right_stats = QLabel("—")
        right_layout.addWidget(self.right_video, 1)
        right_layout.addWidget(self.right_stats)
        previews.addWidget(right_group, 1)
        layout.addLayout(previews, 1)

        # --- Источник ---
        src_box = QGroupBox("Источник кадров")
        src = QGridLayout(src_box)
        self.src_live = QRadioButton("Live (камеры)")
        self.src_video = QRadioButton("Видеофайлы")
        self.src_live.setChecked(True)
        self.src_group = QButtonGroup(self)
        self.src_group.addButton(self.src_live)
        self.src_group.addButton(self.src_video)
        self.src_live.toggled.connect(self._on_source_changed)
        src.addWidget(self.src_live, 0, 0)
        src.addWidget(self.src_video, 0, 1)

        src.addWidget(QLabel("Left MP4:"), 1, 0)
        self.video_left_edit = QLineEdit()
        src.addWidget(self.video_left_edit, 1, 1)
        btn_vl = QPushButton("…")
        btn_vl.clicked.connect(lambda: self._browse_video(self.video_left_edit))
        src.addWidget(btn_vl, 1, 2)

        src.addWidget(QLabel("Right MP4:"), 2, 0)
        self.video_right_edit = QLineEdit()
        src.addWidget(self.video_right_edit, 2, 1)
        btn_vr = QPushButton("…")
        btn_vr.clicked.connect(lambda: self._browse_video(self.video_right_edit))
        src.addWidget(btn_vr, 2, 2)

        src.addWidget(QLabel("или SBS MP4:"), 3, 0)
        self.video_sbs_edit = QLineEdit()
        src.addWidget(self.video_sbs_edit, 3, 1)
        btn_vs = QPushButton("…")
        btn_vs.clicked.connect(lambda: self._browse_video(self.video_sbs_edit))
        src.addWidget(btn_vs, 3, 2)
        self.sbs_swap_cb = QCheckBox("Swap L/R (SBS)")
        src.addWidget(self.sbs_swap_cb, 3, 3)

        self.video_open_btn = QPushButton("Открыть видео")
        self.video_open_btn.clicked.connect(self._open_video_source)
        src.addWidget(self.video_open_btn, 4, 0)
        self.video_play_btn = QPushButton("Play")
        self.video_play_btn.clicked.connect(self._toggle_video_play)
        src.addWidget(self.video_play_btn, 4, 1)
        self.video_slider = QSlider(Qt.Orientation.Horizontal)
        self.video_slider.setEnabled(False)
        self.video_slider.valueChanged.connect(self._on_video_slider)
        src.addWidget(self.video_slider, 4, 2, 1, 2)

        nav = QHBoxLayout()
        self.video_prev_btn = QPushButton("◀ кадр")
        self.video_prev_btn.setToolTip("Предыдущий кадр (← / ,)")
        self.video_prev_btn.clicked.connect(lambda: self._step_video(-1))
        nav.addWidget(self.video_prev_btn)
        self.video_next_btn = QPushButton("кадр ▶")
        self.video_next_btn.setToolTip("Следующий кадр (→ / .)")
        self.video_next_btn.clicked.connect(lambda: self._step_video(1))
        nav.addWidget(self.video_next_btn)
        nav.addWidget(QLabel("Скорость:"))
        self.video_speed = QDoubleSpinBox()
        self.video_speed.setRange(0.05, 2.0)
        self.video_speed.setSingleStep(0.05)
        self.video_speed.setDecimals(2)
        self.video_speed.setValue(0.5)
        self.video_speed.setSuffix("×")
        self.video_speed.setToolTip("Множитель скорости воспроизведения (1.0 = ~30 FPS).")
        self.video_speed.valueChanged.connect(self._on_video_speed_changed)
        nav.addWidget(self.video_speed)
        nav.addStretch(1)
        src.addLayout(nav, 5, 0, 1, 4)

        self.video_frame_label = QLabel("кадр —")
        src.addWidget(self.video_frame_label, 6, 0, 1, 4)
        layout.addWidget(src_box)

        # --- Захват ---
        cap_box = QGroupBox("Захват доски")
        cap = QGridLayout(cap_box)
        self.mode_stereo = QRadioButton("Stereo")
        self.mode_mono_l = QRadioButton("Mono Left")
        self.mode_mono_r = QRadioButton("Mono Right")
        self.mode_stereo.setChecked(True)
        self.mode_group = QButtonGroup(self)
        for b in (self.mode_stereo, self.mode_mono_l, self.mode_mono_r):
            self.mode_group.addButton(b)
        mode_row = QHBoxLayout()
        mode_row.addWidget(self.mode_stereo)
        mode_row.addWidget(self.mode_mono_l)
        mode_row.addWidget(self.mode_mono_r)
        mode_row.addStretch(1)
        cap.addLayout(mode_row, 0, 0, 1, 4)

        cap.addWidget(QLabel("cols"), 1, 0)
        self.board_cols = QSpinBox()
        self.board_cols.setRange(3, 30)
        self.board_cols.setValue(8)
        cap.addWidget(self.board_cols, 1, 1)
        cap.addWidget(QLabel("rows"), 1, 2)
        self.board_rows = QSpinBox()
        self.board_rows.setRange(3, 30)
        self.board_rows.setValue(5)
        cap.addWidget(self.board_rows, 1, 3)
        cap.addWidget(QLabel("square mm"), 2, 0)
        self.board_square = QDoubleSpinBox()
        self.board_square.setRange(1.0, 500.0)
        self.board_square.setDecimals(2)
        self.board_square.setValue(90.0)
        cap.addWidget(self.board_square, 2, 1)

        self.capture_btn = QPushButton("Захватить (Space)")
        self.capture_btn.clicked.connect(self._capture_current)
        cap.addWidget(self.capture_btn, 2, 2, 1, 2)
        self.new_session_btn = QPushButton("Новая сессия")
        self.new_session_btn.clicked.connect(self._new_session)
        cap.addWidget(self.new_session_btn, 3, 0, 1, 2)
        self.session_label = QLabel("Сессия: нет (создастся при первом принятии)")
        cap.addWidget(self.session_label, 3, 2, 1, 2)
        self.accepted_label = QLabel("Принято: 0")
        cap.addWidget(self.accepted_label, 4, 0, 1, 4)
        layout.addWidget(cap_box)

        # --- Запуск калибровки ---
        run_box = QGroupBox("Запуск калибровки из папок")
        run = QGridLayout(run_box)
        run.addWidget(QLabel("Left folder:"), 0, 0)
        self.folder_left = QLineEdit()
        run.addWidget(self.folder_left, 0, 1)
        bl = QPushButton("…")
        bl.clicked.connect(lambda: self._browse_folder(self.folder_left))
        run.addWidget(bl, 0, 2)
        run.addWidget(QLabel("Right folder:"), 1, 0)
        self.folder_right = QLineEdit()
        run.addWidget(self.folder_right, 1, 1)
        br = QPushButton("…")
        br.clicked.connect(lambda: self._browse_folder(self.folder_right))
        run.addWidget(br, 1, 2)
        use_session = QPushButton("Подставить текущую сессию")
        use_session.clicked.connect(self._use_session_folders)
        run.addWidget(use_session, 0, 3, 2, 1)

        run.addWidget(QLabel("Output .npz:"), 2, 0)
        self.output_npz = QLineEdit("stereo_calib.npz")
        run.addWidget(self.output_npz, 2, 1)
        bo = QPushButton("…")
        bo.clicked.connect(self._browse_output_npz)
        run.addWidget(bo, 2, 2)

        run.addWidget(QLabel("Calib left .npz:"), 3, 0)
        self.calib_left_npz = QLineEdit()
        self.calib_left_npz.setPlaceholderText(
            "опционально: готовая калибровка левой (нужен и right)"
        )
        self.calib_left_npz.setToolTip(
            "Как --calib-left в calibrate_stereo: пропустить mono calibrateCamera "
            "и взять intrinsics из файла. Нужно указать оба файла."
        )
        run.addWidget(self.calib_left_npz, 3, 1)
        bcl = QPushButton("…")
        bcl.clicked.connect(lambda: self._browse_npz_file(self.calib_left_npz))
        run.addWidget(bcl, 3, 2)

        run.addWidget(QLabel("Calib right .npz:"), 4, 0)
        self.calib_right_npz = QLineEdit()
        self.calib_right_npz.setPlaceholderText(
            "опционально: готовая калибровка правой (нужен и left)"
        )
        self.calib_right_npz.setToolTip(
            "Как --calib-right в calibrate_stereo. Вместе с left — только stereo + rectify."
        )
        run.addWidget(self.calib_right_npz, 4, 1)
        bcr = QPushButton("…")
        bcr.clicked.connect(lambda: self._browse_npz_file(self.calib_right_npz))
        run.addWidget(bcr, 4, 2)

        run.addWidget(QLabel("alpha"), 5, 0)
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.0, 1.0)
        self.alpha_spin.setSingleStep(0.05)
        self.alpha_spin.setValue(1.0)
        run.addWidget(self.alpha_spin, 5, 1)
        run.addWidget(QLabel("rectify"), 5, 2)
        self.rectify_box = QComboBox()
        self.rectify_box.addItems(["calibrated", "uncalibrated"])
        run.addWidget(self.rectify_box, 5, 3)

        self.zero_dist_cb = QCheckBox("zero-distortion")
        self.fix_k1_cb = QCheckBox("fix-k1")
        self.fix_k2_cb = QCheckBox("fix-k2")
        self.fix_k3_cb = QCheckBox("fix-k3")
        self.fix_tang_cb = QCheckBox("fix-tangential")
        flags_row = QHBoxLayout()
        for w in (
            self.zero_dist_cb,
            self.fix_k1_cb,
            self.fix_k2_cb,
            self.fix_k3_cb,
            self.fix_tang_cb,
        ):
            flags_row.addWidget(w)
        flags_row.addStretch(1)
        run.addLayout(flags_row, 6, 0, 1, 4)

        self.run_calib_btn = QPushButton("Калибровать")
        self.run_calib_btn.clicked.connect(self._run_calibration)
        run.addWidget(self.run_calib_btn, 7, 0, 1, 4)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMinimumHeight(140)
        run.addWidget(self.log_view, 8, 0, 1, 4)
        layout.addWidget(run_box)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(root)
        self.setCentralWidget(scroll)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Режим калибровки. Space — захват кадра с углами.")
        self._on_source_changed(True)

    # ---- network / cameras (same protocol as MainWindow) ----

    @staticmethod
    def _local_ipv4_addresses() -> list[str]:
        addresses = ["0.0.0.0"]
        try:
            host = socket.gethostname()
            for item in socket.getaddrinfo(host, None, socket.AF_INET):
                ip = item[4][0]
                if ip not in addresses:
                    addresses.append(ip)
        except OSError:
            pass
        return addresses

    def _start_discovery(self) -> None:
        bind_ip = self.bind_ip.currentText().strip() or "0.0.0.0"
        self.manager = CameraManager(
            bind_ip=bind_ip,
            on_camera=self.bridge.camera_seen.emit,
            on_frame=self.latest_store.put,
            on_status=self.bridge.camera_status.emit,
            on_error=self.bridge.error.emit,
        )
        self.manager.start_discovery()
        self.statusBar().showMessage(
            f"Телеметрия: UDP 0.0.0.0:53000; команды через {bind_ip}"
        )

    def _restart_discovery(self) -> None:
        self._stop_streams()
        if self.manager:
            self.manager.stop_all()
        self.manager = None
        self._start_discovery()

    def _on_camera_seen(self, info: CameraInfo) -> None:
        self.camera_info[info.camera_ip] = info
        row = self.camera_rows.get(info.camera_ip)
        if row is None:
            self.table.blockSignals(True)
            row = self.table.rowCount()
            self.table.insertRow(row)
            use_item = QTableWidgetItem()
            use_item.setFlags(use_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            use_item.setCheckState(Qt.CheckState.Unchecked)
            self.table.setItem(row, self.COL_USE, use_item)
            self.table.setItem(row, self.COL_ROLE, QTableWidgetItem("—"))
            self.table.setItem(row, self.COL_IP, QTableWidgetItem(info.camera_ip))
            self.table.setItem(
                row, self.COL_REPORTED_IP, QTableWidgetItem(info.reported_ip)
            )
            self.table.setItem(
                row, self.COL_PORT, QTableWidgetItem(str(info.video_port))
            )
            self.table.setItem(
                row, self.COL_TEMP, QTableWidgetItem(f"0x{info.temperature_code:04X}")
            )
            self.table.setItem(row, self.COL_LAST, QTableWidgetItem("сейчас"))
            self.table.setItem(row, self.COL_STATE, QTableWidgetItem("Обнаружен"))
            self.camera_rows[info.camera_ip] = row
            self.table.blockSignals(False)
        else:
            self.table.item(row, self.COL_REPORTED_IP).setText(info.reported_ip)
            self.table.item(row, self.COL_PORT).setText(str(info.video_port))
            self.table.item(row, self.COL_TEMP).setText(
                f"0x{info.temperature_code:04X}"
            )
            self.table.item(row, self.COL_LAST).setText("сейчас")
            if info.camera_ip not in self.active_slots:
                self.table.item(row, self.COL_STATE).setText("Обнаружен")

    def _on_camera_status(self, camera_ip: str, status: str) -> None:
        row = self.camera_rows.get(camera_ip)
        if row is not None:
            self.table.item(row, self.COL_STATE).setText(status)

    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, "Ошибка", message)

    def _selected_camera_ips(self) -> list[str]:
        selected: list[str] = []
        for ip, row in sorted(self.camera_rows.items(), key=lambda item: item[1]):
            item = self.table.item(row, self.COL_USE)
            if item and item.checkState() == Qt.CheckState.Checked:
                selected.append(ip)
        return selected

    def _on_table_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() != self.COL_USE:
            return
        selected = self._selected_camera_ips()
        if len(selected) > 2:
            self.table.blockSignals(True)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.table.blockSignals(False)
            QMessageBox.warning(
                self, "Ограничение", "Можно выбрать не более двух тепловизоров."
            )
            selected = self._selected_camera_ips()
        self._update_roles(selected)

    def _update_roles(self, selected: list[str]) -> None:
        self.role_order = [ip for ip in self.role_order if ip in selected]
        self.role_order.extend(ip for ip in selected if ip not in self.role_order)
        self._apply_role_labels()
        self.swap_roles_btn.setEnabled(len(self.role_order) == 2 or len(self.active_slots) == 2)

    def _apply_role_labels(self) -> None:
        for ip, row in self.camera_rows.items():
            role = "—"
            if ip in self.role_order:
                index = self.role_order.index(ip)
                role = "Левый / единственный" if index == 0 else "Правый"
            self.table.item(row, self.COL_ROLE).setText(role)

    def _swap_roles(self) -> None:
        if self.active_slots:
            ordered = [
                ip
                for ip, _slot in sorted(
                    self.active_slots.items(), key=lambda item: item[1]
                )
            ]
        else:
            ordered = list(self.role_order)
        if len(ordered) != 2:
            return
        ordered.reverse()
        self.role_order = ordered
        self._apply_role_labels()
        if len(self.active_slots) == 2:
            self.active_slots = {ip: slot for slot, ip in enumerate(ordered)}
            for ip in ordered:
                self.displayed_sequence[ip] = 0
            self.left_video.clear_frame("Ожидание левого видеопотока")
            self.right_video.clear_frame("Ожидание правого видеопотока")
            self.left_stats.setText("—")
            self.right_stats.setText("—")

    def _start_selected(self) -> None:
        selected = self._selected_camera_ips()
        if not selected:
            QMessageBox.warning(
                self, "Нет выбора", "Отметьте один или два тепловизора в таблице."
            )
            return
        if self.manager is None:
            self._start_discovery()
        assert self.manager is not None
        self._update_roles(selected)
        ordered = [ip for ip in self.role_order if ip in selected]
        mode_val = int(self.stream_mode_box.currentData())
        self.manager.set_stream_mode(StreamMode(mode_val))
        self._stop_streams()
        notes = _enable_windows_capture_performance()
        self.latest_store.clear()
        self.active_slots = {ip: index for index, ip in enumerate(ordered)}
        self.displayed_sequence = {ip: 0 for ip in ordered}
        for ip in ordered:
            try:
                self.manager.start_camera(ip)
            except (KeyError, OSError) as exc:
                self._show_error(f"Не удалось запустить {ip}: {exc}")
        self.swap_roles_btn.setEnabled(len(ordered) == 2)
        _, _, _, label = STREAM_MODE_SPECS[StreamMode(mode_val)]
        self.statusBar().showMessage(
            f"Запущено потоков: {len(ordered)} | {label}"
            + (f" | {'; '.join(notes)}" if notes else "")
        )
        if not self.src_live.isChecked():
            self.src_live.setChecked(True)

    def _stop_streams(self) -> None:
        if self.manager:
            for ip in list(self.active_slots):
                self.manager.stop_camera(ip)
        self.active_slots.clear()
        self.displayed_sequence.clear()
        _restore_windows_execution_state()
        self.latest_store.clear()
        if self.src_live.isChecked():
            self.left_video.clear_frame("Нет видеопотока")
            self.right_video.clear_frame("Нет второго видеопотока")
            self.left_stats.setText("—")
            self.right_stats.setText("—")
        self.swap_roles_btn.setEnabled(len(self.role_order) == 2)

    def _mark_offline_cameras(self) -> None:
        now = time.monotonic()
        for ip, row in self.camera_rows.items():
            last = self.latest_frame_time.get(ip)
            if last is None:
                continue
            if now - last > 3.0 and ip not in self.active_slots:
                self.table.item(row, self.COL_LAST).setText("нет пакетов")

    def _on_display_tick(self) -> None:
        if self.src_live.isChecked():
            self._refresh_live_frames()

    def _refresh_live_frames(self) -> None:
        for camera_ip, slot in list(self.active_slots.items()):
            item = self.latest_store.get(camera_ip)
            if item is None or item.sequence == self.displayed_sequence.get(camera_ip, 0):
                continue
            self.displayed_sequence[camera_ip] = item.sequence
            self.latest_frame_time[camera_ip] = item.received_at
            frame = item.frame
            meta = item.meta
            if frame.shape[:2] != (FRAME_HEIGHT, FRAME_WIDTH) or frame.ndim not in (2, 3):
                self._on_camera_status(camera_ip, "Получен кадр неверного размера")
                continue
            elapsed = max(item.received_at - item.first_received_at, 1e-3)
            fps = item.total_frames / elapsed
            stats = (
                f"{camera_ip} | кадр {meta.frame_number} | "
                f"{fps:.1f} кадр/с | неполных {meta.incomplete_frames}"
            )
            if slot == 0:
                self._live_preview_l = frame
                self.left_video.set_frame(frame)
                self.left_stats.setText(stats)
            else:
                self._live_preview_r = frame
                self.right_video.set_frame(frame)
                self.right_stats.setText(stats)

    # ---- video source ----

    def _on_source_changed(self, _checked: bool = False) -> None:
        video = self.src_video.isChecked()
        for w in (
            self.video_left_edit,
            self.video_right_edit,
            self.video_sbs_edit,
            self.video_open_btn,
            self.video_play_btn,
            self.video_slider,
            self.sbs_swap_cb,
            self.video_prev_btn,
            self.video_next_btn,
            self.video_speed,
        ):
            w.setEnabled(video)
        if not video:
            self._video_play_timer.stop()
            self.video_play_btn.setText("Play")

    def _browse_video(self, target: QLineEdit) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Видеофайл",
            "",
            "Video (*.mp4 *.avi *.mkv *.mov);;All (*.*)",
        )
        if path:
            target.setText(path)

    def _browse_folder(self, target: QLineEdit) -> None:
        path = QFileDialog.getExistingDirectory(self, "Папка с изображениями")
        if path:
            target.setText(path)

    def _browse_output_npz(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Файл калибровки",
            self.output_npz.text() or "stereo_calib.npz",
            "NumPy (*.npz)",
        )
        if path:
            if not path.lower().endswith(".npz"):
                path += ".npz"
            self.output_npz.setText(path)

    def _browse_npz_file(self, target: QLineEdit) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Калибровка камеры (.npz)",
            target.text() or "",
            "NumPy (*.npz);;All (*.*)",
        )
        if path:
            target.setText(path)

    def _open_video_source(self) -> None:
        sbs = self.video_sbs_edit.text().strip()
        left = self.video_left_edit.text().strip()
        right = self.video_right_edit.text().strip()
        try:
            if sbs:
                self.video_src.open_sbs(sbs, swap_lr=self.sbs_swap_cb.isChecked())
            elif left and right:
                self.video_src.open_pair(left, right)
            else:
                QMessageBox.warning(
                    self,
                    "Видео",
                    "Укажите Left+Right MP4 или один SBS MP4.",
                )
                return
        except ValueError as exc:
            QMessageBox.warning(self, "Видео", str(exc))
            return
        self.src_video.setChecked(True)
        self.video_slider.blockSignals(True)
        self.video_slider.setEnabled(True)
        self.video_slider.setRange(0, max(0, self.video_src.count - 1))
        self.video_slider.setValue(0)
        self.video_slider.blockSignals(False)
        self._show_video_frames(*self.video_src.current())
        self.statusBar().showMessage(
            f"Видео открыто: {self.video_src.count} кадров", 4000
        )

    def _show_video_frames(
        self, fr_l: np.ndarray | None, fr_r: np.ndarray | None
    ) -> None:
        if fr_l is not None:
            self.left_video.set_frame(fr_l)
            self.left_stats.setText(f"video L | кадр {self.video_src.index + 1}/{self.video_src.count}")
        else:
            self.left_video.clear_frame("Нет left-кадра")
            self.left_stats.setText("—")
        if fr_r is not None:
            self.right_video.set_frame(fr_r)
            self.right_stats.setText(f"video R | кадр {self.video_src.index + 1}/{self.video_src.count}")
        else:
            self.right_video.clear_frame("Нет right-кадра")
            self.right_stats.setText("—")
        self.video_frame_label.setText(
            f"кадр {self.video_src.index + 1} / {self.video_src.count}"
        )

    def _on_video_slider(self, value: int) -> None:
        if not self.video_src.open:
            return
        self._show_video_frames(*self.video_src.seek(value))

    def _video_play_interval_ms(self) -> int:
        speed = max(float(self.video_speed.value()), 0.05)
        return max(1, int(round(33.0 / speed)))

    def _on_video_speed_changed(self, _value: float) -> None:
        if self._video_play_timer.isActive():
            self._video_play_timer.setInterval(self._video_play_interval_ms())

    def _step_video(self, delta: int) -> None:
        if not (self.src_video.isChecked() and self.video_src.open):
            return
        if self._video_play_timer.isActive():
            self._video_play_timer.stop()
            self.video_play_btn.setText("Play")
        fr_l, fr_r = self.video_src.step(int(delta))
        self.video_slider.blockSignals(True)
        self.video_slider.setValue(self.video_src.index)
        self.video_slider.blockSignals(False)
        self._show_video_frames(fr_l, fr_r)

    def _toggle_video_play(self) -> None:
        if not self.video_src.open:
            return
        if self._video_play_timer.isActive():
            self._video_play_timer.stop()
            self.video_play_btn.setText("Play")
        else:
            self._video_play_timer.start(self._video_play_interval_ms())
            self.video_play_btn.setText("Pause")

    def _video_play_step(self) -> None:
        if not self.video_src.open:
            self._video_play_timer.stop()
            return
        nxt = self.video_src.index + 1
        if nxt >= self.video_src.count:
            self._video_play_timer.stop()
            self.video_play_btn.setText("Play")
            return
        fr_l, fr_r = self.video_src.seek(nxt)
        self.video_slider.blockSignals(True)
        self.video_slider.setValue(nxt)
        self.video_slider.blockSignals(False)
        self._show_video_frames(fr_l, fr_r)

    # ---- capture ----

    def _calib_mode(self) -> str:
        if self.mode_mono_l.isChecked():
            return "mono_left"
        if self.mode_mono_r.isChecked():
            return "mono_right"
        return "stereo"

    def _current_frames(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        if self.src_video.isChecked() and self.video_src.open:
            fr_l, fr_r = self.video_src.current()
        else:
            fr_l, fr_r = self._live_preview_l, self._live_preview_r
        # Копии: live-буфер может обновиться, пока открыт диалог Accept.
        if fr_l is not None:
            fr_l = np.ascontiguousarray(fr_l.copy())
        if fr_r is not None:
            fr_r = np.ascontiguousarray(fr_r.copy())
        return fr_l, fr_r

    def _ensure_session(self) -> Path:
        if self._session_dir is not None:
            return self._session_dir
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = Path("calib_pairs_new") / stamp
        root.mkdir(parents=True, exist_ok=True)
        (root / "left").mkdir(exist_ok=True)
        (root / "right").mkdir(exist_ok=True)
        self._session_dir = root
        self._accepted_count = 0
        self.session_label.setText(f"Сессия: {root}")
        self.folder_left.setText(str((root / "left").resolve()))
        self.folder_right.setText(str((root / "right").resolve()))
        return root

    def _new_session(self) -> None:
        self._session_dir = None
        self._accepted_count = 0
        self.accepted_label.setText("Принято: 0")
        self.session_label.setText("Сессия: нет (создастся при первом принятии)")
        self.statusBar().showMessage("Новая сессия — папка создастся при Accept", 3000)

    def _use_session_folders(self) -> None:
        if self._session_dir is None:
            QMessageBox.information(self, "Сессия", "Сначала примите хотя бы один кадр.")
            return
        self.folder_left.setText(str((self._session_dir / "left").resolve()))
        self.folder_right.setText(str((self._session_dir / "right").resolve()))

    def _pause_video_for_capture(self) -> bool:
        """Остановить Play на время просмотра углов. True — было воспроизведение."""
        if not (
            self.src_video.isChecked()
            and self.video_src.open
            and self._video_play_timer.isActive()
        ):
            return False
        self._video_play_timer.stop()
        self.video_play_btn.setText("Play")
        return True

    def _resume_video_after_capture(self, was_playing: bool) -> None:
        if not was_playing:
            return
        if not (self.src_video.isChecked() and self.video_src.open):
            return
        if self.video_src.index + 1 >= self.video_src.count:
            return
        self._video_play_timer.start(self._video_play_interval_ms())
        self.video_play_btn.setText("Pause")

    def _capture_current(self) -> None:
        was_playing = self._pause_video_for_capture()
        try:
            cols = int(self.board_cols.value())
            rows = int(self.board_rows.value())
            mode = self._calib_mode()
            fr_l, fr_r = self._current_frames()

            need_l = mode in ("stereo", "mono_left")
            need_r = mode in ("stereo", "mono_right")
            if need_l and fr_l is None:
                self.statusBar().showMessage("Нет левого кадра", 3000)
                return
            if need_r and fr_r is None:
                self.statusBar().showMessage("Нет правого кадра", 3000)
                return

            corners_l = corners_r = None
            vis_l = vis_r = None
            if need_l:
                assert fr_l is not None
                corners_l = find_board_corners(to_gray(fr_l), cols, rows)
                if corners_l is None:
                    self.statusBar().showMessage("Доска не найдена на LEFT", 4000)
                    QMessageBox.warning(self, "Углы", "Доска не найдена на левом кадре.")
                    return
                vis_l = draw_board_corners(fr_l, corners_l, cols, rows)
            if need_r:
                assert fr_r is not None
                corners_r = find_board_corners(to_gray(fr_r), cols, rows)
                if corners_r is None:
                    self.statusBar().showMessage("Доска не найдена на RIGHT", 4000)
                    QMessageBox.warning(self, "Углы", "Доска не найдена на правом кадре.")
                    return
                vis_r = draw_board_corners(fr_r, corners_r, cols, rows)

            dlg = CornerAcceptDialog(self, vis_l=vis_l, vis_r=vis_r)
            if dlg.exec() != QDialog.DialogCode.Accepted:
                self.statusBar().showMessage("Кадр отбракован", 2000)
                return

            session = self._ensure_session()
            self._accepted_count += 1
            idx = self._accepted_count
            saved: list[str] = []
            if need_l and fr_l is not None:
                path = session / "left" / f"left_{idx:03d}.png"
                cv2.imwrite(str(path), fr_l)
                saved.append(str(path))
            if need_r and fr_r is not None:
                path = session / "right" / f"right_{idx:03d}.png"
                cv2.imwrite(str(path), fr_r)
                saved.append(str(path))
            self.accepted_label.setText(f"Принято: {self._accepted_count} → {session}")
            self.statusBar().showMessage(f"Сохранено: {', '.join(saved)}", 4000)
        finally:
            self._resume_video_after_capture(was_playing)

    # ---- run calibration ----

    def _run_calibration(self) -> None:
        if self._calib_worker is not None and self._calib_worker.isRunning():
            QMessageBox.information(self, "Калибровка", "Уже выполняется…")
            return
        mode = self._calib_mode()
        out = self.output_npz.text().strip() or "stereo_calib.npz"
        debug_dir = str(Path("debugCalib").resolve())
        common = dict(
            cols=int(self.board_cols.value()),
            rows=int(self.board_rows.value()),
            square_size=float(self.board_square.value()),
            fix_k1=self.fix_k1_cb.isChecked(),
            fix_k2=self.fix_k2_cb.isChecked(),
            fix_k3=self.fix_k3_cb.isChecked(),
            fix_tangential=self.fix_tang_cb.isChecked(),
            zero_distortion=self.zero_dist_cb.isChecked(),
            debug_dir=debug_dir,
        )
        if mode == "stereo":
            left_dir = self.folder_left.text().strip()
            right_dir = self.folder_right.text().strip()
            if not left_dir or not right_dir:
                QMessageBox.warning(self, "Папки", "Укажите left и right папки.")
                return
            calib_l = self.calib_left_npz.text().strip() or None
            calib_r = self.calib_right_npz.text().strip() or None
            if (calib_l is None) ^ (calib_r is None):
                QMessageBox.warning(
                    self,
                    "Mono calib",
                    "Нужны оба файла: Calib left .npz и Calib right .npz "
                    "(как --calib-left / --calib-right).",
                )
                return
            kwargs = dict(
                left_dir=left_dir,
                right_dir=right_dir,
                output=out,
                alpha=float(self.alpha_spin.value()),
                rectify_mode=str(self.rectify_box.currentText()),
                calib_left=calib_l,
                calib_right=calib_r,
                **common,
            )
            kind = "stereo"
        else:
            folder = (
                self.folder_left.text().strip()
                if mode == "mono_left"
                else self.folder_right.text().strip()
            )
            if not folder:
                QMessageBox.warning(
                    self,
                    "Папка",
                    "Укажите папку с изображениями (Left для mono L, Right для mono R).",
                )
                return
            kwargs = dict(
                images_dir=folder,
                output=out,
                side="left" if mode == "mono_left" else "right",
                **common,
            )
            kind = "mono"

        self.log_view.clear()
        self.log_view.append("Калибровка запущена…")
        self.run_calib_btn.setEnabled(False)
        self._calib_worker = CalibWorker(kind, kwargs)
        self._calib_worker.finished_ok.connect(self._on_calib_ok)
        self._calib_worker.finished_err.connect(self._on_calib_err)
        self._calib_worker.start()

    def _on_calib_ok(self, path: str, log: list) -> None:
        self.run_calib_btn.setEnabled(True)
        self.log_view.append("\n".join(str(x) for x in log))
        debug_dir = str(Path("debugCalib").resolve())
        self.log_view.append(f"\nУглы доски: {debug_dir}")
        self.log_view.append(f"Готово: {path}")
        self.statusBar().showMessage(
            f"Калибровка сохранена: {path} | углы → {debug_dir}", 8000
        )
        QMessageBox.information(
            self,
            "Готово",
            f"Файл калибровки:\n{path}\n\nКадры с углами:\n{debug_dir}",
        )

    def _on_calib_err(self, message: str) -> None:
        self.run_calib_btn.setEnabled(True)
        self.log_view.append(f"Ошибка: {message}")
        QMessageBox.critical(self, "Калибровка", message)

    def closeEvent(self, event: QCloseEvent) -> None:  # type: ignore[override]
        self._video_play_timer.stop()
        self._display_timer.stop()
        self._offline_timer.stop()
        self.video_src.close()
        self._stop_streams()
        if self.manager:
            self.manager.stop_all()
        _restore_windows_execution_state()
        event.accept()
