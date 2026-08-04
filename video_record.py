 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""П139Н-1: прием одного или двух монохромных UDP-видеопотоков, просмотр и запись MP4.

Команда 0x0101 задаёт тип видео (младшие биты):
  b00 — сырое с сенсора 648×520 MONO16;
  b10 — обработанное 640×512 MONO16 (по умолчанию);
  b11 — с наложением графики 960×512 RGB888 (в стартовом пакете fmt=3).
Кадры больше 640×512 обрезаются по центру до 640×512 для превью/записи/стерео.

Зависимости:
    pip install PySide6 numpy opencv-python
"""
from __future__ import annotations


# ========================= protocol.py =========================
import ipaddress
import socket
import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Union


class PacketType(IntEnum):
    START_FRAME = 0x0001
    VIDEO_ROW = 0x0002
    TELEMETRY = 0x0003
    CONTROL = 0x0010


class VideoFormat(IntEnum):
    MONO16 = 0
    RGB888 = 1
    # В стартовом пакете камера часто пишет тот же код, что в 0x0101:
    # 2 = обработанное mono, 3 = графика RGB888.
    STREAM_PROCESSED = 2
    STREAM_OVERLAY_RGB = 3


def is_rgb_video_format(fmt: int) -> bool:
    return fmt in (
        int(VideoFormat.RGB888),
        int(VideoFormat.STREAM_OVERLAY_RGB),
    )


def row_bytes_hint(width: int, video_format: int, expected: VideoFormat) -> int:
    """Ожидаемый размер строки по fmt стартового пакета / режиму 0x0101."""
    w = max(int(width), 1)
    if is_rgb_video_format(video_format) or expected == VideoFormat.RGB888:
        return w * 3
    return w * 2


class StreamMode(IntEnum):
    """Младшие биты значения команды 0x0101 (тип видео на ПК)."""

    RAW_MONO16 = 0b00  # сырое с сенсора 648×520 mono16
    PROCESSED_MONO16 = 0b10  # обработанное 640×512 mono16
    OVERLAY_RGB888 = 0b11  # с наложением графики 960×512 rgb888


STREAM_MODE_SPECS: dict[StreamMode, tuple[int, int, VideoFormat, str]] = {
    StreamMode.RAW_MONO16: (648, 520, VideoFormat.MONO16, "сырое 648×520 MONO16"),
    StreamMode.PROCESSED_MONO16: (640, 512, VideoFormat.MONO16, "обработанное 640×512 MONO16"),
    StreamMode.OVERLAY_RGB888: (960, 512, VideoFormat.RGB888, "графика 960×512 RGB888"),
}

@dataclass(frozen=True)
class StartFramePacket:
    frame_number: int
    width: int
    height: int
    video_format: int
    label_brightness: int
    label_transparency: int


@dataclass(frozen=True)
class VideoRowPacket:
    row_number: int
    row_data: bytes


@dataclass(frozen=True)
class TelemetryPacket:
    temperature_code: int
    video_port: int
    camera_ip: str


ParsedPacket = Union[StartFramePacket, VideoRowPacket, TelemetryPacket]


class ProtocolError(ValueError):
    pass


def _u16be(data: bytes, offset: int) -> int:
    if len(data) < offset + 2:
        raise ProtocolError("Недостаточная длина UDP-пакета")
    return struct.unpack_from(">H", data, offset)[0]


def parse_udp_payload(data: bytes) -> ParsedPacket:
    """Разбирает полезную нагрузку UDP П139Н-1.

    UDP-сокет ОС удаляет Ethernet/IP/UDP-заголовки, поэтому первый байт
    ``data`` соответствует полю преамбулы из таблиц протокола.
    """
    if len(data) < 4:
        raise ProtocolError("UDP-пакет короче 4 байт")

    preamble, packet_type = struct.unpack_from(">HH", data, 0)
    if preamble != 0:
        raise ProtocolError(f"Неверная преамбула 0x{preamble:04X}")

    if packet_type == PacketType.START_FRAME:
        if len(data) < 14:
            raise ProtocolError("Стартовый пакет короче 14 байт")
        return StartFramePacket(
            frame_number=_u16be(data, 4),
            width=_u16be(data, 6),
            height=_u16be(data, 8),
            video_format=_u16be(data, 10),
            label_brightness=data[12],
            label_transparency=data[13],
        )

    if packet_type == PacketType.VIDEO_ROW:
        if len(data) < 6:
            raise ProtocolError("Видеопакет короче 6 байт")
        return VideoRowPacket(row_number=_u16be(data, 4), row_data=data[6:])

    if packet_type == PacketType.TELEMETRY:
        if len(data) < 12:
            raise ProtocolError("Пакет телеметрии короче 12 байт")
        ip_bytes = data[8:12]
        camera_ip = socket.inet_ntoa(ip_bytes)
        # Отсекаем очевидно незаполненный IP, но не запрещаем нестандартные подсети.
        try:
            ipaddress.ip_address(camera_ip)
        except ValueError as exc:
            raise ProtocolError("Некорректный IP в телеметрии") from exc
        return TelemetryPacket(
            temperature_code=_u16be(data, 4),
            video_port=_u16be(data, 6),
            camera_ip=camera_ip,
        )

    raise ProtocolError(f"Неизвестный тип пакета 0x{packet_type:04X}")


def build_control_packet(address: int, command_value: int, extra_data: int = 0) -> bytes:
    """Создает 14-байтную полезную нагрузку пакета управления.

    Формат: преамбула, тип 0x0010, адрес команды (2 байта), значение команды
    (4 байта), дополнительные данные (4 байта). Все многобайтные поля MSB-first.
    """
    if not 0 <= address <= 0xFFFF:
        raise ValueError("Адрес команды должен помещаться в 16 бит")
    if not 0 <= command_value <= 0xFFFFFFFF:
        raise ValueError("Значение команды должно помещаться в 32 бита")
    if not 0 <= extra_data <= 0xFFFFFFFF:
        raise ValueError("Дополнительные данные должны помещаться в 32 бита")
    return struct.pack(">HHHII", 0, PacketType.CONTROL, address, command_value, extra_data)


def build_start_frame_packet(
    frame_number: int,
    width: int,
    height: int,
    video_format: int,
    brightness: int = 0x80,
    transparency: int = 0,
) -> bytes:
    """Вспомогательная функция для тестового симулятора."""
    return struct.pack(
        ">HHHHHHBB",
        0,
        PacketType.START_FRAME,
        frame_number & 0xFFFF,
        width,
        height,
        video_format,
        brightness & 0xFF,
        transparency & 0xFF,
    )


def build_video_row_packet(row_number: int, row_data: bytes) -> bytes:
    """Вспомогательная функция для тестового симулятора."""
    return struct.pack(">HHH", 0, PacketType.VIDEO_ROW, row_number & 0xFFFF) + row_data


def build_telemetry_packet(temperature_code: int, video_port: int, camera_ip: str) -> bytes:
    """Вспомогательная функция для тестового симулятора."""
    return struct.pack(
        ">HHHH4sH",
        0,
        PacketType.TELEMETRY,
        temperature_code & 0xFFFF,
        video_port & 0xFFFF,
        socket.inet_aton(camera_ip),
        0,
    )


# ========================= frame.py =========================
import os
import threading
import queue
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np



FRAME_WIDTH = 640
FRAME_HEIGHT = 512
# Сырой сенсор / RGB с графикой больше обработанного кадра.
RAW_FRAME_WIDTH = 648
RAW_FRAME_HEIGHT = 520
OVERLAY_FRAME_WIDTH = 960
OVERLAY_FRAME_HEIGHT = 512
# По умолчанию — обработанное MONO16 (команда 0x0101 = 0b10).
DEFAULT_STREAM_MODE = StreamMode.PROCESSED_MONO16
MONO_STREAM_MODE = int(DEFAULT_STREAM_MODE)  # совместимость со старым именем

# AGC для просмотра: отсекаем хвосты гистограммы (горячие точки иначе
# «съедают» весь динамический диапазон при NORM_MINMAX → улица чёрная).
# 1–99% недостаточно: человек ~1–2% кадра уже попадает в верхний хвост.
DISPLAY_PERCENTILE_LO = 2.0
DISPLAY_PERCENTILE_HI = 98.0
_DISPLAY_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


class FastMonoAgc:
    """Быстрый AGC: lo/hi обновляются редко, конвейер на каждом кадре одинаковый.

    Раньше CLAHE включался раз в N кадров — из‑за этого картинка «моргала».
    """

    __slots__ = ("_lo", "_hi", "_frame_i")

    def __init__(self) -> None:
        self._lo = 0.0
        self._hi = 1.0
        self._frame_i = 0

    def convert(self, mono16: np.ndarray, *, light: bool = False) -> np.ndarray:
        if mono16.dtype != np.uint16:
            mono16 = mono16.astype(np.uint16, copy=False)
        self._frame_i += 1
        # Пересчёт границ редко — экономия CPU без смены алгоритма кадр-к-кадру.
        period = 16 if light else 8
        if self._frame_i == 1 or self._frame_i % period == 0:
            flat = mono16.reshape(-1)
            sample = flat[::32] if flat.size > 8192 else flat[::8]
            lo = float(np.percentile(sample, DISPLAY_PERCENTILE_LO))
            hi = float(np.percentile(sample, DISPLAY_PERCENTILE_HI))
            if hi <= lo + 1.0:
                med = float(np.median(sample))
                lo, hi = med - 400.0, med + 400.0
            if hi <= lo:
                hi = lo + 1.0
            if self._frame_i == 1:
                self._lo, self._hi = lo, hi
            else:
                self._lo = 0.85 * self._lo + 0.15 * lo
                self._hi = 0.85 * self._hi + 0.15 * hi

        scale = 255.0 / (self._hi - self._lo)
        out = cv2.convertScaleAbs(mono16, alpha=scale, beta=-self._lo * scale)
        # В light-режиме без CLAHE (нагрузка), иначе CLAHE на каждом кадре —
        # стабильная яркость, без мерцания.
        if light:
            return out
        return _DISPLAY_CLAHE.apply(out)


_DEFAULT_MONO_AGC = FastMonoAgc()


def mono16_to_display8(mono16: np.ndarray, *, light: bool = False) -> np.ndarray:
    """MONO16 → 8-бит для превью/MP4 (быстрый AGC)."""
    return _DEFAULT_MONO_AGC.convert(mono16, light=light)


def crop_to_processed_size(frame: np.ndarray) -> np.ndarray:
    """Центральная обрезка до 640×512 (сырое 648×520, графика 960×512)."""
    h, w = frame.shape[:2]
    if (w, h) == (FRAME_WIDTH, FRAME_HEIGHT):
        return frame if frame.flags["C_CONTIGUOUS"] else np.ascontiguousarray(frame)
    if w < FRAME_WIDTH or h < FRAME_HEIGHT:
        return frame if frame.flags["C_CONTIGUOUS"] else np.ascontiguousarray(frame)
    x0 = (w - FRAME_WIDTH) // 2
    y0 = (h - FRAME_HEIGHT) // 2
    return np.ascontiguousarray(frame[y0 : y0 + FRAME_HEIGHT, x0 : x0 + FRAME_WIDTH])


@dataclass(frozen=True)
class FrameMeta:
    frame_number: int
    video_format: int
    incomplete_frames: int
    invalid_packets: int
    dropped_decode_frames: int = 0


@dataclass(frozen=True)
class LatestFrame:
    frame: np.ndarray
    meta: FrameMeta
    sequence: int
    received_at: float
    total_frames: int
    first_received_at: float


class LatestFrameStore:
    """Хранит только самый свежий декодированный кадр каждой камеры."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: dict[str, LatestFrame] = {}
        self._sequences: dict[str, int] = {}
        self._totals: dict[str, int] = {}
        self._first: dict[str, float] = {}

    def put(self, camera_ip: str, frame: np.ndarray, meta: FrameMeta) -> None:
        now = time.monotonic()
        # Копия: приёмный поток не должен иметь шанс затронуть показанный кадр.
        frame = np.ascontiguousarray(frame.copy())
        with self._lock:
            seq = self._sequences.get(camera_ip, 0) + 1
            total = self._totals.get(camera_ip, 0) + 1
            first = self._first.setdefault(camera_ip, now)
            self._sequences[camera_ip] = seq
            self._totals[camera_ip] = total
            self._latest[camera_ip] = LatestFrame(frame, meta, seq, now, total, first)

    def get(self, camera_ip: str) -> Optional[LatestFrame]:
        with self._lock:
            return self._latest.get(camera_ip)

    def clear(self, camera_ip: Optional[str] = None) -> None:
        with self._lock:
            if camera_ip is None:
                self._latest.clear()
                self._sequences.clear()
                self._totals.clear()
                self._first.clear()
            else:
                self._latest.pop(camera_ip, None)
                self._sequences.pop(camera_ip, None)
                self._totals.pop(camera_ip, None)
                self._first.pop(camera_ip, None)


class FrameAssembler:
    """Собирает кадры с минимальным числом выделений памяти.

    Приемный поток не создает объект на каждый UDP-пакет и не склеивает строки
    в список ``bytes``. Строки сразу копируются в заранее выделенный буфер.
    Полные кадры передаются декодеру через очередь длиной один.
    """

    _MAX_HEIGHT = max(RAW_FRAME_HEIGHT, OVERLAY_FRAME_HEIGHT, FRAME_HEIGHT)
    _MAX_FRAME_BYTES = max(
        RAW_FRAME_WIDTH * RAW_FRAME_HEIGHT * 2,
        OVERLAY_FRAME_WIDTH * OVERLAY_FRAME_HEIGHT * 3,
        FRAME_WIDTH * FRAME_HEIGHT * 3,
    )
    _BUFFER_COUNT = 4

    def __init__(
        self,
        camera_ip: str,
        on_frame: Callable[[str, np.ndarray, FrameMeta], None],
        on_status: Callable[[str, str], None],
        *,
        stream_mode: StreamMode = DEFAULT_STREAM_MODE,
    ) -> None:
        self.camera_ip = camera_ip
        self.on_frame = on_frame
        self.on_status = on_status
        self.stream_mode = StreamMode(stream_mode)
        exp_w, exp_h, exp_fmt, _ = STREAM_MODE_SPECS[self.stream_mode]
        self._expected_width = exp_w
        self._expected_height = exp_h
        self._expected_format = exp_fmt

        self._start: Optional[StartFramePacket] = None
        self._received_count = 0
        self._row_seen = bytearray(self._MAX_HEIGHT)
        self._row_bytes = 0  # целевой размер полной строки
        self._row_bytes_hint = 0
        self._row_frags: dict[int, bytearray] = {}
        self._current_buffer = bytearray(self._MAX_FRAME_BYTES)

        self._incomplete_frames = 0
        self._invalid_packets = 0
        self._dropped_decode_frames = 0
        self._logged_row_layout = False
        self._logged_start = False
        self._diag_packets = 0
        self._row_packets = 0
        self._feed_packets = 0
        self._feed_by_type: dict[int, int] = {}
        self._feed_sizes: list[int] = []
        self._last_stats_at = 0.0
        self._udp_diag = os.environ.get("VIDEO_UDP_DIAG", "").strip() not in (
            "",
            "0",
            "false",
            "False",
        )
        self._diag_path = Path(__file__).resolve().parent / "video_udp_diag.log"
        self._last_incomplete_warn = 0
        self._agc = FastMonoAgc()
        self._fast_row = False  # после определения layout — короткий путь
        self._decode_width = 0
        self._decode_bpp = 0
        self._frame_t0 = 0.0
        # Кадр ~40–80 мс; дольше — риск подмеса поздних строк прошлого кадра.
        self._frame_timeout_s = 0.12
        if self._udp_diag:
            try:
                self._diag_path.write_text("", encoding="utf-8")
            except OSError:
                pass

        self._buffer_pool: queue.LifoQueue[bytearray] = queue.LifoQueue(
            maxsize=self._BUFFER_COUNT - 1
        )
        for _ in range(self._BUFFER_COUNT - 1):
            self._buffer_pool.put_nowait(bytearray(self._MAX_FRAME_BYTES))

        self._decode_queue: queue.Queue[
            Optional[tuple[StartFramePacket, bytearray, int]]
        ] = queue.Queue(maxsize=1)
        self._decode_stop = threading.Event()
        self._decode_thread = threading.Thread(
            target=self._decode_loop,
            name=f"P139-decode-{camera_ip}",
            daemon=True,
        )
        self._decode_thread.start()

    @property
    def incomplete_frames(self) -> int:
        return self._incomplete_frames

    @property
    def invalid_packets(self) -> int:
        return self._invalid_packets

    @staticmethod
    def _u16(data: memoryview, offset: int) -> int:
        return (int(data[offset]) << 8) | int(data[offset + 1])

    def feed(self, payload: bytes | bytearray | memoryview) -> None:
        """Быстрый путь разбора стартовых и строчных видеопакетов."""
        view = payload if isinstance(payload, memoryview) else memoryview(payload)
        size = len(view)
        self._feed_packets += 1

        if size < 4 or view[0] != 0 or view[1] != 0:
            self._invalid_packets += 1
            return

        packet_type = self._u16(view, 2)

        if self._udp_diag:
            self._feed_by_type[packet_type] = self._feed_by_type.get(packet_type, 0) + 1
            if len(self._feed_sizes) < 40:
                self._feed_sizes.append(size)
            if self._feed_packets <= 25 or (
                packet_type != int(PacketType.START_FRAME) and self._diag_packets < 15
            ):
                head = bytes(view[: min(16, size)]).hex()
                msg = (
                    f"UDP#{self._feed_packets} size={size} "
                    f"type=0x{packet_type:04X} head={head}"
                )
                self._diag_log(msg)
                self.on_status(self.camera_ip, msg)
                if packet_type != int(PacketType.START_FRAME):
                    self._diag_packets += 1
            now = time.monotonic()
            if now - self._last_stats_at >= 2.0:
                self._last_stats_at = now
                type_summary = ", ".join(
                    f"0x{t:04X}:{c}" for t, c in sorted(self._feed_by_type.items())
                )
                stats = (
                    f"UDP итого {self._feed_packets} | типы [{type_summary}] | "
                    f"строк {self._row_packets} | incomplete {self._incomplete_frames}"
                )
                self._diag_log(stats)
                self.on_status(self.camera_ip, stats)

        if packet_type == PacketType.START_FRAME:
            if size < 14:
                self._invalid_packets += 1
                return
            packet = StartFramePacket(
                frame_number=self._u16(view, 4),
                width=self._u16(view, 6),
                height=self._u16(view, 8),
                video_format=self._u16(view, 10),
                label_brightness=int(view[12]),
                label_transparency=int(view[13]),
            )
            self._begin_frame(packet)
            return

        if packet_type == PacketType.VIDEO_ROW:
            if size < 6:
                self._invalid_packets += 1
                return
            self._add_row_fast(self._u16(view, 4), view[6:])
            return

        # После старта RGB некоторые прошивки шлют строки с другим type.
        if (
            self._start is not None
            and size >= 6
            and packet_type not in (
                int(PacketType.TELEMETRY),
                int(PacketType.CONTROL),
            )
        ):
            self._add_row_fast(self._u16(view, 4), view[6:])

    def _diag_log(self, message: str) -> None:
        try:
            with open(self._diag_path, "a", encoding="utf-8") as fh:
                fh.write(f"{time.strftime('%H:%M:%S')} {self.camera_ip} {message}\n")
        except OSError:
            pass

    def _begin_frame(self, packet: StartFramePacket) -> None:
        if self._start is not None and self._received_count != self._start.height:
            self._incomplete_frames += 1
            # На батарее часто не успеваем добрать строки — сразу сбрасываем
            # фрагменты, чтобы не копить мусор и не отставать ещё сильнее.
            self._row_frags.clear()
            if (
                self._incomplete_frames >= 5
                and self._incomplete_frames - self._last_incomplete_warn >= 25
            ):
                self._last_incomplete_warn = self._incomplete_frames
                self.on_status(
                    self.camera_ip,
                    f"Много неполных кадров ({self._incomplete_frames}): "
                    f"на батарее включите схему «Высокая производительность» "
                    f"или подключите питание",
                )

        if (
            packet.width != self._expected_width
            or packet.height != self._expected_height
        ):
            self._start = None
            self._received_count = 0
            self.on_status(
                self.camera_ip,
                f"Отклонен кадр {packet.width}x{packet.height}; "
                f"ожидается {self._expected_width}x{self._expected_height} "
                f"(0x0101=0b{int(self.stream_mode):02b})",
            )
            return

        self._start = packet
        self._received_count = 0
        self._row_frags.clear()
        self._frame_t0 = time.monotonic()
        # Не фиксируем 960×3 заранее: камера в fmt=3 часто шлёт те же
        # mono-строки 1280 байт (640×2), что и в b10.
        if self._row_bytes <= 0:
            self._row_bytes = 0
        self._row_bytes_hint = row_bytes_hint(
            packet.width, packet.video_format, self._expected_format
        )
        h = packet.height
        for i in range(h):
            self._row_seen[i] = 0
        if not self._logged_start:
            self._logged_start = True
            kind = "RGB888" if is_rgb_video_format(packet.video_format) or (
                self._expected_format == VideoFormat.RGB888
            ) else "MONO16"
            self.on_status(
                self.camera_ip,
                f"Старт кадра {packet.width}x{packet.height} fmt={packet.video_format} "
                f"({kind}), ожидание строк…",
            )

    def _row_size_candidates(self, width: int) -> tuple[int, ...]:
        """Возможные длины полной строки (байты). RGB часто режется по MTU."""
        w = max(int(width), 1)
        # 640×3 на случай, если в 960-кадре полезны только 640 px RGB.
        extra = (FRAME_WIDTH * 3, FRAME_WIDTH * 2)
        if self._expected_format == VideoFormat.RGB888 or is_rgb_video_format(
            self._start.video_format if self._start else 0
        ):
            return (w * 3, w * 2, w * 4, *extra)
        return (w * 2, w * 3, w * 4, *extra)

    def _match_row_size(self, width: int, nbytes: int) -> int:
        """Точное совпадение с кандидатом bpp, иначе 0."""
        for need in self._row_size_candidates(width):
            if nbytes == need:
                return need
        return 0

    def _map_row_number(self, row_number: int, height: int) -> int:
        """Номер строки изображения или -1, если пакет нужно отбросить.

        Раньше row>=height сжимался через //2 или % — из‑за этого чужие
        пакеты затирали чужие строки, а часть строк оставалась от прошлого
        кадра → горизонтальные «помехи».
        """
        if 0 <= row_number < height:
            return row_number
        # Только явный флаг 0x8000|row (вторая половина), без угадываний.
        if row_number & 0x8000:
            idx = row_number & 0x7FFF
            return idx if idx < height else -1
        return -1

    def _commit_full_row(self, row_number: int, row_data: memoryview) -> None:
        start = self._start
        if start is None:
            return
        expected = self._row_bytes
        if expected <= 0 or self._row_seen[row_number]:
            return
        # Только точная длина: лишние/битые байты дают «шумную» полосу.
        if len(row_data) != expected:
            if len(row_data) < expected:
                return
            row_data = row_data[:expected]
        offset = row_number * expected
        self._current_buffer[offset : offset + expected] = row_data
        self._row_seen[row_number] = 1
        self._received_count += 1

        need_rows = start.height
        if self._row_bytes == FRAME_WIDTH * 2 and start.height > FRAME_HEIGHT:
            need_rows = FRAME_HEIGHT
        if self._received_count != need_rows:
            return
        self._finish_assembled_frame(start, expected)

    def _finish_row_bytes(self, row_number: int, data: memoryview, how: str) -> None:
        start = self._start
        if start is None:
            return
        if not self._logged_row_layout:
            self._logged_row_layout = True
            eff_w = self._effective_width(start.width, self._row_bytes)
            bpp = self._row_bytes // max(eff_w, 1)
            self._decode_width = eff_w
            self._decode_bpp = bpp
            self._fast_row = bpp == 2 and self._row_bytes == eff_w * 2
            self.on_status(
                self.camera_ip,
                f"Строка видео: {self._row_bytes} байт ({bpp} байт/пикс), "
                f"эффект. {eff_w}x{start.height} (старт {start.width}x{start.height}), {how}",
            )
        self._commit_full_row(row_number, data)

    @staticmethod
    def _effective_width(start_width: int, row_bytes: int) -> int:
        """Ширина пикселей по длине строки (старт может врать: 960 при данных 640)."""
        for w in (int(start_width), FRAME_WIDTH, OVERLAY_FRAME_WIDTH, RAW_FRAME_WIDTH):
            if w > 0 and row_bytes % w == 0:
                bpp = row_bytes // w
                if bpp in (2, 3, 4):
                    return w
        return int(start_width)

    def _add_row_fast(self, row_number: int, row_data: memoryview) -> None:
        start = self._start
        if start is None:
            return

        # Слишком долгая сборка — высок шанс подмеса хвоста прошлого кадра.
        if self._frame_t0 and (time.monotonic() - self._frame_t0) > self._frame_timeout_s:
            self._incomplete_frames += 1
            self._start = None
            self._received_count = 0
            self._row_frags.clear()
            return

        row_number = self._map_row_number(row_number, start.height)
        if row_number < 0:
            self._invalid_packets += 1
            return

        # Горячий путь после стабилизации layout (типичный mono 640×2).
        if self._fast_row and self._row_bytes > 0:
            if self._row_seen[row_number]:
                return
            expected = self._row_bytes
            if len(row_data) >= expected:
                offset = row_number * expected
                self._current_buffer[offset : offset + expected] = row_data[:expected]
                self._row_seen[row_number] = 1
                self._received_count += 1
                if self._received_count == start.height:
                    self._finish_assembled_frame(start, expected)
                return
            # Иначе короткие фрагменты — общий путь ниже.

        self._row_packets += 1

        matched_one = self._match_row_size(start.width, len(row_data))
        if self._row_bytes <= 0:
            if matched_one > 0:
                self._row_bytes = matched_one
            else:
                self._row_bytes = getattr(self, "_row_bytes_hint", 0) or row_bytes_hint(
                    start.width, start.video_format, self._expected_format
                )
        elif matched_one > 0 and matched_one != self._row_bytes and not self._logged_row_layout:
            self._row_bytes = matched_one

        expected = self._row_bytes
        if expected <= 0:
            return

        if len(row_data) >= expected:
            self._finish_row_bytes(row_number, row_data[:expected], "целиком")
            return

        frag = self._row_frags.get(row_number)
        if frag is None:
            frag = bytearray()
            self._row_frags[row_number] = frag
        frag.extend(row_data)

        if len(frag) >= expected:
            if not self._logged_row_layout:
                matched = self._match_row_size(start.width, len(frag))
                if matched > 0:
                    self._row_bytes = matched
                    expected = matched
            full = memoryview(frag)[:expected]
            del self._row_frags[row_number]
            self._finish_row_bytes(
                row_number, full, "из фрагментов" if len(row_data) < expected else "целиком"
            )
            return

        if len(frag) > max(self._row_size_candidates(start.width)) + 64:
            del self._row_frags[row_number]
            self._invalid_packets += 1
            if not self._logged_row_layout:
                self._logged_row_layout = True
                self.on_status(
                    self.camera_ip,
                    f"Строка видео: не удалось собрать "
                    f"(накоплено {len(frag)} байт, ширина {start.width}, "
                    f"кусок {len(row_data)} байт)",
                )

    def _finish_assembled_frame(self, start: StartFramePacket, expected: int) -> None:
        completed_start = start
        completed_buffer = self._current_buffer
        completed_row_bytes = expected
        self._start = None
        self._received_count = 0
        self._row_frags.clear()

        try:
            next_buffer = self._buffer_pool.get_nowait()
        except queue.Empty:
            self._dropped_decode_frames += 1
            return

        self._current_buffer = next_buffer
        item = (completed_start, completed_buffer, completed_row_bytes)
        try:
            self._decode_queue.put_nowait(item)
        except queue.Full:
            try:
                stale = self._decode_queue.get_nowait()
            except queue.Empty:
                stale = None
            if stale is not None:
                self._return_buffer(stale[1])
                self._dropped_decode_frames += 1
            try:
                self._decode_queue.put_nowait(item)
            except queue.Full:
                self._return_buffer(completed_buffer)
                self._dropped_decode_frames += 1

    def _return_buffer(self, buf: bytearray) -> None:
        try:
            self._buffer_pool.put_nowait(buf)
        except queue.Full:
            pass

    def stop(self) -> None:
        self._decode_stop.set()
        try:
            self._decode_queue.put_nowait(None)
        except queue.Full:
            try:
                stale = self._decode_queue.get_nowait()
            except queue.Empty:
                stale = None
            if stale is not None:
                self._return_buffer(stale[1])
            try:
                self._decode_queue.put_nowait(None)
            except queue.Full:
                pass
        if self._decode_thread.is_alive():
            self._decode_thread.join(timeout=1.0)

    def _decode_loop(self) -> None:
        _set_current_thread_priority_high()
        while not self._decode_stop.is_set():
            try:
                item = self._decode_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if item is None:
                break
            start, raw_buffer, row_bytes = item
            try:
                frame = self._decode(start, raw_buffer, row_bytes, light=False)
                frame = crop_to_processed_size(frame)
                self.on_frame(
                    self.camera_ip,
                    frame,
                    FrameMeta(
                        frame_number=start.frame_number,
                        video_format=start.video_format,
                        incomplete_frames=self._incomplete_frames,
                        invalid_packets=self._invalid_packets,
                        dropped_decode_frames=self._dropped_decode_frames,
                    ),
                )
            except (ValueError, cv2.error) as exc:
                self._invalid_packets += 1
                self.on_status(self.camera_ip, f"Ошибка декодирования: {exc}")
            finally:
                self._return_buffer(raw_buffer)

    def _decode(
        self,
        start: StartFramePacket,
        raw_buffer: bytearray,
        row_bytes: int,
        *,
        light: bool = False,
    ) -> np.ndarray:
        height = start.height
        width = self._decode_width or FrameAssembler._effective_width(
            start.width, row_bytes
        )
        bpp = self._decode_bpp or (int(row_bytes) // max(int(width), 1))
        if height > FRAME_HEIGHT and width == FRAME_WIDTH and bpp == 2:
            height = FRAME_HEIGHT
        if bpp == 3:
            count = height * width * 3
            rgb = np.frombuffer(raw_buffer, dtype=np.uint8, count=count).reshape(
                height, width, 3
            )
            return np.ascontiguousarray(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if bpp == 4:
            count = height * width * 4
            rgba = np.frombuffer(raw_buffer, dtype=np.uint8, count=count).reshape(
                height, width, 4
            )
            return np.ascontiguousarray(cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR))
        if bpp == 2:
            count = height * width
            if is_rgb_video_format(start.video_format) and width >= 800:
                u16 = np.frombuffer(raw_buffer, dtype="<u2", count=count).reshape(
                    height, width
                )
                r = ((u16 >> 11) & 0x1F).astype(np.uint8) * np.uint8(8)
                g = ((u16 >> 5) & 0x3F).astype(np.uint8) * np.uint8(4)
                b = (u16 & 0x1F).astype(np.uint8) * np.uint8(8)
                return np.ascontiguousarray(np.dstack([b, g, r]))
            mono16 = np.frombuffer(raw_buffer, dtype=">u2", count=count).reshape(
                height, width
            )
            return self._agc.convert(mono16, light=light)
        raise ValueError(
            f"Неподдерживаемый bpp={bpp} (row_bytes={row_bytes}, width={width})"
        )



# ========================= network.py =========================
import os
import socket
import sys
import threading
import time
from dataclasses import dataclass, replace
from typing import Callable, Optional

import numpy as np


TELEMETRY_PORT = 53000
CAMERA_CONTROL_PORT = 52000


def _enable_windows_capture_performance() -> list[str]:
    """Снижает троттлинг Windows на батарее (EcoQoS), чтобы UDP не сыпался.

    На AC обычно и так ок; на батареи Win11 часто уводит процесс в EcoQoS →
    неполные кадры, рассинхрон и лаги записи.
    """
    notes: list[str] = []
    if sys.platform != "win32":
        return notes
    try:
        import ctypes
        from ctypes import wintypes
    except ImportError:
        return notes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.SetPriorityClass.restype = wintypes.BOOL
    kernel32.SetProcessInformation.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    kernel32.SetProcessInformation.restype = wintypes.BOOL
    kernel32.SetThreadExecutionState.argtypes = [wintypes.DWORD]
    kernel32.SetThreadExecutionState.restype = wintypes.DWORD

    ProcessPowerThrottling = 4
    PROCESS_POWER_THROTTLING_CURRENT_VERSION = 1
    PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1
    PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION = 0x4
    ABOVE_NORMAL_PRIORITY_CLASS = 0x00008000
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ES_AWAYMODE_REQUIRED = 0x00000040

    class PROCESS_POWER_THROTTLING_STATE(ctypes.Structure):
        _fields_ = [
            ("Version", wintypes.ULONG),
            ("ControlMask", wintypes.ULONG),
            ("StateMask", wintypes.ULONG),
        ]

    handle = kernel32.GetCurrentProcess()

    def _set_throttle(control: int, state_mask: int) -> bool:
        st = PROCESS_POWER_THROTTLING_STATE()
        st.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION
        st.ControlMask = control
        st.StateMask = state_mask
        return bool(
            kernel32.SetProcessInformation(
                handle,
                ProcessPowerThrottling,
                ctypes.byref(st),
                ctypes.sizeof(st),
            )
        )

    # Выключаем EcoQoS (отдельные вызовы — на части сборок Win один фланг даёт ERROR_INVALID_PARAMETER).
    if _set_throttle(PROCESS_POWER_THROTTLING_EXECUTION_SPEED, 0):
        notes.append("EcoQoS выключен")
    else:
        err = ctypes.get_last_error()
        if err not in (0, 87):
            notes.append(f"EcoQoS err={err}")

    if _set_throttle(PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION, 0):
        notes.append("timer resolution allowed")

    if kernel32.SetPriorityClass(handle, ABOVE_NORMAL_PRIORITY_CLASS):
        notes.append("priority=ABOVE_NORMAL")
    else:
        notes.append(f"SetPriorityClass err={ctypes.get_last_error()}")

    # Не усыплять систему, пока идёт приём/запись.
    kernel32.SetThreadExecutionState(
        ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
    )
    notes.append("sleep inhibited")

    try:
        winmm = ctypes.WinDLL("winmm")
        if winmm.timeBeginPeriod(1) == 0:
            notes.append("timer 1ms")
    except OSError:
        pass
    return notes


def _set_current_thread_priority_high() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentThread.restype = wintypes.HANDLE
        kernel32.SetThreadPriority.argtypes = [wintypes.HANDLE, ctypes.c_int]
        kernel32.SetThreadPriority.restype = wintypes.BOOL
        THREAD_PRIORITY_HIGHEST = 2
        kernel32.SetThreadPriority(kernel32.GetCurrentThread(), THREAD_PRIORITY_HIGHEST)
    except OSError:
        pass


def _restore_windows_execution_state() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ES_CONTINUOUS = 0x80000000
        ctypes.WinDLL("kernel32").SetThreadExecutionState(ES_CONTINUOUS)
    except OSError:
        pass


@dataclass(frozen=True)
class CameraInfo:
    camera_ip: str
    reported_ip: str
    video_port: int
    temperature_code: int
    last_seen: float
    source_ip: str


class DiscoveryService:
    def __init__(
        self,
        bind_ip: str,
        on_camera: Callable[[CameraInfo], None],
        on_error: Callable[[str], None],
    ) -> None:
        self.bind_ip = bind_ip
        self.on_camera = on_camera
        self.on_error = on_error
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._socket: Optional[socket.socket] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="P139-telemetry", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._socket:
            try:
                self._socket.close()
            except OSError:
                pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)

    def _run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket = sock
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 * 1024 * 1024)
        sock.settimeout(0.5)
        try:
            sock.bind((self.bind_ip, TELEMETRY_PORT))
        except OSError as exc:
            self.on_error(f"Не удалось открыть UDP {self.bind_ip}:{TELEMETRY_PORT}: {exc}")
            return

        while not self._stop.is_set():
            try:
                payload, source = sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break

            try:
                packet = parse_udp_payload(payload)
            except ValueError:
                continue
            if not isinstance(packet, TelemetryPacket):
                continue

            source_ip = source[0]
            reported_ip = packet.camera_ip
            camera_ip = source_ip if source_ip != "0.0.0.0" else reported_ip
            if not 1 <= packet.video_port <= 65535:
                continue
            self.on_camera(
                CameraInfo(
                    camera_ip=camera_ip,
                    reported_ip=reported_ip,
                    video_port=packet.video_port,
                    temperature_code=packet.temperature_code,
                    last_seen=time.monotonic(),
                    source_ip=source_ip,
                )
            )


class PortReceiver:
    """Один UDP-сокет на каждый уникальный локальный видеопорт.

    Используется ``recvfrom_into`` с повторно используемым буфером, чтобы не
    выделять новый объект ``bytes`` для каждого из десятков тысяч UDP-пакетов.
    """

    _DATAGRAM_BUFFER_SIZE = 65535
    _REQUESTED_RCVBUF = 64 * 1024 * 1024

    def __init__(self, bind_ip: str, port: int, on_error: Callable[[str], None]) -> None:
        self.bind_ip = bind_ip
        self.port = port
        self.on_error = on_error
        self._callbacks: dict[str, Callable[[memoryview], None]] = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._socket: Optional[socket.socket] = None
        self.actual_rcvbuf = 0
        # Кэш единственного callback — без lock на каждый UDP-пакет.
        self._cached_cb: Optional[Callable[[memoryview], None]] = None
        self._cached_ip: Optional[str] = None

    def register(self, camera_ip: str, callback: Callable[[memoryview], None]) -> None:
        with self._lock:
            self._callbacks[camera_ip] = callback
            self._refresh_cache_locked()
        self.start()

    def unregister(self, camera_ip: str) -> None:
        with self._lock:
            self._callbacks.pop(camera_ip, None)
            self._refresh_cache_locked()

    def _refresh_cache_locked(self) -> None:
        if len(self._callbacks) == 1:
            ip, cb = next(iter(self._callbacks.items()))
            self._cached_ip = ip
            self._cached_cb = cb
        else:
            self._cached_ip = None
            self._cached_cb = None

    def is_empty(self) -> bool:
        with self._lock:
            return not self._callbacks

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=f"P139-video-{self.port}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._socket:
            try:
                self._socket.close()
            except OSError:
                pass
            self._socket = None
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        self._thread = None

    def _run(self) -> None:
        _set_current_thread_priority_high()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket = sock
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self._REQUESTED_RCVBUF)
        except OSError:
            pass
        self.actual_rcvbuf = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)

        if hasattr(socket, "SIO_UDP_CONNRESET"):
            try:
                sock.ioctl(socket.SIO_UDP_CONNRESET, False)
            except OSError:
                pass

        try:
            sock.bind((self.bind_ip, self.port))
        except OSError as exc:
            self.on_error(f"Не удалось открыть видеопорт UDP {self.bind_ip}:{self.port}: {exc}")
            return

        # Блокирующий recv: не крутим пустой цикл на timeout (экономия на батарее).
        # Stop закрывает сокет → OSError → выход.
        sock.setblocking(True)
        buffer = bytearray(self._DATAGRAM_BUFFER_SIZE)
        view = memoryview(buffer)

        while not self._stop.is_set():
            try:
                size, source = sock.recvfrom_into(buffer)
            except OSError:
                break

            cached = self._cached_cb
            if cached is not None:
                cached(view[:size])
            else:
                source_ip = source[0]
                with self._lock:
                    callback = self._callbacks.get(source_ip)
                    if callback is None and self._callbacks:
                        callback = next(iter(self._callbacks.values()))
                if callback is not None:
                    callback(view[:size])

            # Сливаем очередь сокета пачкой — меньше переключений контекста.
            sock.setblocking(False)
            try:
                while not self._stop.is_set():
                    try:
                        size, source = sock.recvfrom_into(buffer)
                    except (BlockingIOError, InterruptedError):
                        break
                    except OSError:
                        return
                    cached = self._cached_cb
                    if cached is not None:
                        cached(view[:size])
                    else:
                        source_ip = source[0]
                        with self._lock:
                            callback = self._callbacks.get(source_ip)
                            if callback is None and self._callbacks:
                                callback = next(iter(self._callbacks.values()))
                        if callback is not None:
                            callback(view[:size])
            finally:
                sock.setblocking(True)


class CameraSession:
    def __init__(
        self,
        info: CameraInfo,
        receiver: PortReceiver,
        on_frame: Callable[[str, np.ndarray, FrameMeta], None],
        on_status: Callable[[str, str], None],
        local_ip: str = "0.0.0.0",
        stream_mode: StreamMode = DEFAULT_STREAM_MODE,
    ) -> None:
        self.info = info
        self.receiver = receiver
        self.local_ip = local_ip
        self.on_status = on_status
        self.stream_mode = StreamMode(stream_mode)
        self.assembler = FrameAssembler(
            info.camera_ip,
            on_frame=on_frame,
            on_status=on_status,
            stream_mode=self.stream_mode,
        )
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self.receiver.register(self.info.camera_ip, self.assembler.feed)
        self._started = True
        # 0x0101: младшие биты = тип видео; 0x0100 включает поток.
        self._send_control(0x0101, int(self.stream_mode))
        time.sleep(0.05)
        self._send_control(0x0100, 1)
        _, _, _, label = STREAM_MODE_SPECS[self.stream_mode]
        self.on_status(
            self.info.camera_ip,
            f"Поток запрошен ({label}, 0x0101=0b{int(self.stream_mode):02b})",
        )

    def stop(self) -> None:
        if not self._started:
            return
        self._send_control(0x0100, 0)
        self.receiver.unregister(self.info.camera_ip)
        self.assembler.stop()
        self._started = False
        self.on_status(self.info.camera_ip, "Поток остановлен")

    def _send_control(self, address: int, value: int) -> None:
        payload = build_control_packet(address, value, 0)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            if self.local_ip and self.local_ip != "0.0.0.0":
                sock.bind((self.local_ip, 0))
            sock.sendto(payload, (self.info.camera_ip, CAMERA_CONTROL_PORT))
        except OSError as exc:
            self.on_status(self.info.camera_ip, f"Ошибка команды UDP: {exc}")
        finally:
            sock.close()


class CameraManager:
    def __init__(
        self,
        bind_ip: str,
        on_camera: Callable[[CameraInfo], None],
        on_frame: Callable[[str, np.ndarray, FrameMeta], None],
        on_status: Callable[[str, str], None],
        on_error: Callable[[str], None],
    ) -> None:
        self.bind_ip = bind_ip
        self.on_camera = on_camera
        self.on_frame = on_frame
        self.on_status = on_status
        self.on_error = on_error
        self.discovery = DiscoveryService("0.0.0.0", self._camera_seen, on_error)
        self.cameras: dict[str, CameraInfo] = {}
        self.receivers: dict[int, PortReceiver] = {}
        self.sessions: dict[str, CameraSession] = {}
        self.stream_mode: StreamMode = DEFAULT_STREAM_MODE
        self._lock = threading.RLock()

    def set_stream_mode(self, mode: StreamMode) -> None:
        self.stream_mode = StreamMode(mode)

    def start_discovery(self) -> None:
        self.discovery.start()

    def stop_discovery(self) -> None:
        self.discovery.stop()

    def _camera_seen(self, info: CameraInfo) -> None:
        with self._lock:
            old = self.cameras.get(info.camera_ip)
            if old and old.video_port != info.video_port and info.camera_ip in self.sessions:
                self.on_status(info.camera_ip, "Видеопорт изменился; перезапустите поток")
            self.cameras[info.camera_ip] = info
        self.on_camera(info)

    def start_camera(self, camera_ip: str) -> None:
        with self._lock:
            if camera_ip in self.sessions:
                return
            info = self.cameras[camera_ip]
            receiver = self.receivers.get(info.video_port)
            if receiver is None:
                receiver = PortReceiver("0.0.0.0", info.video_port, self.on_error)
                self.receivers[info.video_port] = receiver
            session = CameraSession(
                info=info,
                receiver=receiver,
                on_frame=self.on_frame,
                on_status=self.on_status,
                local_ip=self.bind_ip,
                stream_mode=self.stream_mode,
            )
            self.sessions[camera_ip] = session
        session.start()

    def stop_camera(self, camera_ip: str) -> None:
        with self._lock:
            session = self.sessions.pop(camera_ip, None)
        if session is None:
            return
        port = session.info.video_port
        session.stop()
        with self._lock:
            receiver = self.receivers.get(port)
            if receiver and receiver.is_empty():
                receiver.stop()
                self.receivers.pop(port, None)

    def stop_all(self) -> None:
        for camera_ip in list(self.sessions):
            self.stop_camera(camera_ip)
        self.stop_discovery()


# ========================= recording.py =========================
import queue
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class RecordingError(RuntimeError):
    pass


class MultiCameraRecorder:
    """Асинхронная запись: кодирование MP4 не блокирует поток интерфейса."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._active = False
        self._output_dir = Path.cwd()
        self._fps = 25.0
        self._timestamp = ""
        self._selected: set[str] = set()
        self._roles: dict[str, str] = {}
        self._writers: dict[str, cv2.VideoWriter] = {}
        self._paths: dict[str, Path] = {}
        self._queues: dict[str, queue.Queue[Optional[np.ndarray]]] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._dropped: dict[str, int] = {}

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active

    def start(
        self,
        output_dir: str | Path,
        camera_roles: dict[str, str],
        fps: float,
    ) -> None:
        selected = dict(camera_roles)
        if not selected:
            raise RecordingError("Не выбран ни один тепловизор")
        invalid_roles = set(selected.values()) - {"left", "right"}
        if invalid_roles:
            raise RecordingError(f"Недопустимые метки камер: {sorted(invalid_roles)}")
        directory = Path(output_dir).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        with self._lock:
            if self._active:
                raise RecordingError("Запись уже выполняется")
            self._output_dir = directory
            self._fps = float(fps)
            self._timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._selected = set(selected)
            self._roles = selected
            self._writers.clear()
            self._paths.clear()
            self._queues.clear()
            self._threads.clear()
            self._dropped = {ip: 0 for ip in selected}
            try:
                for ip, role in selected.items():
                    self._writers[ip] = self._create_writer(ip, role)
                    self._queues[ip] = queue.Queue(maxsize=1)
                self._active = True
                for ip in selected:
                    thread = threading.Thread(
                        target=self._writer_loop,
                        args=(ip,),
                        name=f"P139-record-{ip}",
                        daemon=True,
                    )
                    self._threads[ip] = thread
                    thread.start()
            except Exception:
                for writer in self._writers.values():
                    writer.release()
                self._writers.clear()
                self._queues.clear()
                self._selected.clear()
                self._roles.clear()
                self._active = False
                raise

    def write(self, camera_ip: str, frame_bgr: np.ndarray) -> None:
        with self._lock:
            if not self._active or camera_ip not in self._selected:
                return
            if frame_bgr.shape[:2] != (FRAME_HEIGHT, FRAME_WIDTH) or frame_bgr.ndim not in (2, 3):
                raise RecordingError(
                    f"Кадр {camera_ip} имеет {frame_bgr.shape[1]}x{frame_bgr.shape[0]}, ожидается 640x512"
                )
            target = self._queues[camera_ip]
        try:
            target.put_nowait(frame_bgr)
        except queue.Full:
            # Для низкой задержки удаляем старейший кадр, а не накапливаем очередь.
            try:
                target.get_nowait()
            except queue.Empty:
                pass
            with self._lock:
                self._dropped[camera_ip] = self._dropped.get(camera_ip, 0) + 1
            try:
                target.put_nowait(frame_bgr)
            except queue.Full:
                pass

    def _writer_loop(self, camera_ip: str) -> None:
        target = self._queues[camera_ip]
        writer = self._writers[camera_ip]
        while True:
            try:
                frame = target.get(timeout=0.25)
            except queue.Empty:
                with self._lock:
                    if not self._active:
                        break
                continue
            if frame is None:
                break
            if frame.ndim == 2:
                frame_to_write = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.ndim == 3 and frame.shape[2] == 3:
                frame_to_write = frame
            else:
                continue
            writer.write(np.ascontiguousarray(frame_to_write))

    def _create_writer(self, camera_ip: str, role: str) -> cv2.VideoWriter:
        safe_ip = re.sub(r"[^0-9A-Za-z_.-]", "_", camera_ip).replace(".", "-")
        path = self._output_dir / f"P139_{role}_{safe_ip}_{self._timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, self._fps, (FRAME_WIDTH, FRAME_HEIGHT), True)
        if not writer.isOpened():
            writer.release()
            raise RecordingError(
                "OpenCV не смог открыть MP4-кодек mp4v. Установите сборку OpenCV с FFmpeg."
            )
        self._paths[camera_ip] = path
        return writer

    def stop(self) -> list[Path]:
        with self._lock:
            if not self._active:
                return []
            self._active = False
            queues = dict(self._queues)
            threads = dict(self._threads)
        for target in queues.values():
            try:
                target.put_nowait(None)
            except queue.Full:
                try:
                    target.get_nowait()
                except queue.Empty:
                    pass
                try:
                    target.put_nowait(None)
                except queue.Full:
                    pass
        for thread in threads.values():
            thread.join(timeout=3.0)
        with self._lock:
            for writer in self._writers.values():
                writer.release()
            paths = list(self._paths.values())
            self._writers.clear()
            self._paths.clear()
            self._queues.clear()
            self._threads.clear()
            self._selected.clear()
            self._roles.clear()
            return paths


class AnnotatedRecorder:
    """Отдельная асинхронная запись аннотированного (overlay) MP4."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._active = False
        self._writer: Optional[cv2.VideoWriter] = None
        self._path: Optional[Path] = None
        self._queue: Optional[queue.Queue[Optional[np.ndarray]]] = None
        self._thread: Optional[threading.Thread] = None
        self._dropped = 0
        self._fps = 25.0
        self._frame_size: Optional[tuple[int, int]] = None

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active

    def start(self, output_dir: str | Path, fps: float) -> Path:
        directory = Path(output_dir).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = directory / f"P139_annotated_{timestamp}.mp4"
        with self._lock:
            if self._active:
                raise RecordingError("Аннотированная запись уже выполняется")
            self._fps = float(fps)
            self._path = path
            self._frame_size = None
            self._writer = None
            self._queue = queue.Queue(maxsize=1)
            self._dropped = 0
            self._active = True
            self._thread = threading.Thread(
                target=self._writer_loop,
                name="P139-annotated-record",
                daemon=True,
            )
            self._thread.start()
        return path

    def write(self, frame_bgr: np.ndarray) -> None:
        with self._lock:
            if not self._active or self._queue is None:
                return
            target = self._queue
            if self._writer is None:
                if frame_bgr.ndim == 2:
                    h, w = frame_bgr.shape
                elif frame_bgr.ndim == 3:
                    h, w = frame_bgr.shape[:2]
                else:
                    return
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    str(self._path), fourcc, self._fps, (w, h), True
                )
                if not writer.isOpened():
                    writer.release()
                    raise RecordingError(
                        "OpenCV не смог открыть MP4 для аннотированной записи."
                    )
                self._writer = writer
                self._frame_size = (w, h)
            elif self._frame_size is not None:
                w, h = self._frame_size
                if frame_bgr.shape[1] != w or frame_bgr.shape[0] != h:
                    frame_bgr = cv2.resize(
                        frame_bgr, (w, h), interpolation=cv2.INTER_AREA
                    )
        try:
            target.put_nowait(frame_bgr)
        except queue.Full:
            try:
                target.get_nowait()
            except queue.Empty:
                pass
            with self._lock:
                self._dropped += 1
            try:
                target.put_nowait(frame_bgr)
            except queue.Full:
                pass

    def _writer_loop(self) -> None:
        while True:
            with self._lock:
                target = self._queue
                active = self._active
            if target is None:
                break
            try:
                frame = target.get(timeout=0.25)
            except queue.Empty:
                if not active:
                    break
                continue
            if frame is None:
                break
            with self._lock:
                writer = self._writer
            if writer is None:
                continue
            if frame.ndim == 2:
                frame_to_write = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.ndim == 3 and frame.shape[2] == 3:
                frame_to_write = frame
            else:
                continue
            writer.write(np.ascontiguousarray(frame_to_write))

    def stop(self) -> list[Path]:
        with self._lock:
            if not self._active:
                return []
            self._active = False
            target = self._queue
            thread = self._thread
            path = self._path
            writer = self._writer
        if target is not None:
            try:
                target.put_nowait(None)
            except queue.Full:
                try:
                    target.get_nowait()
                except queue.Empty:
                    pass
                try:
                    target.put_nowait(None)
                except queue.Full:
                    pass
        if thread is not None:
            thread.join(timeout=3.0)
        with self._lock:
            if writer is not None:
                writer.release()
            self._writer = None
            self._queue = None
            self._thread = None
            self._path = None
            self._frame_size = None
        return [path] if path is not None and path.exists() else []


# ========================= ui.py =========================
import os
import socket
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QObject, QPoint, QRect, Qt, QTimer, Signal
from PySide6.QtGui import QCloseEvent, QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
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
    QRubberBand,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from stereo_live import LiveTrackDepthController



class Bridge(QObject):
    camera_seen = Signal(object)
    frame_ready = Signal(str, object, object)
    camera_status = Signal(str, str)
    error = Signal(str)


class VideoLabel(QLabel):
    def __init__(self, title: str) -> None:
        super().__init__()
        self._title = title
        self._image: Optional[QImage] = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(360, 288)
        self.setStyleSheet("background:#111; color:#bbb; border:1px solid #555;")
        self.setText(title)

    def set_frame(self, frame: np.ndarray) -> None:
        if frame.ndim == 2:
            h, w = frame.shape
            image_format = QImage.Format.Format_Grayscale8
        elif frame.ndim == 3 and frame.shape[2] == 3:
            h, w, _ = frame.shape
            image_format = QImage.Format.Format_BGR888
        else:
            return
        frame = np.ascontiguousarray(frame)
        # copy() обязателен: буфер кадра может быть перезаписан приёмником.
        self._image = QImage(
            frame.data, w, h, int(frame.strides[0]), image_format
        ).copy()
        self._update_pixmap()

    def clear_frame(self, text: Optional[str] = None) -> None:
        self._image = None
        self.clear()
        self.setText(text or self._title)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._update_pixmap()

    def image_size(self) -> Optional[tuple[int, int]]:
        if self._image is None:
            return None
        return self._image.width(), self._image.height()

    def widget_to_image(self, pos: QPoint) -> Optional[tuple[int, int]]:
        """Координаты виджета → пиксели кадра (с учётом letterbox KeepAspectRatio)."""
        if self._image is None:
            return None
        iw, ih = self._image.width(), self._image.height()
        ww, wh = max(self.width(), 1), max(self.height(), 1)
        scale = min(ww / iw, wh / ih)
        dw, dh = iw * scale, ih * scale
        x0 = (ww - dw) * 0.5
        y0 = (wh - dh) * 0.5
        x = (pos.x() - x0) / scale
        y = (pos.y() - y0) / scale
        if 0 <= x < iw and 0 <= y < ih:
            return int(x), int(y)
        return None

    def image_rect_to_widget(self, x: int, y: int, w: int, h: int) -> QRect:
        if self._image is None:
            return QRect()
        iw, ih = self._image.width(), self._image.height()
        ww, wh = max(self.width(), 1), max(self.height(), 1)
        scale = min(ww / iw, wh / ih)
        dw, dh = iw * scale, ih * scale
        x0 = (ww - dw) * 0.5
        y0 = (wh - dh) * 0.5
        return QRect(
            int(round(x0 + x * scale)),
            int(round(y0 + y * scale)),
            max(1, int(round(w * scale))),
            max(1, int(round(h * scale))),
        )

    def _update_pixmap(self) -> None:
        if self._image is None:
            return
        pixmap = QPixmap.fromImage(self._image).scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        self.setPixmap(pixmap)


class InteractiveVideoLabel(VideoLabel):
    """Превью с кликом (цель) и drag-ROI (режим рамки)."""

    clicked = Signal(int, int)  # x, y в координатах кадра
    roi_dragged = Signal(int, int, int, int)  # x, y, w, h
    box_mode_changed = Signal(bool)

    def __init__(self, title: str) -> None:
        super().__init__(title)
        self.setMouseTracking(True)
        self._box_mode = False
        self._drag_origin: Optional[QPoint] = None
        self._rubber: Optional[QRubberBand] = None
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setToolTip(
            "Клик — выбрать объект сразу.\n"
            "R — режим рамки (drag), Esc — отмена рамки."
        )

    @property
    def box_mode(self) -> bool:
        return self._box_mode

    def set_box_mode(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if self._box_mode == enabled:
            return
        self._box_mode = enabled
        self._cancel_drag()
        if enabled:
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.unsetCursor()
        self.box_mode_changed.emit(enabled)

    def _cancel_drag(self) -> None:
        self._drag_origin = None
        if self._rubber is not None:
            self._rubber.hide()

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        if self._box_mode:
            self._drag_origin = event.position().toPoint()
            if self._rubber is None:
                self._rubber = QRubberBand(QRubberBand.Shape.Rectangle, self)
            self._rubber.setGeometry(QRect(self._drag_origin, self._drag_origin))
            self._rubber.show()
            return
        pt = self.widget_to_image(event.position().toPoint())
        if pt is not None:
            self.clicked.emit(pt[0], pt[1])

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if self._box_mode and self._drag_origin is not None and self._rubber is not None:
            rect = QRect(self._drag_origin, event.position().toPoint()).normalized()
            self._rubber.setGeometry(rect)
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._box_mode
            and self._drag_origin is not None
        ):
            rect = QRect(self._drag_origin, event.position().toPoint()).normalized()
            self._cancel_drag()
            p0 = self.widget_to_image(rect.topLeft())
            p1 = self.widget_to_image(rect.bottomRight())
            if p0 is not None and p1 is not None:
                x0, y0 = p0
                x1, y1 = p1
                w, h = max(1, x1 - x0), max(1, y1 - y0)
                if w >= 8 and h >= 8:
                    self.roi_dragged.emit(x0, y0, w, h)
                    self.set_box_mode(False)
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        key = event.key()
        if key == Qt.Key.Key_R:
            self.set_box_mode(not self._box_mode)
            return
        if key == Qt.Key.Key_Escape:
            if self._box_mode:
                self.set_box_mode(False)
                return
        super().keyPressEvent(event)


class TrackViewWindow(QMainWindow):
    """Отдельное окно левой камеры с overlay и live-выбором ROI."""

    closed = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Трекинг / дистанция — левая камера")
        self.resize(960, 720)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

        root = QWidget()
        layout = QVBoxLayout(root)
        self.hint = QLabel(
            "Клик по объекту — трекинг сразу. "
            "R — рамка. Esc — отмена рамки. X — сброс. "
            "[ / ] — полоса дистанции. T — тройной режим. A — auto."
        )
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)
        self.video = InteractiveVideoLabel("Ожидание кадра…")
        self.video.setMinimumSize(640, 512)
        layout.addWidget(self.video, 1)
        self.status = QLabel("—")
        layout.addWidget(self.status)
        self.setCentralWidget(root)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage(
            "Клик=цель | R=рамка | [/]=полоса | T=тройной | A=auto | X=сброс"
        )

        self.video.box_mode_changed.connect(self._on_box_mode)
        QShortcut(QKeySequence("R"), self, activated=self._toggle_box_mode)
        QShortcut(QKeySequence("Escape"), self, activated=self._cancel_box_mode)

    def _toggle_box_mode(self) -> None:
        self.video.set_box_mode(not self.video.box_mode)
        self.video.setFocus()

    def _cancel_box_mode(self) -> None:
        if self.video.box_mode:
            self.video.set_box_mode(False)

    def _on_box_mode(self, enabled: bool) -> None:
        if enabled:
            self.statusBar().showMessage("Режим рамки: зажмите ЛКМ и выделите область")
            self.hint.setText("Режим рамки активен — тяните прямоугольник. Esc — отмена.")
        else:
            self.statusBar().showMessage(
                "Клик=цель | R=рамка | [/]=полоса | T=тройной | A=auto | X=сброс"
            )
            self.hint.setText(
                "Клик по объекту — трекинг сразу. "
                "R — рамка. Esc — отмена рамки. X — сброс. "
                "[ / ] — полоса дистанции. T — тройной режим. A — auto."
            )

    def set_frame(self, frame: np.ndarray) -> None:
        self.video.set_frame(frame)

    def set_status(self, text: str) -> None:
        self.status.setText(text)

    def closeEvent(self, event: QCloseEvent) -> None:  # type: ignore[override]
        self.closed.emit()
        super().closeEvent(event)


class MainWindow(QMainWindow):
    COL_USE = 0
    COL_ROLE = 1
    COL_IP = 2
    COL_REPORTED_IP = 3
    COL_PORT = 4
    COL_TEMP = 5
    COL_LAST = 6
    COL_STATE = 7

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("П139Н-1 — стереопара и запись MP4 (оптимизировано)")
        self.resize(1350, 900)

        self.bridge = Bridge()
        self.bridge.camera_seen.connect(self._on_camera_seen)
        self.bridge.camera_status.connect(self._on_camera_status)
        self.bridge.error.connect(self._show_error)

        self.manager: Optional[CameraManager] = None
        self.recorder = MultiCameraRecorder()
        self.annotated_recorder = AnnotatedRecorder()
        self.latest_store = LatestFrameStore()
        self.live_track = LiveTrackDepthController(
            z_near_m=10.0,
            z_far_m=100.0,
            long_range=None,
            on_log=self._on_live_log,
        )
        self._live_overlay: Optional[np.ndarray] = None
        self._track_window: Optional[TrackViewWindow] = None
        self.camera_rows: dict[str, int] = {}
        self.camera_info: dict[str, CameraInfo] = {}
        self.active_slots: dict[str, int] = {}
        # Порядок IP определяет роли: индекс 0 — left, индекс 1 — right.
        # Он сохраняется при остановке потоков, чтобы роли можно было
        # назначить до запуска камер.
        self.role_order: list[str] = []
        self.latest_frame_time: dict[str, float] = {}
        self.displayed_sequence: dict[str, int] = {}
        self.recorded_sequence: dict[str, int] = {}
        self.next_record_time: dict[str, float] = {}

        self._build_ui()
        self._offline_timer = QTimer(self)
        self._offline_timer.timeout.connect(self._update_online_states)
        self._offline_timer.start(1000)
        self._display_timer = QTimer(self)
        self._display_timer.timeout.connect(self._refresh_latest_frames)
        self._record_timer = QTimer(self)
        self._record_timer.timeout.connect(self._record_latest_frames)
        self._track_timer = QTimer(self)
        self._track_timer.timeout.connect(self._process_live_track)
        self.record_fps.valueChanged.connect(self._apply_fps_limit)
        self._apply_fps_limit()
        self._start_discovery()

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)
        # Не сжимать содержимое ниже минимума — иначе скролл не появится.
        layout.setSizeConstraint(QVBoxLayout.SizeConstraint.SetMinimumSize)
        root.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum,
        )

        network_box = QGroupBox("Сеть")
        net = QGridLayout(network_box)
        net.addWidget(QLabel("Ethernet-IP ПК для отправки команд:"), 0, 0)
        self.bind_ip = QComboBox()
        self.bind_ip.setEditable(True)
        self.bind_ip.setToolTip("Телеметрия и видео принимаются на всех интерфейсах; этот IP задает исходный Ethernet-интерфейс для управляющих команд.")
        self.bind_ip.addItems(self._local_ipv4_addresses())
        self.bind_ip.setCurrentText("0.0.0.0")
        net.addWidget(self.bind_ip, 0, 1)
        self.discovery_btn = QPushButton("Перезапустить обнаружение")
        self.discovery_btn.clicked.connect(self._restart_discovery)
        net.addWidget(self.discovery_btn, 0, 2)

        net.addWidget(QLabel("Тип видео (0x0101):"), 1, 0)
        self.stream_mode_box = QComboBox()
        self.stream_mode_box.setToolTip(
            "Младшие биты команды 0x0101:\n"
            "b00 — сырое 648×520 MONO16\n"
            "b10 — обработанное 640×512 MONO16\n"
            "b11 — с графикой 960×512 RGB888\n"
            "Кадры шире/выше обрезаются по центру до 640×512."
        )
        for mode in (
            StreamMode.PROCESSED_MONO16,
            StreamMode.RAW_MONO16,
            StreamMode.OVERLAY_RGB888,
        ):
            _, _, _, label = STREAM_MODE_SPECS[mode]
            self.stream_mode_box.addItem(
                f"0b{int(mode):02b} — {label}",
                int(mode),
            )
        self.stream_mode_box.setCurrentIndex(0)
        net.addWidget(self.stream_mode_box, 1, 1, 1, 2)
        layout.addWidget(network_box)

        cameras_box = QGroupBox("Обнаруженные тепловизоры (выберите один или два)")
        cameras_layout = QVBoxLayout(cameras_box)
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            ["Исп.", "Роль", "IP источника", "IP в телеметрии", "Видео UDP", "Темп. код", "Последний пакет", "Состояние"]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.itemChanged.connect(self._on_table_item_changed)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self.COL_STATE, QHeaderView.ResizeMode.Stretch)
        cameras_layout.addWidget(self.table)

        camera_buttons = QHBoxLayout()
        self.start_btn = QPushButton("Запустить выбранные")
        self.start_btn.clicked.connect(self._start_selected)
        self.stop_btn = QPushButton("Остановить потоки")
        self.stop_btn.clicked.connect(self._stop_streams)
        camera_buttons.addWidget(self.start_btn)
        camera_buttons.addWidget(self.stop_btn)
        camera_buttons.addStretch(1)
        cameras_layout.addLayout(camera_buttons)
        layout.addWidget(cameras_box)

        previews = QHBoxLayout()
        left_group = QGroupBox("Левый / единственный")
        left_layout = QVBoxLayout(left_group)
        self.left_video = VideoLabel("Нет видеопотока")
        self.left_stats = QLabel("—")
        left_layout.addWidget(self.left_video, 1)
        left_layout.addWidget(self.left_stats)
        previews.addWidget(left_group, 1)

        # Кнопка расположена непосредственно между окнами предпросмотра.
        swap_column = QVBoxLayout()
        swap_column.addStretch(1)
        self.swap_roles_btn = QPushButton("⇄\nL / R")
        self.swap_roles_btn.setFixedSize(72, 72)
        self.swap_roles_btn.setToolTip(
            "Поменять тепловизоры местами: левый станет правым, правый — левым."
        )
        self.swap_roles_btn.setEnabled(False)
        self.swap_roles_btn.clicked.connect(self._swap_roles)
        swap_column.addWidget(self.swap_roles_btn)
        swap_column.addStretch(1)
        previews.addLayout(swap_column)

        right_group = QGroupBox("Правый")
        right_layout = QVBoxLayout(right_group)
        self.right_video = VideoLabel("Нет второго видеопотока")
        self.right_stats = QLabel("—")
        right_layout.addWidget(self.right_video, 1)
        right_layout.addWidget(self.right_stats)
        previews.addWidget(right_group, 1)
        layout.addLayout(previews, 1)

        record_box = QGroupBox("Запись")
        rec = QGridLayout(record_box)
        rec.addWidget(QLabel("Каталог:"), 0, 0)
        self.output_dir = QLineEdit(str(Path.home() / "P139_records"))
        rec.addWidget(self.output_dir, 0, 1)
        browse = QPushButton("Выбрать…")
        browse.clicked.connect(self._choose_output_dir)
        rec.addWidget(browse, 0, 2)
        rec.addWidget(QLabel("FPS (запись / превью / трекинг):"), 1, 0)
        self.record_fps = QSpinBox()
        self.record_fps.setRange(1, 60)
        self.record_fps.setValue(25)
        self.record_fps.setToolTip(
            "Общий лимит кадров/с для записи MP4, обновления превью и трекинга."
        )
        rec.addWidget(self.record_fps, 1, 1)
        self.record_btn = QPushButton("Начать запись MP4")
        self.record_btn.clicked.connect(self._toggle_recording)
        rec.addWidget(self.record_btn, 1, 2)
        self.record_note = QLabel(
            "Метки left/right соответствуют текущим ролям; для одной камеры используется left."
        )
        rec.addWidget(self.record_note, 2, 0, 1, 3)
        layout.addWidget(record_box)

        track_box = QGroupBox("Трекинг / дистанция")
        track = QGridLayout(track_box)
        track.addWidget(QLabel("Калибровка (.npz):"), 0, 0)
        self.calib_path = QLineEdit()
        self.calib_path.setPlaceholderText("stereo_calib.npz")
        track.addWidget(self.calib_path, 0, 1)
        calib_browse = QPushButton("Обзор…")
        calib_browse.clicked.connect(self._choose_calib_file)
        track.addWidget(calib_browse, 0, 2)
        self.calib_load_btn = QPushButton("Загрузить")
        self.calib_load_btn.clicked.connect(self._load_calib)
        track.addWidget(self.calib_load_btn, 0, 3)

        self.track_enable = QCheckBox("Включить трекинг + дистанцию")
        self.track_enable.toggled.connect(self._on_track_enable_toggled)
        track.addWidget(self.track_enable, 1, 0, 1, 2)
        self.annotate_record = QCheckBox("Писать аннотированное видео (отдельный MP4)")
        track.addWidget(self.annotate_record, 1, 2, 1, 2)

        track.addWidget(QLabel("z-near (м):"), 2, 0)
        self.z_near_spin = QDoubleSpinBox()
        self.z_near_spin.setRange(1.0, 5000.0)
        self.z_near_spin.setDecimals(1)
        self.z_near_spin.setSingleStep(1.0)
        self.z_near_spin.setValue(10.0)
        self.z_near_spin.setToolTip("Ближняя граница сцены для SGBM/дистанции.")
        track.addWidget(self.z_near_spin, 2, 1)
        track.addWidget(QLabel("z-far (м):"), 2, 2)
        self.z_far_spin = QDoubleSpinBox()
        self.z_far_spin.setRange(2.0, 10000.0)
        self.z_far_spin.setDecimals(1)
        self.z_far_spin.setSingleStep(5.0)
        self.z_far_spin.setValue(100.0)
        self.z_far_spin.setToolTip("Дальняя граница сцены для SGBM/дистанции.")
        track.addWidget(self.z_far_spin, 2, 3)

        self.long_range_cb = QCheckBox("Long-range")
        self.long_range_cb.setToolTip(
            "Параметры SGBM для дальних сцен. Включается только этой галочкой "
            "(z-far сам по себе long-range не включает). Для режима Auto."
        )
        track.addWidget(self.long_range_cb, 3, 0, 1, 2)
        self.scene_apply_btn = QPushButton("Применить диапазон")
        self.scene_apply_btn.clicked.connect(lambda: self._apply_scene_range())
        track.addWidget(self.scene_apply_btn, 3, 2, 1, 2)

        track.addWidget(QLabel("Режим дисп.:"), 4, 0)
        self.range_mode_box = QComboBox()
        self.range_mode_box.addItem("Auto (z-near/z-far)", "auto")
        self.range_mode_box.addItem("Полосы ([/])", "bands")
        self.range_mode_box.addItem("Тройной (выброс→среднее)", "triple")
        self.range_mode_box.setToolTip(
            "Auto — подбор по z-near/z-far. "
            "Полосы — 3 диапазона, переключение [ / ]. "
            "Тройной — 3 SGBM в точке, отброс выброса, среднее d."
        )
        self.range_mode_box.currentIndexChanged.connect(self._on_range_mode_changed)
        track.addWidget(self.range_mode_box, 4, 1)
        track.addWidget(QLabel("Границы (м):"), 4, 2)
        self.band_edges_edit = QLineEdit("100,500,1000,3000")
        self.band_edges_edit.setToolTip(
            "4+ числа через запятую → полосы, напр. 100,500,1000,3000."
        )
        track.addWidget(self.band_edges_edit, 4, 3)

        band_row = QHBoxLayout()
        self.band_prev_btn = QPushButton("◀ Полоса")
        self.band_prev_btn.clicked.connect(lambda: self._cycle_band(-1))
        band_row.addWidget(self.band_prev_btn)
        self.band_next_btn = QPushButton("Полоса ▶")
        self.band_next_btn.clicked.connect(lambda: self._cycle_band(1))
        band_row.addWidget(self.band_next_btn)
        self.band_apply_btn = QPushButton("Применить границы")
        self.band_apply_btn.clicked.connect(self._apply_band_edges)
        band_row.addWidget(self.band_apply_btn)
        track.addLayout(band_row, 5, 0, 1, 4)

        self.force_gray_cb = QCheckBox("Gray (ч/б)")
        self.force_gray_cb.setChecked(True)
        self.force_gray_cb.setToolTip(
            "Принудительно переводить кадры в gray для трека/превью. "
            "Выключите для цветных камер (BGR). Для SGBM gray всё равно "
            "берётся из яркости. На уже монохромном ТПВ разницы почти нет."
        )
        self.force_gray_cb.toggled.connect(self._on_force_gray_toggled)
        track.addWidget(self.force_gray_cb, 6, 0, 1, 2)
        self.clahe_cb = QCheckBox("CLAHE")
        self.clahe_cb.setChecked(True)
        self.clahe_cb.setToolTip(
            "Локальное повышение контраста на ректифицированном кадре (ТПВ/ИК)."
        )
        self.clahe_cb.toggled.connect(self._on_clahe_toggled)
        track.addWidget(self.clahe_cb, 6, 2, 1, 2)

        self.roi_box_btn = QPushButton("ROI рамкой (окно)")
        self.roi_box_btn.setToolTip("Либо клавиша R в окне трекинга и drag мышью.")
        self.roi_box_btn.clicked.connect(self._select_roi_box_live)
        track.addWidget(self.roi_box_btn, 7, 0)
        self.roi_click_btn = QPushButton("Показать окно трекинга")
        self.roi_click_btn.clicked.connect(self._show_track_window)
        track.addWidget(self.roi_click_btn, 7, 1)
        self.track_reset_btn = QPushButton("Сброс трекинга")
        self.track_reset_btn.clicked.connect(self._reset_tracking)
        track.addWidget(self.track_reset_btn, 7, 2)
        self.track_status = QLabel(
            "Трекинг выключен. В окне: клик = цель, R = рамка, [/]=полоса, T=тройной."
        )
        track.addWidget(self.track_status, 8, 0, 1, 4)
        layout.addWidget(track_box)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setWidget(root)
        self.setCentralWidget(scroll)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ожидание телеметрии UDP 53000")

    def _fps_interval_ms(self) -> int:
        fps = max(int(self.record_fps.value()), 1)
        return max(1, int(round(1000.0 / fps)))

    def _apply_fps_limit(self) -> None:
        """Превью, трекинг и опрос записи — с тем же FPS, что и запись."""
        interval = self._fps_interval_ms()
        self._display_timer.start(interval)
        self._record_timer.start(interval)
        self._track_timer.start(interval)

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
        self.statusBar().showMessage(f"Телеметрия: UDP 0.0.0.0:53000; команды через {bind_ip}")

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
            self.table.setItem(row, self.COL_REPORTED_IP, QTableWidgetItem(info.reported_ip))
            self.table.setItem(row, self.COL_PORT, QTableWidgetItem(str(info.video_port)))
            self.table.setItem(row, self.COL_TEMP, QTableWidgetItem(f"0x{info.temperature_code:04X}"))
            self.table.setItem(row, self.COL_LAST, QTableWidgetItem("сейчас"))
            self.table.setItem(row, self.COL_STATE, QTableWidgetItem("Обнаружен"))
            self.camera_rows[info.camera_ip] = row
            self.table.blockSignals(False)
        else:
            self.table.item(row, self.COL_REPORTED_IP).setText(info.reported_ip)
            self.table.item(row, self.COL_PORT).setText(str(info.video_port))
            self.table.item(row, self.COL_TEMP).setText(f"0x{info.temperature_code:04X}")
            self.table.item(row, self.COL_LAST).setText("сейчас")
            if info.camera_ip not in self.active_slots:
                self.table.item(row, self.COL_STATE).setText("Обнаружен")

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
            QMessageBox.warning(self, "Ограничение", "Можно выбрать не более двух тепловизоров.")
            selected = self._selected_camera_ips()
        self._update_roles(selected)

    def _update_roles(self, selected: list[str]) -> None:
        """Сохраняет назначенный пользователем порядок ролей.

        Уже назначенные камеры не меняют роль при обновлении таблицы.
        Новая выбранная камера добавляется в свободную роль.
        """
        self.role_order = [ip for ip in self.role_order if ip in selected]
        self.role_order.extend(ip for ip in selected if ip not in self.role_order)
        self._apply_role_labels()
        self._update_swap_button_state()

    def _apply_role_labels(self) -> None:
        for ip, row in self.camera_rows.items():
            role = "—"
            if ip in self.role_order:
                index = self.role_order.index(ip)
                role = "Левый / единственный" if index == 0 else "Правый"
            self.table.item(row, self.COL_ROLE).setText(role)

    def _update_swap_button_state(self) -> None:
        if not hasattr(self, "swap_roles_btn"):
            return
        if self.active_slots:
            two_cameras = len(self.active_slots) == 2
        else:
            two_cameras = len(self.role_order) == 2
        self.swap_roles_btn.setEnabled(two_cameras and not self.recorder.active)

    def _swap_roles(self) -> None:
        """Инвертирует left/right для таблицы, предпросмотра и записи."""
        if self.recorder.active:
            QMessageBox.warning(
                self,
                "Идет запись",
                "Остановите запись перед изменением ролей камер.",
            )
            return

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
            # Принудительно перерисовываем оба окна уже в новых ролях.
            for ip in ordered:
                self.displayed_sequence[ip] = 0
            self.left_video.clear_frame("Ожидание левого видеопотока")
            self.right_video.clear_frame("Ожидание правого видеопотока")
            self.left_stats.setText("—")
            self.right_stats.setText("—")
            self._refresh_latest_frames()

        self.statusBar().showMessage(
            f"Роли инвертированы: left — {ordered[0]}, right — {ordered[1]}",
            5000,
        )
        self._update_swap_button_state()

    def _start_selected(self) -> None:
        selected = self._selected_camera_ips()
        if not selected:
            QMessageBox.warning(self, "Нет выбора", "Отметьте один или два тепловизора в таблице.")
            return
        if self.manager is None:
            self._start_discovery()
        assert self.manager is not None

        # Учитываем назначение left/right, заданное кнопкой инверсии.
        self._update_roles(selected)
        ordered = [ip for ip in self.role_order if ip in selected]

        mode_val = int(self.stream_mode_box.currentData())
        self.manager.set_stream_mode(StreamMode(mode_val))

        self._stop_streams()
        notes = _enable_windows_capture_performance()
        self.latest_store.clear()
        self.active_slots = {ip: index for index, ip in enumerate(ordered)}
        self.displayed_sequence = {ip: 0 for ip in ordered}
        self.recorded_sequence = {ip: 0 for ip in ordered}
        self.next_record_time = {ip: 0.0 for ip in ordered}
        for ip in ordered:
            try:
                self.manager.start_camera(ip)
            except (KeyError, OSError) as exc:
                self._show_error(f"Не удалось запустить {ip}: {exc}")
        self._update_swap_button_state()
        _, _, _, label = STREAM_MODE_SPECS[StreamMode(mode_val)]
        self.statusBar().showMessage(
            f"Запрошено потоков: {len(ordered)} | {label}"
            + (f" | {'; '.join(notes)}" if notes else "")
        )

    def _stop_streams(self) -> None:
        if self.recorder.active:
            self._stop_recording()
        if self.manager:
            for ip in list(self.active_slots):
                self.manager.stop_camera(ip)
        self.active_slots.clear()
        self.displayed_sequence.clear()
        self.recorded_sequence.clear()
        _restore_windows_execution_state()
        self.next_record_time.clear()
        self.latest_store.clear()
        self.left_video.clear_frame("Нет видеопотока")
        self.right_video.clear_frame("Нет второго видеопотока")
        self.left_stats.setText("—")
        self.right_stats.setText("—")
        self._update_swap_button_state()

    def _refresh_latest_frames(self) -> None:
        # Интерфейс забирает только последний кадр. Старые кадры никогда не
        # накапливаются в очереди Qt, поэтому задержка остается ограниченной.
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
                f"{camera_ip} | кадр {meta.frame_number} | формат {meta.video_format} | "
                f"{fps:.1f} кадр/с | неполных {meta.incomplete_frames} | "
                f"декод пропущено {meta.dropped_decode_frames} | ошибок {meta.invalid_packets}"
            )
            if slot == 0:
                # При активном окне трекинга левый превью в главном UI не рисуем
                # (экономия CPU — кадр только в отдельном окне).
                if self._track_window_active():
                    if self.left_video.image_size() is not None:
                        self.left_video.clear_frame("См. окно трекинга")
                elif self.track_enable.isChecked() and self._live_overlay is not None:
                    self.left_video.set_frame(self._live_overlay)
                else:
                    self.left_video.set_frame(frame)
                self.left_stats.setText(stats)
            else:
                self.right_video.set_frame(frame)
                self.right_stats.setText(stats)

    def _record_latest_frames(self) -> None:
        if not self.recorder.active:
            return
        now = time.monotonic()
        interval = 1.0 / max(float(self.record_fps.value()), 1.0)
        for camera_ip in list(self.active_slots):
            item = self.latest_store.get(camera_ip)
            if item is None:
                continue
            if item.sequence == self.recorded_sequence.get(camera_ip, 0):
                continue
            if now < self.next_record_time.get(camera_ip, 0.0):
                continue
            try:
                self.recorder.write(camera_ip, item.frame)
            except RecordingError as exc:
                self._show_error(str(exc))
                self._stop_recording()
                return
            self.recorded_sequence[camera_ip] = item.sequence
            self.next_record_time[camera_ip] = now + interval

    def _left_right_ips(self) -> tuple[Optional[str], Optional[str]]:
        by_slot = {slot: ip for ip, slot in self.active_slots.items()}
        return by_slot.get(0), by_slot.get(1)

    def _on_live_log(self, message: str) -> None:
        self.statusBar().showMessage(message, 5000)

    def _update_track_status_label(self) -> None:
        st = self.live_track.status()
        if not st.enabled:
            self.track_status.setText("Трекинг выключен")
            return
        parts: list[str] = []
        if st.tracking_ok and st.roi is not None:
            parts.append("OK")
        elif self.live_track.tracker.initialized:
            parts.append("LOST")
        else:
            parts.append("нет ROI")
        if st.distance_mm is not None:
            parts.append(f"dist={st.distance_mm / 1000.0:.2f} m")
        elif not st.has_calib:
            parts.append("без calib (track-only)")
        if st.disparity_px is not None:
            parts.append(f"disp={st.disparity_px:.1f}px")
        parts.append(f"range=[{st.disp_min},{st.disp_min + st.disp_num})")
        if st.band_label:
            parts.append(st.band_label)
        if st.sgbm_busy:
            parts.append("SGBM…")
        if st.message:
            parts.append(st.message)
        self.track_status.setText(" | ".join(parts))

    def _on_force_gray_toggled(self, checked: bool) -> None:
        self.live_track.set_force_gray(bool(checked))
        self.statusBar().showMessage(
            f"Gray: {'ВКЛ' if checked else 'ВЫКЛ (цвет BGR)'}", 2500
        )

    def _on_clahe_toggled(self, checked: bool) -> None:
        self.live_track.set_clahe(bool(checked))
        self.statusBar().showMessage(
            f"CLAHE: {'ВКЛ' if checked else 'ВЫКЛ'}", 2500
        )

    def _process_live_track(self) -> None:
        if not self.track_enable.isChecked():
            return
        left_ip, right_ip = self._left_right_ips()
        if left_ip is None or right_ip is None:
            self.track_status.setText("Нужны две активные камеры (L/R)")
            return
        item_l = self.latest_store.get(left_ip)
        item_r = self.latest_store.get(right_ip)
        if item_l is None or item_r is None:
            return
        try:
            overlay = self.live_track.process(
                item_l.frame,
                item_r.frame,
                t_l=item_l.received_at,
                t_r=item_r.received_at,
            )
        except Exception as exc:
            self.track_status.setText(f"Ошибка трекинга: {exc}")
            return
        if overlay is not None:
            self._live_overlay = overlay
            if self._track_window_active():
                self._track_window.set_frame(overlay)
            else:
                self.left_video.set_frame(overlay)
            if self.annotated_recorder.active:
                try:
                    self.annotated_recorder.write(overlay)
                except RecordingError as exc:
                    self._show_error(str(exc))
                    self._stop_recording()
                    return
        self._update_track_status_label()
        if self._track_window_active():
            self._track_window.set_status(self.track_status.text())

    def _choose_calib_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Файл калибровки",
            self.calib_path.text() or str(Path.cwd()),
            "NPZ (*.npz);;Все файлы (*.*)",
        )
        if path:
            self.calib_path.setText(path)

    def _load_calib(self) -> None:
        path = self.calib_path.text().strip()
        if not path:
            QMessageBox.warning(self, "Калибровка", "Укажите путь к файлу .npz")
            return
        try:
            self.live_track.load_calib(path)
        except Exception as exc:
            self._show_error(f"Не удалось загрузить калибровку: {exc}")
            return
        self._update_track_status_label()
        self.statusBar().showMessage(f"Калибровка загружена: {path}", 5000)

    def _apply_scene_range(self, *, quiet: bool = False) -> bool:
        z_near = float(self.z_near_spin.value())
        z_far = float(self.z_far_spin.value())
        if z_near >= z_far:
            if not quiet:
                QMessageBox.warning(self, "Диапазон", "Нужно z-near < z-far.")
            return False
        long_range = bool(self.long_range_cb.isChecked())
        try:
            self.live_track.set_scene_range(
                z_near, z_far, long_range=long_range
            )
        except ValueError as exc:
            if not quiet:
                QMessageBox.warning(self, "Диапазон", str(exc))
            return False
        # После ручного z-near/z-far обычно нужен Auto.
        idx = self.range_mode_box.findData("auto")
        if idx >= 0 and self.range_mode_box.currentData() != "auto":
            self.range_mode_box.blockSignals(True)
            self.range_mode_box.setCurrentIndex(idx)
            self.range_mode_box.blockSignals(False)
            self.live_track.set_range_mode("auto")
        if not quiet:
            self.statusBar().showMessage(
                f"Сцена: {z_near:.0f}–{z_far:.0f} м, "
                f"long_range={self.live_track.long_range}",
                5000,
            )
        self._update_track_status_label()
        return True

    def _apply_band_edges(self) -> None:
        text = self.band_edges_edit.text().strip()
        try:
            self.live_track.set_band_edges(text)
        except ValueError as exc:
            QMessageBox.warning(self, "Полосы", str(exc))
            return
        mode = self.range_mode_box.currentData()
        if mode in ("bands", "triple"):
            self.live_track.set_range_mode(str(mode))
        self.statusBar().showMessage(
            f"Границы полос: {text} → {self.live_track.range_mode_label()}",
            5000,
        )
        self._update_track_status_label()

    def _on_range_mode_changed(self, _index: int = 0) -> None:
        mode = self.range_mode_box.currentData()
        if mode is None:
            return
        try:
            if mode in ("bands", "triple"):
                self.live_track.set_band_edges(self.band_edges_edit.text().strip())
            self.live_track.set_range_mode(str(mode))
        except Exception as exc:
            QMessageBox.warning(self, "Режим", str(exc))
            return
        self.statusBar().showMessage(
            f"Режим диспаритета: {self.live_track.range_mode_label()}", 4000
        )
        self._update_track_status_label()

    def _cycle_band(self, delta: int) -> None:
        try:
            self.live_track.set_band_edges(self.band_edges_edit.text().strip())
        except ValueError:
            pass
        if self.live_track.range_mode != "bands":
            idx = self.range_mode_box.findData("bands")
            if idx >= 0:
                self.range_mode_box.blockSignals(True)
                self.range_mode_box.setCurrentIndex(idx)
                self.range_mode_box.blockSignals(False)
            self.live_track.set_range_mode("bands")
        self.live_track.cycle_band(int(delta))
        self.statusBar().showMessage(self.live_track.range_mode_label(), 3000)
        self._update_track_status_label()

    def _toggle_triple_mode(self) -> None:
        self.live_track.toggle_triple()
        mode = self.live_track.range_mode
        idx = self.range_mode_box.findData(mode)
        if idx >= 0:
            self.range_mode_box.blockSignals(True)
            self.range_mode_box.setCurrentIndex(idx)
            self.range_mode_box.blockSignals(False)
        self.statusBar().showMessage(self.live_track.range_mode_label(), 3000)
        self._update_track_status_label()

    def _set_auto_range_mode(self) -> None:
        idx = self.range_mode_box.findData("auto")
        if idx >= 0:
            self.range_mode_box.setCurrentIndex(idx)
        else:
            self.live_track.set_range_mode("auto")
            self._update_track_status_label()

    def _track_window_active(self) -> bool:
        return self._track_window is not None and self._track_window.isVisible()

    def _ensure_track_window(self) -> TrackViewWindow:
        if self._track_window is None:
            win = TrackViewWindow(self)
            win.video.clicked.connect(self._on_track_click)
            win.video.roi_dragged.connect(self._on_track_roi_drag)
            win.closed.connect(self._on_track_window_closed)
            # X в окне трекинга — сброс ROI (как Backspace в CLI).
            QShortcut(QKeySequence("X"), win, activated=self._reset_tracking)
            QShortcut(QKeySequence("["), win, activated=lambda: self._cycle_band(-1))
            QShortcut(QKeySequence("]"), win, activated=lambda: self._cycle_band(1))
            QShortcut(QKeySequence("T"), win, activated=self._toggle_triple_mode)
            QShortcut(QKeySequence("A"), win, activated=self._set_auto_range_mode)
            self._track_window = win
        return self._track_window

    def _show_track_window(self) -> None:
        if not self.track_enable.isChecked():
            QMessageBox.information(self, "Трекинг", "Сначала включите трекинг.")
            return
        win = self._ensure_track_window()
        if self._live_overlay is not None:
            win.set_frame(self._live_overlay)
        win.show()
        win.raise_()
        win.activateWindow()
        win.video.setFocus()
        self.left_video.clear_frame("См. окно трекинга")

    def _on_track_window_closed(self) -> None:
        # Закрытие окна не выключает трекинг — только возвращает превью в main.
        if self.track_enable.isChecked() and self._live_overlay is not None:
            self.left_video.set_frame(self._live_overlay)

    def _on_track_click(self, x: int, y: int) -> None:
        if not self.track_enable.isChecked():
            return
        ok = self.live_track.init_roi_at_point(x, y)
        self._update_track_status_label()
        if self._track_window_active():
            self._track_window.set_status(
                self.track_status.text()
                if ok
                else "Не удалось выбрать объект по клику"
            )

    def _on_track_roi_drag(self, x: int, y: int, w: int, h: int) -> None:
        if not self.track_enable.isChecked():
            return
        ok = self.live_track.init_roi((x, y, w, h))
        self._update_track_status_label()
        if self._track_window_active():
            self._track_window.set_status(
                self.track_status.text() if ok else "ROI рамкой отклонён"
            )

    def _select_roi_box_live(self) -> None:
        """Включить режим рамки в окне трекинга (без OpenCV selectROI)."""
        if not self.track_enable.isChecked():
            QMessageBox.information(self, "ROI", "Сначала включите трекинг.")
            return
        self._show_track_window()
        assert self._track_window is not None
        self._track_window.video.set_box_mode(True)
        self._track_window.video.setFocus()

    def _on_track_enable_toggled(self, checked: bool) -> None:
        if checked:
            if len(self.active_slots) != 2:
                self.track_enable.blockSignals(True)
                self.track_enable.setChecked(False)
                self.track_enable.blockSignals(False)
                QMessageBox.warning(
                    self,
                    "Трекинг",
                    "Для трекинга нужны две активные камеры с ролями Left/Right.",
                )
                return
            if self.live_track.calib is None:
                reply = QMessageBox.question(
                    self,
                    "Без калибровки",
                    "Калибровка не загружена. Включить только трекинг без дистанции?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if reply != QMessageBox.StandardButton.Yes:
                    self.track_enable.blockSignals(True)
                    self.track_enable.setChecked(False)
                    self.track_enable.blockSignals(False)
                    return
                self.live_track.clear_calib()
            self._apply_scene_range(quiet=True)
            self.live_track.set_enabled(True)
            self._show_track_window()
        else:
            self.live_track.set_enabled(False)
            self._live_overlay = None
            self.track_status.setText("Трекинг выключен")
            if self._track_window is not None:
                self._track_window.hide()
        self._update_track_status_label()

    def _reset_tracking(self) -> None:
        self.live_track.reset()
        self._live_overlay = None
        self._update_track_status_label()
        if self._track_window_active():
            self._track_window.set_status("Трекер сброшен — кликните по объекту")

    def _on_camera_status(self, camera_ip: str, status: str) -> None:
        row = self.camera_rows.get(camera_ip)
        if row is not None:
            self.table.item(row, self.COL_STATE).setText(status)
        self.statusBar().showMessage(f"{camera_ip}: {status}", 5000)

    def _update_online_states(self) -> None:
        now = time.monotonic()
        for ip, info in self.camera_info.items():
            row = self.camera_rows[ip]
            age = now - info.last_seen
            self.table.item(row, self.COL_LAST).setText(f"{age:.0f} с")
            if age > 3.5 and ip not in self.active_slots:
                self.table.item(row, self.COL_STATE).setText("Нет телеметрии")

    def _choose_output_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Каталог записи", self.output_dir.text())
        if directory:
            self.output_dir.setText(directory)

    def _toggle_recording(self) -> None:
        if self.recorder.active:
            self._stop_recording()
            return
        camera_roles = {
            ip: ("left" if slot == 0 else "right")
            for ip, slot in sorted(self.active_slots.items(), key=lambda item: item[1])
        }
        if not camera_roles:
            QMessageBox.warning(self, "Нет потока", "Сначала запустите один или два видеопотока.")
            return
        try:
            self.recorder.start(
                self.output_dir.text(),
                camera_roles,
                self.record_fps.value(),
            )
            if self.annotate_record.isChecked():
                self.annotated_recorder.start(
                    self.output_dir.text(),
                    float(self.record_fps.value()),
                )
        except (RecordingError, OSError) as exc:
            self.recorder.stop()
            self.annotated_recorder.stop()
            self._show_error(str(exc))
            return
        self.record_btn.setText("Остановить запись")
        note = "Идет запись…"
        if self.annotate_record.isChecked():
            note += " (+ аннотированный MP4)"
        self.record_note.setText(note)
        self._update_swap_button_state()
        self.statusBar().showMessage("Запись MP4 начата")

    def _stop_recording(self) -> None:
        was_active = self.recorder.active or self.annotated_recorder.active
        paths = self.recorder.stop()
        ann_paths = self.annotated_recorder.stop()
        if not was_active:
            return
        all_paths = list(paths) + list(ann_paths)
        self.record_btn.setText("Начать запись MP4")
        if all_paths:
            text = "Запись завершена: " + ", ".join(str(path) for path in all_paths)
        else:
            text = "Запись остановлена; полные кадры не поступили."
        self.record_note.setText(text)
        self._update_swap_button_state()
        self.statusBar().showMessage(text, 10000)

    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, "Ошибка", message)
        self.statusBar().showMessage(message, 10000)

    def closeEvent(self, event: QCloseEvent) -> None:  # type: ignore[override]
        self._stop_recording()
        self._stop_streams()
        if self._track_window is not None:
            self._track_window.close()
            self._track_window = None
        if self.manager:
            self.manager.stop_all()
        self.live_track.shutdown()
        event.accept()


def run() -> int:
    # Два параллельных декодера быстрее и стабильнее работают без внутренних
    # пулов OpenCV, которые иначе конкурируют друг с другом за ядра процессора.
    cv2.setNumThreads(1)
    try:
        cv2.ocl.setUseOpenCL(False)
    except AttributeError:
        pass
    # Сразу отключаем EcoQoS Windows — на батарее иначе сыпятся UDP-кадры.
    _enable_windows_capture_performance()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    code = app.exec()
    _restore_windows_execution_state()
    return code


if __name__ == "__main__":
    raise SystemExit(run())
