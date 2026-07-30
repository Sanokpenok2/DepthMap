#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""П139Н-1: прием одного или двух монохромных UDP-видеопотоков, просмотр и запись MP4.

Рабочий режим фиксирован: обработанное MONO16, 640x512 (команда 0x02).
Файлы стереопары получают метки ``left`` и ``right`` в имени.
Роли камер можно инвертировать кнопкой между окнами предпросмотра.

Зависимости:
    pip install PySide6 numpy opencv-python

Запуск:
    python p139_stereo_single_mono_swap_roles.py
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
import threading
import queue
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np



FRAME_WIDTH = 640
FRAME_HEIGHT = 512
# Значение команды 0x0101 для обработанного монохромного потока 640x512.
MONO_STREAM_MODE = 0x02

DISPLAY_PERCENTILE_LO = 2.0
DISPLAY_PERCENTILE_HI = 98.0
_DISPLAY_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def mono16_to_display8(mono16: np.ndarray) -> np.ndarray:
    """MONO16 → 8-бит для превью/MP4 (процентили + CLAHE, как экранный AGC)."""
    if mono16.dtype != np.uint16 and mono16.dtype != np.float32:
        mono16 = mono16.astype(np.uint16, copy=False)
    flat = mono16.reshape(-1)
    sample = flat[::8] if flat.size > 4096 else flat
    lo = float(np.percentile(sample, DISPLAY_PERCENTILE_LO))
    hi = float(np.percentile(sample, DISPLAY_PERCENTILE_HI))
    if hi <= lo + 1.0:
        s = sample.astype(np.float32)
        med = float(np.median(s))
        mad = float(np.median(np.abs(s - med))) + 1.0
        lo, hi = med - 5.0 * mad, med + 5.0 * mad
    if hi <= lo:
        return np.zeros(mono16.shape, dtype=np.uint8)
    scale = 255.0 / (hi - lo)
    out = np.clip((mono16.astype(np.float32) - lo) * scale, 0, 255).astype(np.uint8)
    return _DISPLAY_CLAHE.apply(out)

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

    Приемный поток не создает объект на каждый UDP-пакет и не склеивает 512
    объектов ``bytes``. Строки сразу копируются в заранее выделенный буфер.
    Полные кадры передаются декодеру через очередь длиной один: устаревший кадр
    заменяется новым, поэтому задержка не накапливается.
    """

    _ZERO_ROWS = bytes(FRAME_HEIGHT)
    _MAX_FRAME_BYTES = FRAME_WIDTH * FRAME_HEIGHT * 3
    _BUFFER_COUNT = 4

    def __init__(
        self,
        camera_ip: str,
        on_frame: Callable[[str, np.ndarray, FrameMeta], None],
        on_status: Callable[[str, str], None],
    ) -> None:
        self.camera_ip = camera_ip
        self.on_frame = on_frame
        self.on_status = on_status

        self._start: Optional[StartFramePacket] = None
        self._received_count = 0
        self._row_seen = bytearray(FRAME_HEIGHT)
        self._current_buffer = bytearray(self._MAX_FRAME_BYTES)

        self._incomplete_frames = 0
        self._invalid_packets = 0
        self._dropped_decode_frames = 0

        self._buffer_pool: queue.LifoQueue[bytearray] = queue.LifoQueue(
            maxsize=self._BUFFER_COUNT - 1
        )
        for _ in range(self._BUFFER_COUNT - 1):
            self._buffer_pool.put_nowait(bytearray(self._MAX_FRAME_BYTES))

        self._decode_queue: queue.Queue[
            Optional[tuple[StartFramePacket, bytearray]]
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
        if size < 4 or view[0] != 0 or view[1] != 0:
            self._invalid_packets += 1
            return

        packet_type = self._u16(view, 2)
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

    def _begin_frame(self, packet: StartFramePacket) -> None:
        if self._start is not None and self._received_count != self._start.height:
            self._incomplete_frames += 1

        if packet.width != FRAME_WIDTH or packet.height != FRAME_HEIGHT:
            self._start = None
            self._received_count = 0
            self.on_status(
                self.camera_ip,
                f"Отклонен кадр {packet.width}x{packet.height}; требуется 640x512",
            )
            return

        if packet.video_format != VideoFormat.MONO16:
            self._start = None
            self._received_count = 0
            self.on_status(
                self.camera_ip,
                f"Отклонен немонохромный формат видео: {packet.video_format}",
            )
            return

        self._start = packet
        self._received_count = 0
        self._row_seen[:] = self._ZERO_ROWS

    def _add_row_fast(self, row_number: int, row_data: memoryview) -> None:
        start = self._start
        if start is None:
            return
        if row_number >= FRAME_HEIGHT:
            self._invalid_packets += 1
            return

        expected = self._expected_row_bytes(start.video_format, FRAME_WIDTH)
        if expected <= 0 or len(row_data) < expected:
            self._invalid_packets += 1
            return
        if self._row_seen[row_number]:
            return

        offset = row_number * expected
        self._current_buffer[offset : offset + expected] = row_data[:expected]
        self._row_seen[row_number] = 1
        self._received_count += 1

        if self._received_count != FRAME_HEIGHT:
            return

        completed_start = start
        completed_buffer = self._current_buffer
        self._start = None
        self._received_count = 0

        try:
            next_buffer = self._buffer_pool.get_nowait()
        except queue.Empty:
            # Декодер не успевает. Текущий буфер сразу переиспользуется, а
            # завершенный кадр сознательно пропускается ради минимальной задержки.
            self._dropped_decode_frames += 1
            return

        self._current_buffer = next_buffer
        item = (completed_start, completed_buffer)
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

    def _return_buffer(self, buffer: bytearray) -> None:
        try:
            self._buffer_pool.put_nowait(buffer)
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
        while not self._decode_stop.is_set():
            try:
                item = self._decode_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if item is None:
                break
            start, raw_buffer = item
            try:
                frame = self._decode(start, raw_buffer)
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

    @staticmethod
    def _expected_row_bytes(video_format: int, width: int) -> int:
        if video_format != VideoFormat.MONO16:
            return 0
        return width * 2

    @staticmethod
    def _decode(start: StartFramePacket, raw_buffer: bytearray) -> np.ndarray:
        if start.video_format != VideoFormat.MONO16:
            raise ValueError(f"Ожидался MONO16, получен формат {start.video_format}")

        height, width = start.height, start.width
        count = height * width
        mono16 = np.frombuffer(raw_buffer, dtype=">u2", count=count).reshape(height, width)
        # Для просмотра и MP4 преобразуем 16-битный монохромный кадр в 8 бит.
        return mono16_to_display8(mono16)


# ========================= network.py =========================
import socket
import threading
import time
from dataclasses import dataclass, replace
from typing import Callable, Optional

import numpy as np



TELEMETRY_PORT = 53000
CAMERA_CONTROL_PORT = 52000


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

    def register(self, camera_ip: str, callback: Callable[[memoryview], None]) -> None:
        with self._lock:
            self._callbacks[camera_ip] = callback
        self.start()

    def unregister(self, camera_ip: str) -> None:
        with self._lock:
            self._callbacks.pop(camera_ip, None)

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
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)

    def _run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket = sock
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self._REQUESTED_RCVBUF)
        except OSError:
            pass
        self.actual_rcvbuf = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)

        # Windows может генерировать WSAECONNRESET после ICMP Port Unreachable.
        # Для постоянно работающего UDP-приемника это поведение нежелательно.
        if hasattr(socket, "SIO_UDP_CONNRESET"):
            try:
                sock.ioctl(socket.SIO_UDP_CONNRESET, False)
            except OSError:
                pass

        sock.settimeout(0.25)
        try:
            sock.bind((self.bind_ip, self.port))
        except OSError as exc:
            self.on_error(f"Не удалось открыть видеопорт UDP {self.bind_ip}:{self.port}: {exc}")
            return

        buffer = bytearray(self._DATAGRAM_BUFFER_SIZE)
        view = memoryview(buffer)
        while not self._stop.is_set():
            try:
                size, source = sock.recvfrom_into(buffer)
            except socket.timeout:
                continue
            except OSError:
                break

            source_ip = source[0]
            with self._lock:
                callback = self._callbacks.get(source_ip)
                if callback is None and len(self._callbacks) == 1:
                    callback = next(iter(self._callbacks.values()))
            if callback is not None:
                # Callback обязан синхронно скопировать нужные данные до следующего recv.
                callback(view[:size])


class CameraSession:
    def __init__(
        self,
        info: CameraInfo,
        receiver: PortReceiver,
        on_frame: Callable[[str, np.ndarray, FrameMeta], None],
        on_status: Callable[[str, str], None],
        local_ip: str = "0.0.0.0",
    ) -> None:
        self.info = info
        self.receiver = receiver
        self.local_ip = local_ip
        self.on_status = on_status
        self.assembler = FrameAssembler(info.camera_ip, on_frame=on_frame, on_status=on_status)
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self.receiver.register(self.info.camera_ip, self.assembler.feed)
        self._started = True
        # 0x0101 фиксирует обработанный MONO16 640x512; 0x0100 включает поток.
        self._send_control(0x0101, MONO_STREAM_MODE)
        time.sleep(0.05)
        self._send_control(0x0100, 1)
        self.on_status(self.info.camera_ip, "Поток запрошен")

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
        self._lock = threading.RLock()

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
import re
import threading
from datetime import datetime
from pathlib import Path

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


# ========================= ui.py =========================
import os
import socket
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QCloseEvent, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
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
    QSpinBox,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)



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

    def _update_pixmap(self) -> None:
        if self._image is None:
            return
        pixmap = QPixmap.fromImage(self._image).scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        self.setPixmap(pixmap)


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
        self.latest_store = LatestFrameStore()
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
        self._display_timer.start(33)
        self._record_timer = QTimer(self)
        self._record_timer.timeout.connect(self._record_latest_frames)
        self._record_timer.start(10)
        self._start_discovery()

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)

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

        mono_note = QLabel("Режим: обработанное MONO16, 640x512")
        mono_note.setToolTip("Режим фиксирован; тепловизорам отправляется команда 0x02.")
        net.addWidget(mono_note, 1, 0, 1, 3)
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
        rec.addWidget(QLabel("FPS файла:"), 1, 0)
        self.record_fps = QSpinBox()
        self.record_fps.setRange(1, 60)
        self.record_fps.setValue(25)
        rec.addWidget(self.record_fps, 1, 1)
        self.record_btn = QPushButton("Начать запись MP4")
        self.record_btn.clicked.connect(self._toggle_recording)
        rec.addWidget(self.record_btn, 1, 2)
        self.record_note = QLabel(
            "Метки left/right соответствуют текущим ролям; для одной камеры используется left."
        )
        rec.addWidget(self.record_note, 2, 0, 1, 3)
        layout.addWidget(record_box)

        self.setCentralWidget(root)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ожидание телеметрии UDP 53000")

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

        self._stop_streams()
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
        self.statusBar().showMessage(f"Запрошено потоков: {len(ordered)}")

    def _stop_streams(self) -> None:
        if self.recorder.active:
            self._stop_recording()
        if self.manager:
            for ip in list(self.active_slots):
                self.manager.stop_camera(ip)
        self.active_slots.clear()
        self.displayed_sequence.clear()
        self.recorded_sequence.clear()
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
        except (RecordingError, OSError) as exc:
            self._show_error(str(exc))
            return
        self.record_btn.setText("Остановить запись")
        self.record_note.setText("Идет запись…")
        self._update_swap_button_state()
        self.statusBar().showMessage("Запись MP4 начата")

    def _stop_recording(self) -> None:
        paths = self.recorder.stop()
        self.record_btn.setText("Начать запись MP4")
        if paths:
            text = "Запись завершена: " + ", ".join(str(path) for path in paths)
        else:
            text = "Запись остановлена; полные кадры не поступили."
        self.record_note.setText(text)
        self._update_swap_button_state()
        self.statusBar().showMessage(text, 10000)

    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, "Ошибка", message)
        self.statusBar().showMessage(message, 10000)

    def closeEvent(self, event: QCloseEvent) -> None:  # type: ignore[override]
        self._stop_streams()
        if self.manager:
            self.manager.stop_all()
        event.accept()


def run() -> int:
    # Два параллельных декодера быстрее и стабильнее работают без внутренних
    # пулов OpenCV, которые иначе конкурируют друг с другом за ядра процессора.
    cv2.setNumThreads(1)
    try:
        cv2.ocl.setUseOpenCL(False)
    except AttributeError:
        pass
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(run())
