"""
Переименование Snapshot_*.png в left_*.png / right_*.png без удаления существующих файлов.

Первая половина снимков (по времени в имени) -> left_NN.png
Вторая половина -> right_NN.png
Нумерация продолжается после уже существующих left_*/right_* в папке.

Пример:
    python rename_snapshots.py calib_pairs_new
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Переименовать Snapshot_* в left_/right_ без перезаписи.")
    p.add_argument("folder", help="Папка с Snapshot_*.png")
    return p.parse_args()


def sort_key(path: Path) -> str:
    match = re.search(r"(\d{14})", path.stem)
    return match.group(1) if match else path.name


def next_index(folder: Path, prefix: str) -> int:
    indices: list[int] = []
    for path in folder.glob(f"{prefix}_*.png"):
        match = re.match(rf"{prefix}_(\d+)\.png$", path.name, re.IGNORECASE)
        if match:
            indices.append(int(match.group(1)))
    return max(indices, default=-1) + 1


def rename_snapshots(folder: Path) -> list[str]:
    snapshots = sorted(
        [
            p
            for p in folder.iterdir()
            if p.is_file() and p.name.lower().startswith("snapshot_") and p.suffix.lower() == ".png"
        ],
        key=sort_key,
    )
    if not snapshots:
        raise ValueError(f"В '{folder}' нет файлов Snapshot_*.png")

    left_start = next_index(folder, "left")
    right_start = next_index(folder, "right")
    half = len(snapshots) // 2
    if len(snapshots) % 2:
        print(
            f"Предупреждение: нечётное число Snapshot ({len(snapshots)}); "
            f"левая половина получит на 1 кадр больше.",
            file=sys.stderr,
        )
        half = (len(snapshots) + 1) // 2

    log: list[str] = []
    temp: list[Path] = []
    for i, src in enumerate(snapshots):
        tmp = folder / f"__snapshot_tmp_{i:03d}.png"
        src.rename(tmp)
        temp.append(tmp)

    for i, src in enumerate(temp[:half]):
        dst = folder / f"left_{left_start + i:02d}.png"
        if dst.exists():
            raise ValueError(f"Файл уже существует: {dst}")
        src.rename(dst)
        log.append(f"{snapshots[i].name} -> {dst.name}")

    for i, src in enumerate(temp[half:]):
        dst = folder / f"right_{right_start + i:02d}.png"
        if dst.exists():
            raise ValueError(f"Файл уже существует: {dst}")
        src.rename(dst)
        log.append(f"{snapshots[half + i].name} -> {dst.name}")

    return log


def main() -> None:
    args = parse_args()
    folder = Path(args.folder)
    if not folder.is_dir():
        sys.exit(f"Ошибка: папка не найдена: {folder}")

    try:
        log = rename_snapshots(folder)
    except ValueError as exc:
        sys.exit(f"Ошибка: {exc}")

    for line in log:
        print(line)
    print(f"Готово: {len(log)} файлов.")


if __name__ == "__main__":
    main()
