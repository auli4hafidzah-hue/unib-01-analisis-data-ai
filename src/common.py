"""Path bersama untuk semua repo studi kasus UNIB.

Setiap repo punya salinan file ini supaya bisa berdiri sendiri (tanpa bergantung
pada repo lain). Struktur folder yang diharapkan:

    <repo>/
      src/common.py
      src/generate_*.py
      src/materi*.py
      data/      (dibuat saat runtime)
      outputs/   (dibuat saat runtime)
"""
from __future__ import annotations

import os

DATA_DIR = os.environ.get("UNIB_DATA_DIR", "data")
OUT_DIR = os.environ.get("UNIB_OUT_DIR", "outputs")


def ensure_dirs() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)


def data_path(name: str) -> str:
    """Path file dataset (belum tentu ada -- generate dulu)."""
    return os.path.join(DATA_DIR, name)


def out_path(name: str) -> str:
    ensure_dirs()
    return os.path.join(OUT_DIR, name)


def write_report(name: str, text: str) -> str:
    """Simpan laporan .txt dan kembalikan path-nya."""
    path = out_path(name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def seed_default() -> int:
    return 20261021
