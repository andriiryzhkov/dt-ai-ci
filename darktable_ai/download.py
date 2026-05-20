"""Download checkpoints from HTTP URLs and Google Drive."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from darktable_ai.config import Checkpoint

GDRIVE_PATTERNS = [
    re.compile(r"^gdrive://(.+)$"),
    re.compile(r"/file/d/([^/]+)"),
    re.compile(r"[?&]id=([^&]+)"),
]


def parse_gdrive_url(url: str) -> str | None:
    """Extract Google Drive file ID from various URL formats."""
    for pattern in GDRIVE_PATTERNS:
        match = pattern.search(url)
        if match:
            return match.group(1)
    return None


def download_file(url: str, dest: Path) -> None:
    """Download a file via curl."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["curl", "-L", "-o", str(dest), url], check=True)


def download_gdrive(file_id: str, dest: Path) -> None:
    """Download a file from Google Drive."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = (
        f"https://drive.usercontent.google.com/download"
        f"?id={file_id}&export=download&confirm=t"
    )
    subprocess.run(["curl", "-L", "-o", str(dest), url], check=True)


def download_checkpoints(
    checkpoints: list[Checkpoint], root_dir: Path
) -> None:
    """Download all checkpoints that don't already exist."""
    for cp in checkpoints:
        dest = root_dir / cp.path
        if dest.exists():
            print(f"  Checkpoint already exists: {dest.name}")
            continue
        print(f"  Downloading: {dest.name}...")
        gdrive_id = parse_gdrive_url(cp.url)
        if gdrive_id:
            download_gdrive(gdrive_id, dest)
        else:
            download_file(cp.url, dest)
