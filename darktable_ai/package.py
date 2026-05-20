"""Package model output as .dtmodel archive."""

from __future__ import annotations

import zipfile
from pathlib import Path

from darktable_ai.config import ModelConfig


def package_model(config: ModelConfig) -> Path:
    """Create a .dtmodel zip archive from the model's output directory."""
    output_dir = config.output_dir
    package_path = config.root_dir / "output" / f"{config.id}.dtmodel"

    if not output_dir.is_dir():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    print(f"  Packaging {config.id}.dtmodel...")

    with zipfile.ZipFile(package_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(output_dir.rglob("*")):
            if file.is_file():
                arcname = f"{config.id}/{file.relative_to(output_dir)}"
                zf.write(file, arcname)

    print(f"  Created: {package_path}")
    return package_path
