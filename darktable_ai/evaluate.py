"""Run evaluation benchmarks on models."""

from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


def run_evaluation(
    task: str,
    model_id: str,
    root_dir: Path,
    extra_args: tuple[str, ...] = (),
) -> None:
    """Run evaluation script for a task/model combination."""
    model_dir = root_dir / "output" / model_id
    encoder = model_dir / "encoder.onnx"
    decoder = model_dir / "decoder.onnx"

    if not encoder.exists() or not decoder.exists():
        raise FileNotFoundError(
            f"ONNX models not found in {model_dir}. "
            f"Run: dtai convert {model_id}"
        )

    # Download DAVIS dataset if needed (for mask tasks)
    dataset_path = root_dir / "temp" / "DAVIS"
    if task.startswith("mask") and not (dataset_path / "JPEGImages").exists():
        _download_davis(root_dir)

    # Parse extra CLI args into kwargs
    kwargs = _parse_extra_args(extra_args)

    if task.startswith("mask"):
        from darktable_ai.evaluation.mask import evaluate
        evaluate(
            encoder=encoder,
            decoder=decoder,
            dataset_path=dataset_path,
            **kwargs,
        )
    else:
        raise ValueError(f"No evaluation script for task '{task}'")


def _parse_extra_args(args: tuple[str, ...]) -> dict:
    """Parse CLI-style extra args into kwargs for evaluate()."""
    kwargs = {}
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--"):
            key = arg[2:].replace("-", "_")
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                value = args[i + 1]
                # Try numeric conversion
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                kwargs[key] = value
                i += 2
            else:
                kwargs[key] = True
                i += 1
        else:
            i += 1
    return kwargs


def _download_davis(root_dir: Path) -> None:
    """Download and extract DAVIS-2017-trainval-480p dataset."""
    temp_dir = root_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    zip_path = temp_dir / "davis-2017-trainval-480p.zip"

    if not zip_path.exists():
        url = "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip"
        print(f"Downloading DAVIS dataset...")
        urlretrieve(url, zip_path)

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(temp_dir)
