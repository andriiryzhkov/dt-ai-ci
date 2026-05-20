"""Validate ONNX model output: check files exist, load, print I/O metadata."""

from __future__ import annotations

import json
import sys
from glob import glob
from pathlib import Path

from darktable_ai.config import ModelConfig


def validate_onnx(path: Path, label: str = "model") -> bool:
    """Load an ONNX model and print its input/output metadata."""
    import onnxruntime as ort

    if not path.is_file():
        print(f"  FAIL: {label} not found: {path}")
        return False

    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  {label}: {path.name} ({size_mb:.1f} MB)")

    try:
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    except Exception as e:
        print(f"  FAIL: cannot load {label}: {e}")
        return False

    print("    Inputs:")
    for inp in session.get_inputs():
        print(f"      {inp.name}: {inp.shape} ({inp.type})")
    print("    Outputs:")
    for out in session.get_outputs():
        print(f"      {out.name}: {out.shape} ({out.type})")

    return True


def validate_config_json(path: Path) -> bool:
    """Validate config.json exists and has required fields."""
    if not path.is_file():
        print(f"  FAIL: config.json not found: {path}")
        return False

    with open(path) as f:
        config = json.load(f)

    required = ["id", "name", "description", "task", "backend", "version"]
    missing = [k for k in required if k not in config]
    if missing:
        print(f"  FAIL: config.json missing fields: {', '.join(missing)}")
        return False

    print(f"  config.json: OK (task={config['task']}, tiling={config.get('tiling', False)})")
    return True


def run_validation(config: ModelConfig) -> None:
    """Validate ONNX output for a model."""
    output_dir = config.output_dir
    print(f"Validating: {config.id}")

    ok = True
    ok &= validate_config_json(output_dir / "config.json")

    if config.type == "split":
        ok &= validate_onnx(output_dir / "encoder.onnx", "encoder")
        ok &= validate_onnx(output_dir / "decoder.onnx", "decoder")
    elif config.type == "multi":
        onnx_files = sorted(Path(p) for p in glob(str(output_dir / "*.onnx")))
        if not onnx_files:
            print(f"  FAIL: no .onnx files found in {output_dir}")
            ok = False
        for onnx_file in onnx_files:
            ok &= validate_onnx(onnx_file, onnx_file.stem)
    else:
        ok &= validate_onnx(output_dir / "model.onnx", "model")

    if ok:
        print("  Result: PASS")
    else:
        print("  Result: FAIL")
        sys.exit(1)
