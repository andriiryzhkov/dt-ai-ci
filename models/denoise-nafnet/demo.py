# Demo: run the NAFNet ONNX denoiser on an image.
#
# The ONNX export is static at 768x768, so the demo tiles the input,
# mirror-pads tile edges, and stitches with overlap trimming — matching
# what darktable does at runtime with OVERLAP_DENOISE=64.

import argparse
import os
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort


def _run_tiled(session, input_name, arr: np.ndarray,
               tile_size: int = 768, overlap: int = 64) -> np.ndarray:
    """Tiled inference with mirror-padded edges and overlap-trimmed stitching.

    `arr` is (1, 3, H, W) float32. Returns (1, 3, H, W) float32.

    step = T - 2·overlap; each tile reads a T×T window with `overlap`
    border on each side (mirror-padded at the image boundary), and only
    the core (step×step) region of each tile is written to the output —
    which keeps tile seams seamless.

    Defaults T=768, overlap=64 match the static-shape ONNX export and
    darktable's OVERLAP_DENOISE constant.
    """
    _, _, H, W = arr.shape
    T = tile_size
    O = overlap
    step = T - 2 * O
    assert step > 0, "tile_size must exceed 2 * overlap"

    n_y = (H + step - 1) // step
    n_x = (W + step - 1) // step

    pad_before = O
    pad_after_y = max(O, (n_y - 1) * step + T - H - O)
    pad_after_x = max(O, (n_x - 1) * step + T - W - O)
    padded = np.pad(
        arr,
        ((0, 0), (0, 0), (pad_before, pad_after_y), (pad_before, pad_after_x)),
        mode="reflect",
    )

    out = np.zeros_like(arr)
    for ty in range(n_y):
        core_y = ty * step
        core_h = min(step, H - core_y)
        for tx in range(n_x):
            core_x = tx * step
            core_w = min(step, W - core_x)
            tile = padded[:, :, core_y:core_y + T, core_x:core_x + T]
            tile = np.ascontiguousarray(tile)
            [tile_out] = session.run(None, {input_name: tile})
            out[:, :, core_y:core_y + core_h, core_x:core_x + core_w] = \
                tile_out[:, :, O:O + core_h, O:O + core_w].astype(np.float32)
    return out


def run_inference(model_path, input_path, output_path, max_size=1024,
                  tile_size=768, overlap=64):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input image not found at {input_path}")

    t0 = time.perf_counter()
    print(f"Loading ONNX model: {model_path}")
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    print(f"  Input name:    {input_name}")
    print(f"  Load model:    {time.perf_counter() - t0:.3f}s")

    print(f"Reading image: {input_path}")
    img = cv2.imread(input_path)  # BGR
    if img is None:
        raise ValueError(f"Could not read image: {input_path}")
    h, w = img.shape[:2]
    print(f"  Original size: {w}x{h}")
    if max_size > 0 and max(h, w) > max_size:
        s = max_size / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        print(f"  Resized to:    {img.shape[1]}x{img.shape[0]}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    arr = img.astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[np.newaxis]

    print(f"Running tiled inference (T={tile_size}, overlap={overlap})...")
    output = _run_tiled(session, input_name, arr,
                        tile_size=tile_size, overlap=overlap)

    output = output[0].transpose(1, 2, 0)
    output = np.clip(output, 0, 1)
    output = (output * 255.0).astype(np.uint8)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    print(f"Saving result to: {output_path}")
    cv2.imwrite(output_path, output)
    print(f"Total: {time.perf_counter() - t0:.3f}s")


def demo(model, image, output, **kwargs):
    """Entry point for programmatic demo."""
    run_inference(model, image, output,
                  max_size=kwargs.get("max_size", 1024),
                  tile_size=kwargs.get("tile_size", 768),
                  overlap=kwargs.get("overlap", 64))


def main():
    parser = argparse.ArgumentParser(description='Run NAFNet ONNX Demo')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--max-size', type=int, default=1024)
    parser.add_argument('--tile-size', type=int, default=768)
    parser.add_argument('--overlap', type=int, default=64)
    args = parser.parse_args()

    demo(args.model, args.image, args.output,
         max_size=args.max_size,
         tile_size=args.tile_size,
         overlap=args.overlap)


if __name__ == '__main__':
    main()
