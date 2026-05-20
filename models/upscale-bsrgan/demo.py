# Demo: run the BSRGAN x2/x4 ONNX models on an image.
#
# The ONNX exports are static (x2: 512x512 input → 1024x1024 output;
# x4: 256x256 input → 1024x1024 output). The demo tiles the input,
# mirror-pads tile edges, and stitches with overlap trimming — matching
# what darktable does at runtime with OVERLAP_UPSCALE=16.

import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort


def _detect_scale_and_tile(session) -> tuple[int, int]:
    """Infer scale factor and required input tile size from the ONNX session.

    Reads the static input shape and runs a dummy inference to read the
    output shape; scale = output_h / input_h.
    """
    in_shape = session.get_inputs()[0].shape
    tile_h = int(in_shape[2]) if isinstance(in_shape[2], int) else 256
    tile_w = int(in_shape[3]) if isinstance(in_shape[3], int) else 256
    assert tile_h == tile_w, f"non-square tile: {tile_h}x{tile_w}"

    dummy = np.zeros((1, 3, tile_h, tile_w), dtype=np.float32)
    out = session.run(None, {session.get_inputs()[0].name: dummy})[0]
    scale = out.shape[2] // tile_h
    return scale, tile_h


def _run_tiled(session, input_name, arr: np.ndarray,
               tile_size: int, scale: int, overlap: int = 16) -> np.ndarray:
    """Tiled inference with mirror-padded edges and overlap-trimmed stitching.

    `arr` is (1, 3, H, W) float32. Returns (1, 3, H*scale, W*scale) float32.

    step = T - 2·overlap; each tile reads a T×T input window with `overlap`
    border on each side (mirror-padded at the image boundary), and only the
    core (step·scale × step·scale) region of each tile's output is written
    to the final image — which keeps tile seams seamless.

    overlap=16 matches darktable's OVERLAP_UPSCALE constant.
    """
    _, _, H, W = arr.shape
    T = tile_size
    O = overlap
    S = scale
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

    out = np.zeros((1, 3, H * S, W * S), dtype=np.float32)
    for ty in range(n_y):
        core_y = ty * step
        core_h = min(step, H - core_y)
        for tx in range(n_x):
            core_x = tx * step
            core_w = min(step, W - core_x)
            tile = padded[:, :, core_y:core_y + T, core_x:core_x + T]
            tile = np.ascontiguousarray(tile)
            [tile_out] = session.run(None, {input_name: tile})
            out[:, :,
                core_y * S:(core_y + core_h) * S,
                core_x * S:(core_x + core_w) * S] = \
                tile_out[:, :,
                         O * S:(O + core_h) * S,
                         O * S:(O + core_w) * S].astype(np.float32)
    return out


def run_inference(model_path, input_path, output_path, max_size=512,
                  overlap=16):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input image not found at {input_path}")

    t0 = time.perf_counter()
    print(f"Loading ONNX model: {model_path}")
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    scale, tile_size = _detect_scale_and_tile(session)
    input_name = session.get_inputs()[0].name
    print(f"  Scale:         x{scale}")
    print(f"  Tile size:     {tile_size}x{tile_size}")

    print(f"Reading image: {input_path}")
    img = cv2.imread(input_path)
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

    print(f"Running tiled inference (T={tile_size}, overlap={overlap}, scale={scale})...")
    output = _run_tiled(session, input_name, arr,
                        tile_size=tile_size, scale=scale, overlap=overlap)

    output = output[0].transpose(1, 2, 0)
    output = np.clip(output, 0, 1)
    output = (output * 255.0).astype(np.uint8)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    out_h, out_w = output.shape[:2]
    print(f"  Output size:   {out_w}x{out_h}")

    print(f"Saving result to: {output_path}")
    cv2.imwrite(output_path, output)
    print(f"Total: {time.perf_counter() - t0:.3f}s")


def demo(model_dir, image, output, **kwargs):
    """Entry point for programmatic demo. Runs all *.onnx in model_dir."""
    max_size = kwargs.get("max_size", 512)
    overlap = kwargs.get("overlap", 16)
    models = sorted(glob.glob(os.path.join(model_dir, "*.onnx")))
    if not models:
        raise FileNotFoundError(f"No ONNX models found in {model_dir}")
    base, ext = os.path.splitext(output)
    for model_path in models:
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        output_path = f"{base}_{model_name}{ext}"
        print(f"\n--- {model_name} ---")
        run_inference(model_path, image, output_path,
                      max_size=max_size, overlap=overlap)


def main():
    parser = argparse.ArgumentParser(description='Run BSRGAN ONNX Demo')
    parser.add_argument('--model', type=str)
    parser.add_argument('--model-dir', type=str)
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--max-size', type=int, default=512)
    parser.add_argument('--overlap', type=int, default=16)
    args = parser.parse_args()

    if args.model_dir:
        demo(args.model_dir, args.image, args.output,
             max_size=args.max_size, overlap=args.overlap)
    elif args.model:
        run_inference(args.model, args.image, args.output,
                      max_size=args.max_size, overlap=args.overlap)
    else:
        print("Error: provide --model or --model-dir")
        sys.exit(1)


if __name__ == '__main__':
    main()
