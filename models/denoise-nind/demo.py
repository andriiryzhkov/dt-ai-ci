# Demo: run the NIND UNet denoiser ONNX model on an image.
#
# The ONNX export is static at 512x512, so the demo tiles the input,
# mirror-pads tile edges, and stitches with overlap trimming — matching
# what darktable does at runtime.

import argparse
import os
import time

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps


def _run_tiled(session, input_name, arr: np.ndarray,
               tile_size: int = 768, overlap: int = 64) -> np.ndarray:
    """Tiled inference with mirror-padded edges and overlap-trimmed stitching.

    `arr` is (1, 3, H, W) float32; H and W need NOT be multiples of T.
    Returns (1, 3, H, W) float32.

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


def run_inference(model_path, image_path, output_path, max_size=1024,
                  tile_size=768, overlap=64):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    t0 = time.perf_counter()

    print(f"Loading model: {model_path}")
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    t_model = time.perf_counter()
    print(f"  Input name:    {input_name}")
    print(f"  Load model:    {t_model - t0:.3f}s")

    print(f"Loading image: {image_path}")
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    t_image = time.perf_counter()
    print(f"  Original size: {image.size[0]}x{image.size[1]}")
    if max_size > 0:
        image.thumbnail((max_size, max_size), Image.LANCZOS)
        print(f"  Resized to:    {image.size[0]}x{image.size[1]}")
    print(f"  Load image:    {t_image - t_model:.3f}s")

    # Preprocess: RGB [0, 1], BCHW
    arr = np.array(image).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[np.newaxis]

    print(f"Running tiled inference (T={tile_size}, overlap={overlap})...")
    output = _run_tiled(session, input_name, arr,
                        tile_size=tile_size, overlap=overlap)
    t_infer = time.perf_counter()
    print(f"  Inference:     {t_infer - t_image:.3f}s")

    # Postprocess: BCHW -> HWC, clip, uint8
    output = output[0].transpose(1, 2, 0)
    output = np.clip(output, 0, 1)
    output = (output * 255).astype(np.uint8)

    Image.fromarray(output).save(output_path)
    print(f"Saved: {output_path}")
    print(f"  Total:         {time.perf_counter() - t0:.3f}s")


def demo(model, image, output, **kwargs):
    """Entry point for programmatic demo."""
    run_inference(model, image, output,
                  max_size=kwargs.get("max_size", 1024),
                  tile_size=kwargs.get("tile_size", 768),
                  overlap=kwargs.get("overlap", 64))


def main():
    parser = argparse.ArgumentParser(description="NIND UNet ONNX denoising demo.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max-size", type=int, default=1024)
    parser.add_argument("--tile-size", type=int, default=768)
    parser.add_argument("--overlap", type=int, default=64)
    args = parser.parse_args()

    demo(args.model, args.image, args.output,
         max_size=args.max_size,
         tile_size=args.tile_size,
         overlap=args.overlap)


if __name__ == "__main__":
    main()
