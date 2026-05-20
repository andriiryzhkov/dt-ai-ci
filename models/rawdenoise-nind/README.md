# RawNIND UtNet2 (Bayer + Linear Rec.2020 variants)

Two UtNet2 raw denoisers trained on the Raw Natural Image Noise Dataset
(RawNIND). Bundled into a single `type: multi` package with sensor-based
auto-dispatch.

| Variant        | Input                        | Output                                   | Use for                          |
|----------------|------------------------------|------------------------------------------|----------------------------------|
| `model_bayer`  | 4ch packed Bayer [R,G1,G2,B] | 3ch camRGB at 2× spatial, arbitrary gain | Bayer sensors (pre-demosaic)     |
| `model_linear` | 3ch linear Rec.2020          | 3ch linear Rec.2020, arbitrary gain      | X-Trans, Foveon, post-demosaic   |

Both models perform the same denoising task; the Bayer variant additionally
does the demosaic (via a PixelShuffle output head that 2× upsamples) and
emits its output in the camera's native RGB space — the ColorMatrix is not
baked into the graph, so consumers must apply it after inference to reach
linear Rec.2020. The linear variant is a pure 3→3 denoiser, in and out of
linear Rec.2020. Both variants output at an arbitrary learned gain and
require a scalar gain-match against the input mean before use.

## Source

- Repository: https://github.com/trougnouf/rawnind_jddc
- Paper: [Learning Joint Denoising, Demosaicing, and Compression from the Raw Natural Image Noise Dataset](https://arxiv.org/abs/2501.08924) (Brummer & De Vleeschouwer, 2025)
- License: GPL-3.0

## Architecture

UtNet2 — 4-pool U-Net encoder-decoder (input H,W must be divisible by 16):

- `funit=32`, activation `LeakyReLU` (package default for both variants)
- Bayer output head:  `Conv2d(32 → 12, 1×1) → PixelShuffle(2)` (4 → 3 ch at 2× spatial)
- Linear output head: `Conv2d(32 → 3, 1×1)` (3 → 3 ch, same spatial)

## Checkpoints

- Bayer:  `DenoiserTrainingBayerToProfiledRGB_4ch_2024-02-21-bayer_ms-ssim_mgout_notrans_valeither_-4` (iter 4350000)
- Linear: `DenoiserTrainingProfiledRGBToProfiledRGB_3ch_2024-10-09-prgb_ms-ssim_mgout_notrans_valeither_-1` (iter 3900000)

Both are the canonical base variants from the `graph_denoise_models_definitions.yaml`
config map (`in_channels: 4` and `in_channels: 3`, no other options set). Training
used `match_gain: output` — the raw network outputs are at an arbitrary learned
scale; the demo rescales against the input mean at inference.

## ONNX Models

| File                | Input                              | Output                                |
|---------------------|------------------------------------|---------------------------------------|
| `model_bayer.onnx`  | `input` — float32 [1, 4, 512, 512] | `output` — float32 [1, 3, 1024, 1024] |
| `model_linear.onnx` | `input` — float32 [1, 3, 512, 512] | `output` — float32 [1, 3, 512, 512]   |

Both variants are exported with static input shapes baked at 512×512 (opset
20, FP32). Callers must tile at exactly 512×512 — darktable reads the
per-variant tile size from the manifest:

```yaml
attributes:
  input_sizes: [512]
  model_bayer:
    input_kind: bayer_v1
    bayer_orientation: force_rggb
    edge_pad: mirror_cropped
    wb_norm: none
    output_scale: match_gain
  model_linear:
    input_kind: linear_v1
    input_colorspace: lin_rec2020
    wb_norm: none
    output_scale: match_gain
    target_mean: null
```

The Bayer variant is pinned to the CoreML CPU compute units via the
top-level `cpu_only` block — its intermediate activations overflow FP16
on Apple's ANE / GPU and produce NaN/Inf output. The linear variant has
no such issue and runs on the user's configured EP unchanged:

```yaml
cpu_only:
  model_bayer: [coreml]
```

## Demo pipeline

`demo.py` auto-dispatches based on `rawpy.imread(image).raw_pattern.shape`:

- `(2, 2)` → Bayer variant:
  1. Normalise per-channel black level → white level, clip to [0, 1]
  2. Pack to [R, G1, G2, B] half-resolution tensor
  3. Crop to mod-16
  4. Inference → camRGB (arbitrary scale, 2× input spatial size)
  5. Gain-match to input mean
  6. camRGB → linear Rec.2020 via `inv(rgb_xyz_matrix[:3,:]) → XYZ → Rec.2020`
- anything else (X-Trans 6×6, Foveon, …) → Linear variant:
  1. `rawpy.postprocess` with linear Rec.2020 output, camera WB, no gamma
  2. Crop to mod-16
  3. Inference → lin-Rec.2020 (arbitrary scale)
  4. Gain-match to input mean

Output is a 16-bit linear Rec.2020 TIFF (or `.exr` if the output path has that
suffix). Linear Rec.2020 looks very dark in typical image viewers — open in
darktable / rawtherapee / a PQ-aware viewer.

## Selection Criteria

| Property                 | Value                                                                                                     |
|--------------------------|-----------------------------------------------------------------------------------------------------------|
| Model license            | GPL-3.0                                                                                                   |
| OSAID v1.0               | Open Source AI                                                                                            |
| MOF                      | Class I (Open Science)                                                                                    |
| Training data license    | CC BY 4.0 / CC0 (per-image, Wikimedia Commons)                                                            |
| Training data provenance | [RawNIND](https://dataverse.uclouvain.be/dataverse/rawnind) – real-world raw noise/clean pairs captured by authors |
| Training code            | [GPL-3.0](https://github.com/trougnouf/rawnind_jddc)                                                      |
| Known limitations        | Authors flag the code as academic state; Bayer-only 2x output upsample baked into the Bayer variant        |
| Published research       | [arXiv:2501.08924](https://arxiv.org/abs/2501.08924)                                                      |
| Inference                | Local only, no cloud dependencies                                                                         |
| Scope                    | Raw and linear-RGB image denoising                                                                        |
| Reproducibility          | Full pipeline (setup, convert, clean, demo)                                                               |
