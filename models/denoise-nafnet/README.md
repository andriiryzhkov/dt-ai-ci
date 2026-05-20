# NAFNet SIDD Width-32

Image denoiser trained on the SIDD (Smartphone Image Denoising Dataset).
Lightweight variant with width=32.

## Source

- Repository: https://github.com/megvii-research/NAFNet
- Paper: [Simple Baselines for Image Restoration](https://arxiv.org/abs/2204.04676) (ECCV 2022)
- License: MIT

## Architecture

NAFNet (Nonlinear Activation Free Network) — encoder-decoder with 4 stages,
channel widths [32, 64, 128, 256], 12 middle blocks.

## ONNX Model

| Property    | Value                                     |
|-------------|-------------------------------------------|
| File        | `model.onnx`                              |
| Input       | `input` — float32 [1, 3, 768, 768]        |
| Output      | `output` — float32 [1, 3, 768, 768]       |
| Resolution  | Static, baked at 768×768                  |
| Opset       | 20                                        |
| Normalize   | [0, 1] range (divide by 255)              |
| Tiling      | Yes (`attributes.input_sizes: [768]`)     |

## Notes

- Input and output are both RGB images in [0, 1] range.
- Output should be clipped to [0, 1] before converting back to uint8.
- Exported with FP32 precision (FP16 via `--fp16` is supported but off
  by default).
- The 768×768 input is baked into the graph so JIT-compiling EPs
  (CoreML, MIGraphX) only pay the compile cost once. Callers must
  tile at exactly this size; darktable reads `input_sizes` from the
  manifest and locks the runtime tile size accordingly.

## Selection Criteria

| Property                 | Value                                                                                   |
|--------------------------|-----------------------------------------------------------------------------------------|
| Model license            | MIT                                                                                     |
| OSAID v1.0               | Open Source AI                                                                          |
| MOF                      | Class I (Open Science)                                                                  |
| Training data license    | MIT                                                                                     |
| Training data provenance | [SIDD](https://abdokamel.github.io/sidd/) — 30K real smartphone noisy/clean pairs captured by authors (5 devices)            |
| Training code            | [MIT](https://github.com/megvii-research/NAFNet)                                       |
| Known limitations        | None — all components publicly available under permissive licenses                      |
| Published research       | [Simple Baselines for Image Restoration](https://arxiv.org/abs/2204.04676) (ECCV 2022) |
| Inference                | Local only, no cloud dependencies                                                       |
| Scope                    | Image denoising                                                                         |
| Reproducibility          | Full pipeline (setup, convert, clean, demo)                                             |
