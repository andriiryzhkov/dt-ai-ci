"""Export NIND UNet denoiser to ONNX format.

Uses UNet from the cloned nind-denoise repository:
https://github.com/trougnouf/nind-denoise

By default exports with dynamic spatial axes. Pass --static (or
`static: true` from model.yaml) to bake the input height/width into the
graph so JIT-compiling EPs (CoreML, MIGraphX) only pay the compile cost
once.
"""

import argparse
import os
import sys

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DTAI_ROOT = os.environ.get("DTAI_ROOT", os.path.join(SCRIPT_DIR, "../.."))
sys.path.insert(0, os.path.join(DTAI_ROOT, "vendor", "nind-denoise", "src"))

from nind_denoise.networks.ThirdPartyNets import UNet

try:
    import onnxconverter_common
    HAS_ONNX_CONVERTER = True
except ImportError:
    HAS_ONNX_CONVERTER = False


def load_model(checkpoint_path):
    model = UNet(n_channels=3, n_classes=3)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def export_to_onnx(model, output_path, input_height=512, input_width=512,
                   dynamic_shapes=True, opset_version=20, fp16=False):
    dummy_input = torch.randn(1, 3, input_height, input_width)

    dynamic_axes = None
    if dynamic_shapes:
        dynamic_axes = {
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size", 2: "height", 3: "width"},
        }

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )
    print(f"Model exported to {output_path}")

    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verification passed!")

    try:
        import onnxsim
        print("Simplifying model...")
        onnx_model, ok = onnxsim.simplify(onnx_model)
        if ok:
            onnx.save(onnx_model, output_path)
            print("Model simplified successfully")
        else:
            print("Warning: simplification failed, using unsimplified model")
    except ImportError:
        print("onnx-simplifier not installed, skipping.")

    if fp16:
        if not HAS_ONNX_CONVERTER:
            print("Warning: onnxconverter-common not installed. Skipping FP16 conversion.")
            return
        print("Converting to FP16...")
        from onnxconverter_common import float16
        fp16_model = float16.convert_float_to_float16(onnx_model)
        onnx.save(fp16_model, output_path)
        print(f"FP16 model saved to {output_path}")


def convert(checkpoint, output="model.onnx", height=512, width=512,
            dynamic_shapes=True, opset=20, fp16=False, static=False):
    """Entry point for programmatic conversion."""
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    print("Loading NIND UNet model...")
    model = load_model(checkpoint)

    print("Exporting to ONNX...")
    export_to_onnx(model, output, input_height=height, input_width=width,
                   dynamic_shapes=dynamic_shapes and not static,
                   opset_version=opset, fp16=fp16)


def main():
    parser = argparse.ArgumentParser(description="Export NIND UNet to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="model.onnx")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--opset", type=int, default=20)
    parser.add_argument("--fp16", action="store_true",
                        help="convert weights to FP16 after export (default: FP32)")
    parser.add_argument("--static", action="store_true",
                        help="bake input height/width into the graph "
                             "(disables dynamic shape axes)")
    args = parser.parse_args()

    convert(args.checkpoint, args.output,
            height=args.height, width=args.width,
            opset=args.opset, fp16=args.fp16, static=args.static)


if __name__ == "__main__":
    main()
