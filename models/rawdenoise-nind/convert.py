"""Export RawNIND UtNet2 raw denoiser to ONNX.

Uses UtNet2 from the cloned rawnind_jddc repository:
https://github.com/trougnouf/rawnind_jddc

The bayer2prgb variant takes a 4-channel packed Bayer tensor and produces a
3-channel linear Rec.2020 RGB image at the same spatial resolution as the
packed Bayer input (i.e. half the sensor resolution on each axis).
"""

import argparse
import importlib.util
import os
import sys
import types

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DTAI_ROOT = os.environ.get("DTAI_ROOT", os.path.join(SCRIPT_DIR, "../.."))
_RAWNIND_SRC = os.path.join(DTAI_ROOT, "vendor", "rawnind_jddc", "src")


def _load_utnet2():
    """Load UtNet2 from raw_denoiser.py without triggering rawnind's package __init__.

    The upstream package's __init__.py chains in tools/libs/models, which pulls
    psutil, configargparse and a long tail of research-pipeline deps we don't
    need for ONNX export. UtNet2 itself only depends on torch; the only sibling
    import it makes (rawnind.libs.rawproc) is used exclusively by the
    Passthrough class, so we stub it out.
    """
    # Stub the parent packages + the one real sibling module UtNet2's file imports.
    for name in ("rawnind", "rawnind.libs", "rawnind.libs.rawproc"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

    path = os.path.join(_RAWNIND_SRC, "rawnind", "models", "raw_denoiser.py")
    spec = importlib.util.spec_from_file_location("_rawnind_raw_denoiser", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.UtNet2


UtNet2 = _load_utnet2()

try:
    import onnxconverter_common
    HAS_ONNX_CONVERTER = True
except ImportError:
    HAS_ONNX_CONVERTER = False


def load_model(checkpoint_path, in_channels=4, funit=32, activation="PReLU",
               preupsample=False):
    model = UtNet2(
        in_channels=in_channels,
        funit=funit,
        activation=activation,
        preupsample=preupsample,
    )
    # weights_only=False: the rawnind_jddc checkpoints are plain pickle dumps
    # from torch.save on the nn.Module / training state, not pure tensor dicts.
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Full-model pickle (author saved the nn.Module itself)
    if isinstance(loaded, torch.nn.Module):
        state_dict = loaded.state_dict()
    elif isinstance(loaded, dict):
        state_dict = loaded
        for key in ("state_dict", "model_state_dict", "params", "params_ema", "model", "generator"):
            if key in state_dict:
                state_dict = state_dict[key]
                if isinstance(state_dict, torch.nn.Module):
                    state_dict = state_dict.state_dict()
                break
    else:
        raise TypeError(f"Unexpected checkpoint type: {type(loaded)}")

    # Strip "module." prefix if present (DataParallel)
    cleaned = {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(cleaned, strict=True)
    model.eval()
    return model


def export_to_onnx(model, output_path, in_channels=4,
                   input_height=256, input_width=256,
                   dynamic_shapes=True, opset_version=17, fp16=False):
    dummy_input = torch.randn(1, in_channels, input_height, input_width)

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


def convert(checkpoint, output="model.onnx", in_channels=4, funit=32,
            activation="PReLU", preupsample=False,
            height=512, width=512, dynamic_shapes=True, opset=20, fp16=False,
            static=False):
    """Entry point for programmatic conversion."""
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    print("Loading RawNIND UtNet2 model...")
    model = load_model(checkpoint, in_channels=in_channels, funit=funit,
                       activation=activation, preupsample=preupsample)

    print("Exporting to ONNX...")
    export_to_onnx(model, output, in_channels=in_channels,
                   input_height=height, input_width=width,
                   dynamic_shapes=dynamic_shapes and not static,
                   opset_version=opset, fp16=fp16)


def main():
    parser = argparse.ArgumentParser(description="Export RawNIND UtNet2 to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="model.onnx")
    parser.add_argument("--in-channels", type=int, default=4)
    parser.add_argument("--funit", type=int, default=32)
    parser.add_argument("--activation", type=str, default="PReLU")
    parser.add_argument("--preupsample", action="store_true")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--static", action="store_true",
                        help="bake input height/width into the graph "
                             "(disables dynamic shape axes)")
    args = parser.parse_args()

    convert(args.checkpoint, args.output,
            in_channels=args.in_channels, funit=args.funit,
            activation=args.activation, preupsample=args.preupsample,
            height=args.height, width=args.width,
            opset=args.opset, fp16=args.fp16, static=args.static)


if __name__ == "__main__":
    main()
