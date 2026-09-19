#!/usr/bin/env python3
"""Export an existing MegaLoc checkout/weights and a reproducible parity corpus."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from safetensors.torch import load_file


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True, help="Directory containing image frames")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(args.repo.resolve()))
    from megaloc_model import MegaLoc
    model = MegaLoc().eval()
    model.load_state_dict(load_file(str(args.weights)), strict=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize([322, 322], antialias=True),
    ])
    paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if len(paths) < args.count or args.count < 2:
        raise ValueError("Need at least count images, with count >= 2")
    paths = [paths[i] for i in np.linspace(0, len(paths)-1, args.count, dtype=int)]
    inputs = [transform(Image.open(p).convert("RGB")).unsqueeze(0) for p in paths]
    print("Loaded model and preprocessed images", flush=True)
    model.cuda()
    with torch.inference_mode():
        for _ in range(3):
            model(inputs[0].cuda())
        torch.cuda.synchronize()
        refs, times = [], []
        for i, x in enumerate(inputs):
            start = time.perf_counter()
            y = model(x.cuda()).cpu().numpy()
            times.append((time.perf_counter()-start)*1000)
            if y.shape != (1, 8448) or not np.isfinite(y).all():
                raise ValueError(f"Invalid descriptor {y.shape}")
            x.numpy().astype("<f4").tofile(args.out / f"input_{i}.bin")
            y.astype("<f4").tofile(args.out / f"reference_{i}.bin")
            refs.append(y)
    print("Saved PyTorch reference descriptors", flush=True)
    model.cpu()
    torch.cuda.empty_cache()
    with torch.inference_mode():
        torch.onnx.export(model, inputs[0], str(args.out / "megaloc_fp32.onnx"),
                          input_names=["images"], output_names=["descriptor"],
                          opset_version=17, do_constant_folding=True)
    import onnx
    onnx.checker.check_model(str(args.out / "megaloc_fp32.onnx"))
    manifest = {
        "torch": torch.__version__, "onnx": onnx.__version__,
        "gpu": torch.cuda.get_device_name(0), "shape": [1, 3, 322, 322],
        "output_shape": [1, 8448], "model_source": str(args.repo.resolve()),
        "source_sha256": digest(args.repo / "megaloc_model.py"),
        "weights": str(args.weights.resolve()), "weights_sha256": digest(args.weights),
        "onnx_sha256": digest(args.out / "megaloc_fp32.onnx"),
        "images": [{"path": str(p), "sha256": digest(p)} for p in paths],
        "preprocessing": "RGB; ToTensor; ImageNet Normalize; Resize 322x322 bilinear antialias=True",
        "pytorch_host_to_host_ms": times,
        "reference_norms": np.linalg.norm(np.concatenate(refs), axis=1).tolist(),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
