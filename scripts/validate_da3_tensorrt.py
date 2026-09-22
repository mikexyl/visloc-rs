#!/usr/bin/env python3
"""Compare camera-conditioned TensorRT fixtures against unpatched upstream DA3.

PyTorch is used only offline. Optionally rerun a rebuilt engine on recorded
inputs; otherwise compare the original recorded inference outputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def load_fixture(path):
    with np.load(path, allow_pickle=False) as fixture:
        if 'images' in fixture:
            inputs = {key: fixture[key] for key in ('images', 'input_extrinsics', 'input_intrinsics')}
            outputs = {key: fixture[key] for key in ('depth', 'depth_conf')}
        else:
            # Accepted archives contain the exact resized images/camera inputs.
            rgb = np.repeat(fixture['gray'][:, None].astype(np.float32)/255., 3, axis=1)
            images = ((rgb-np.array([.485,.456,.406], np.float32)[None,:,None,None])
                      / np.array([.229,.224,.225], np.float32)[None,:,None,None])[None]
            inputs = dict(images=images, input_extrinsics=fixture['input_world_to_camera'][None],
                          input_intrinsics=fixture['intrinsics'][None])
            outputs = dict(depth=fixture['raw_depth'], depth_conf=fixture['confidence'])
        return inputs, outputs, fixture['frame_ids'].tolist()


def compare(path, model, torch, session=None):
    inputs, outputs, frame_ids = load_fixture(path)
    if session:
        outputs = session.run(inputs)
    images = torch.from_numpy(inputs['images']).to('cuda')
    extrinsics = torch.from_numpy(inputs['input_extrinsics']).to('cuda')
    intrinsics = torch.from_numpy(inputs['input_intrinsics']).to('cuda')
    # TensorRT performs this official normalization inside the graph.
    normalized = model._normalize_extrinsics(extrinsics.clone())
    original = torch.cuda.is_bf16_supported
    try:
        torch.cuda.is_bf16_supported = lambda *a, **k: False
        with torch.no_grad():
            reference = model.forward(images, extrinsics=normalized, intrinsics=intrinsics,
                                      export_feat_layers=[], infer_gs=False,
                                      use_ray_pose=False, ref_view_strategy='middle')
    finally:
        torch.cuda.is_bf16_supported = original
    report = dict(fixture=str(path.resolve()), frame_ids=frame_ids, outputs={})
    passed = True
    for key in ('depth', 'depth_conf'):
        actual = outputs[key].astype(np.float32)
        expected = reference[key].detach().float().cpu().numpy().reshape(actual.shape)
        finite = bool(np.isfinite(expected).all() and np.isfinite(actual).all())
        if finite:
            relative = np.abs(actual-expected) / np.maximum(np.abs(expected), 1e-6)
            stats = dict(all_finite=True, median_relative=float(np.median(relative)),
                         p99_relative=float(np.quantile(relative, .99)),
                         max_relative=float(np.max(relative)),
                         mean_absolute=float(np.mean(np.abs(actual-expected))))
            stats['passed'] = stats['median_relative'] <= .02 and stats['p99_relative'] <= .10
        else:
            stats = dict(all_finite=False, passed=False)
        passed &= stats['passed']
        report['outputs'][key] = stats
    report['passed'] = passed
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--da3-repo', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--fixture', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--engine', type=Path, help='Optional rebuilt TensorRT engine to validate')
    parser.add_argument('--library', type=Path, help='Native TensorRT bridge; required with --engine')
    args = parser.parse_args()
    if bool(args.engine) != bool(args.library):
        parser.error('--engine and --library must be supplied together')
    sys.path.insert(0, str(args.da3_repo.resolve() / 'src'))
    import torch
    from depth_anything_3.api import DepthAnything3
    from tensorrt_session import Session

    model = DepthAnything3.from_pretrained(args.model).to('cuda').eval()
    session = Session(args.engine, args.library) if args.engine else None
    try:
        fixtures = [compare(path, model, torch, session) for path in args.fixture]
    finally:
        if session:
            session.close()
    report = dict(reference='unpatched upstream DA3, official pose normalization, FP16 autocast',
                  torch_version=torch.__version__, fixtures=fixtures,
                  passed=all(f['passed'] for f in fixtures))
    if args.engine:
        report.update(engine=str(args.engine.resolve()),
                      engine_sha256=hashlib.sha256(args.engine.read_bytes()).hexdigest())
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2))
    if not report['passed']:
        raise SystemExit('DA3 TensorRT parity failed')


if __name__ == '__main__':
    main()
