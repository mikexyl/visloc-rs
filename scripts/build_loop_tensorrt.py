#!/usr/bin/env python3
"""Build the JIST/XFeat/LighterGlue engine bundle on the deployment GPU.

Requires the XFeat-trained lg_320x224_dyn export, not SuperPoint LightGlue.
All models use FP32 without TF32. LighterGlue uses a fixed keypoint profile,
following the deployment approach in Derkai52/XFeat-Lightglue-TRT. This avoids
the dynamic FP32 Myelin compiler failure without losing numerical accuracy.
TensorRT plans are GPU/SDK specific and are deliberately not checked into git.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('target/loop_models'))
    p.add_argument('--trtexec', default='/usr/src/tensorrt/bin/trtexec')
    p.add_argument('--only', choices=('jist', 'xfeat', 'lighterglue'))
    p.add_argument('--keypoints', type=int, default=128, help='Fixed LighterGlue input count (16..1024)')
    a = p.parse_args()
    if not 16 <= a.keypoints <= 1024:
        p.error('--keypoints must be in 16..1024')
    a.output.mkdir(parents=True, exist_ok=True)
    models = {'jist': 'JIST_r18_512_seqgem_frames.onnx',
              'xfeat': 'xfeat_320x224.onnx', 'lighterglue': 'lg_320x224_dyn.onnx'}
    manifest_path = a.output / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if a.only and manifest_path.exists() else {'models': {}}
    for name, filename in models.items():
        if a.only and name != a.only:
            continue
        source = (a.model_dir / filename).resolve(strict=True)
        plan = (a.output / f'{name}.engine').resolve()
        temporary = plan.with_suffix('.building.engine')
        cmd = [a.trtexec, f'--onnx={source}', f'--saveEngine={temporary}',
               '--skipInference', '--noTF32', '--memPoolSize=workspace:2048M',
               '--builderOptimizationLevel=3']
        if name == 'lighterglue':
            for flag in ('minShapes', 'optShapes', 'maxShapes'):
                count = a.keypoints
                cmd.append(f'--{flag}=mkpts0:1x{count}x2,feats0:1x{count}x64,'
                           f'mkpts1:1x{count}x2,feats1:1x{count}x64')
        print(f'Building {name}: {plan}', flush=True)
        with (a.output / f'{name}.build.log').open('w') as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        temporary.replace(plan)
        manifest['models'][name] = {
            'onnx': str(source), 'onnx_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'engine': str(plan), 'engine_sha256': hashlib.sha256(plan.read_bytes()).hexdigest(),
            'precision': 'fp32', 'command': cmd}
        if name == 'lighterglue':
            manifest['models'][name]['keypoints'] = a.keypoints
        (a.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if set(manifest['models']) != set(models):
        print('Partial bundle built; build the other models before running VIO.', flush=True)
        return
    config = {'jist_engine': manifest['models']['jist']['engine'],
              'xfeat_engine': manifest['models']['xfeat']['engine'],
              'lighterglue_engine': manifest['models']['lighterglue']['engine'],
              'matcher_keypoints': manifest['models']['lighterglue']['keypoints'],
              'min_similarity': 0.8, 'covisibility_min_shared': 15, 'covisibility_hops': 2}
    (a.output / 'loop_config.json').write_text(json.dumps(config, indent=2) + '\n')
    print(f'Bundle ready: {a.output / "loop_config.json"}', flush=True)


if __name__ == '__main__':
    main()
