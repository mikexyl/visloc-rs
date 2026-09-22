#!/usr/bin/env python3
"""Compare native Rust TensorRT I/O against CPU ONNX Runtime (validation only).

Exercises normalized retrieval outputs, dense XFeat outputs, and data-dependent
LighterGlue outputs, including empty matches and repeated output-buffer reuse.
No Python/ORT inference is used by the online pipeline.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import onnxruntime as ort


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, default=Path('target/loop_models'))
    p.add_argument('--runner', type=Path, default=Path('target/release/examples/infer_fixtures'))
    a = p.parse_args()
    manifest = json.loads((a.bundle / 'manifest.json').read_text())
    root = (a.bundle / 'validation').resolve()
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(47)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    report = {}
    for name, model in manifest['models'].items():
        for kind in ('onnx', 'engine'):
            assert hashlib.sha256(Path(model[kind]).read_bytes()).hexdigest() == model[f'{kind}_sha256'], f'{name}: {kind} changed since build'
        session = ort.InferenceSession(model['onnx'], options, providers=['CPUExecutionProvider'])
        if name == 'jist':
            pixels = rng.uniform(0, 1, (1, 5, 3, 288, 512)).astype(np.float32)
            pixels = (pixels - np.array([.485, .456, .406], np.float32)[None, None, :, None, None]) / np.array([.229, .224, .225], np.float32)[None, None, :, None, None]
            cases = [{'input': pixels}]
        elif name == 'xfeat':
            cases = [{'input': rng.uniform(0, 1, (1, 3, 224, 320)).astype(np.float32)}]
        else:
            cases = []
            k = model.get('keypoints', 128)
            for n, m, shared in [(k, k, True), (k, k, False), (k, k, True)]:
                size = max(n, m)
                desc = rng.normal(size=(size, 64)).astype(np.float32)
                desc /= np.linalg.norm(desc, axis=1, keepdims=True)
                pixels = rng.uniform([0, 0], [800, 550], size=(size, 2)).astype(np.float32)
                other = desc[:m].copy() if shared else rng.normal(size=(m, 64)).astype(np.float32)
                other /= np.linalg.norm(other, axis=1, keepdims=True)
                cases.append({'mkpts0': pixels[None, :n], 'feats0': desc[None, :n],
                              'image0_size': np.array([800, 550], np.float32),
                              'mkpts1': pixels[None, :m], 'feats1': other[None],
                              'image1_size': np.array([800, 550], np.float32)})
        fixtures = []
        references = []
        for i, case in enumerate(cases):
            inputs = {}
            for key, value in case.items():
                path = root / f'{name}_{i}_{key}.bin'
                value.tofile(path)
                inputs[key] = {'path': str(path), 'shape': list(value.shape)}
            fixtures.append(inputs)
            references.append(dict(zip([o.name for o in session.get_outputs()], session.run(None, case))))
        fixture_path = root / f'{name}.json'
        fixture_path.write_text(json.dumps(fixtures))
        output = root / name
        subprocess.run([str(a.runner.resolve()), model['engine'], str(fixture_path), str(output)], check=True)
        outputs = json.loads((output / 'outputs.json').read_text())
        comparisons = []
        for reference, result in zip(references, outputs):
            measured = {key: np.fromfile(value['path'], dtype={'F32': np.float32, 'I64': np.int64}[value['dtype']])
                        .reshape(value['shape']) for key, value in result['outputs'].items()}
            comparison = {'mean_ms': result['mean_ms']}
            if name == 'lighterglue':
                expected = {tuple(pair): float(score) for pair, score in zip(reference['matches'], reference['scores'])}
                actual = {tuple(pair): float(score) for pair, score in zip(measured['matches'], measured['scores'])}
                common = set(expected) & set(actual)
                agreement = len(common) / max(1, len(set(expected) | set(actual))) if expected or actual else 1.0
                score_error = max((abs(expected[k]-actual[k]) for k in common), default=0.0)
                comparison.update(matches_onnx=len(expected), matches_trt=len(actual),
                                  pair_agreement=agreement, max_score_error=score_error)
                assert agreement == 1.0 and score_error < .001, comparison
            else:
                for key, expected in reference.items():
                    actual = measured[key]
                    assert np.isfinite(actual).all(), (name, key)
                    np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=2e-3, err_msg=f'{name}/{key}')
                    comparison[key] = {'max_absolute_error': float(np.max(np.abs(actual-expected)))}
            comparisons.append(comparison)
        report[name] = comparisons
        print(name, json.dumps(comparisons), flush=True)
    (root / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
