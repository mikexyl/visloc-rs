#!/usr/bin/env python3
"""Summarize Rust parity CSV and compare the small corpus's retrieval similarities."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("directory", type=Path)
args = parser.parse_args()
directory = args.directory
with (directory / "parity.csv").open() as stream:
    rows = list(csv.DictReader(stream))
if len(rows) < 2:
    raise ValueError("Need at least two descriptors")
reference = np.stack([np.fromfile(directory / f"reference_{i}.bin", dtype="<f4") for i in range(len(rows))]).astype(np.float64)
actual = np.stack([np.fromfile(directory / f"tensorrt_{i}.bin", dtype="<f4") for i in range(len(rows))]).astype(np.float64)
if reference.shape != actual.shape or reference.shape[1] != 8448:
    raise ValueError("Invalid descriptor shapes")
if not np.isfinite(reference).all() or not np.isfinite(actual).all():
    raise ValueError("Nonfinite descriptors")
ref_norms = np.linalg.norm(reference, axis=1)
trt_norms = np.linalg.norm(actual, axis=1)
if (ref_norms == 0).any() or (trt_norms == 0).any():
    raise ValueError("Zero descriptors")
cosines = (reference * actual).sum(axis=1) / (ref_norms * trt_norms)
ref_unit = reference / ref_norms[:, None]
trt_unit = actual / trt_norms[:, None]
ref_similarity = ref_unit @ ref_unit.T
trt_similarity = trt_unit @ trt_unit.T
pairwise_error = float(np.max(np.abs(ref_similarity - trt_similarity)))
np.fill_diagonal(ref_similarity, -np.inf)
np.fill_diagonal(trt_similarity, -np.inf)
reference_nearest = np.argmax(ref_similarity, axis=1)
actual_nearest = np.argmax(trt_similarity, axis=1)
max_abs_error = float(np.max(np.abs(reference - actual)))
max_norm_error = float(np.max(np.abs(trt_norms - 1)))
summary = {
    "images": len(rows), "descriptor_dimensions": 8448,
    "minimum_descriptor_cosine": float(cosines.min()),
    "max_absolute_descriptor_error": max_abs_error,
    "descriptor_rmse": float(np.sqrt(np.mean((reference - actual)**2))),
    "maximum_output_norm_error": max_norm_error,
    "max_pairwise_cosine_error": pairwise_error,
    "nearest_neighbor_agreement_excluding_self": int(np.sum(reference_nearest == actual_nearest)),
    "reference_nearest_neighbors": reference_nearest.tolist(),
    "tensorrt_nearest_neighbors": actual_nearest.tolist(),
    "median_of_per_image_median_host_to_host_ms": float(np.median([float(r["host_to_host_ms"]) for r in rows])),
    "thresholds": {"min_cosine": 0.9999, "max_absolute_error": 0.001, "max_norm_error": 0.0001},
    "passed": bool(cosines.min() >= 0.9999 and max_abs_error <= 0.001 and max_norm_error <= 0.0001),
    "scope": "Numerical parity on eight sampled frames, not a VPR recall benchmark. Timing includes host/device copies and synchronization, excludes preprocessing and engine loading.",
}
(directory / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
print(json.dumps(summary, indent=2))
if not summary["passed"]:
    raise SystemExit(1)
