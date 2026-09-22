#!/usr/bin/env python3
"""Build the existing visloc native TensorRT bridge as a NumPy-loadable library."""
import argparse
import os
from pathlib import Path
import platform
import subprocess


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=repo / '.runtime/libvisloc_tensorrt.so')
    args = parser.parse_args()
    cuda = Path(os.environ.get('CUDA_HOME', '/usr/local/cuda'))
    include = os.environ.get('TENSORRT_INCLUDE_DIR', f'/usr/include/{platform.machine()}-linux-gnu')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = ['c++', '-std=c++17', '-O2', '-shared', '-fPIC',
               '-I' + include, '-I' + str(cuda / 'include'),
               str(repo / 'crates/tensorrt-runtime/native/bridge.cpp'),
               '-L' + str(cuda / 'lib64'), '-Wl,-rpath,' + str(cuda / 'lib64'),
               '-lnvinfer', '-lnvinfer_plugin', '-lcudart', '-o', str(args.output)]
    if 'TENSORRT_LIB_DIR' in os.environ:
        command[1:1] = ['-L' + os.environ['TENSORRT_LIB_DIR'], '-Wl,-rpath,' + os.environ['TENSORRT_LIB_DIR']]
    subprocess.run(command, check=True)
    print(args.output.resolve())


if __name__ == '__main__':
    main()
