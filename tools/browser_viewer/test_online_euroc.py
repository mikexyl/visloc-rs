#!/usr/bin/env python3
"""Exercise online estimation and browser publishing using paced EuRoC sensor input."""
import argparse
import csv
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import time
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts/realsense'))
from browser_stream import BrowserStream


def manifest(path):
    with path.open() as file:
        return [r for r in csv.reader(file) if r and not r[0].startswith('#')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--euroc-dir', type=Path, default=Path('/data/euroc/MH_01_easy'))
    parser.add_argument('--url', default='http://127.0.0.1:8090')
    parser.add_argument('--frames', type=int, default=1200)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    mav = args.euroc_dir/'mav0'
    cameras = [manifest(mav/f'cam{i}/data.csv') for i in range(2)]
    imu = [[int(r[0])]+[float(x) for x in r[1:]] for r in manifest(mav/'imu0/data.csv')]
    calibration = root/'configs/basalt/variants/official_euroc_ds'
    process = subprocess.Popen([str(root/'target/release/examples/basalt_stream_vio'),
                                str(calibration/'euroc_ds_calib.json'),str(calibration/'euroc_config.json'),
                                str(args.output)], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    viewer = BrowserStream(args.url)
    cursor = 0
    start = time.monotonic()
    first = int(cameras[0][0][0])
    previous = first-100_000_000
    try:
        for i, row in enumerate(cameras[0][:args.frames]):
            t = int(row[0])
            assert int(cameras[1][i][0]) == t
            samples = []
            while cursor < len(imu) and imu[cursor][0] <= t:
                if imu[cursor][0] > previous:
                    samples.append(imu[cursor])
                cursor += 1
            raw, jpegs = [], []
            for camera in range(2):
                with Image.open(mav/f'cam{camera}/data'/cameras[camera][i][1].strip()) as source:
                    image = source.convert('L'); raw.append(image.tobytes())
                    out = io.BytesIO(); image.save(out, format='JPEG', quality=80); jpegs.append(out.getvalue())
                    width, height = image.size
            header = json.dumps(dict(timestamp_ns=t,width=width,height=height,imu=samples,
                                     initialization_imu=imu[cursor] if i==0 else None)).encode()
            time.sleep(max(0, start+(t-first)*1e-9-time.monotonic()))
            process.stdin.write(struct.pack('<I', len(header))+header+b''.join(raw)); process.stdin.flush()
            line = process.stdout.readline()
            if not line:
                raise RuntimeError(f'VIO exited: {process.poll()}')
            result = json.loads(line)
            viewer.publish(result, jpegs)
            if i%100==0:
                print(json.dumps(result), flush=True)
            previous = t
    finally:
        process.stdin.close()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait()
        viewer.close()
    if process.returncode:
        raise RuntimeError(f'VIO failed: {process.returncode}')


if __name__ == '__main__':
    main()
