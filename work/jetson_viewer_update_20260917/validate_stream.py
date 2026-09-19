import csv
import json
import math
from pathlib import Path
import struct
import subprocess
import time
import cv2

mav = Path('/dataset/mav0')
def rows(path):
    with path.open() as f:
        return [r for r in csv.reader(f) if r and not r[0].startswith('#')]
cameras = [rows(mav/f'cam{i}/data.csv') for i in range(2)]
imu = [[int(r[0])]+[float(v) for v in r[1:]] for r in rows(mav/'imu0/data.csv')]
calib = Path('/opt/visloc-rs/configs/basalt/variants/official_euroc_ds')
process = subprocess.Popen(['/usr/local/bin/basalt_stream_vio',str(calib/'euroc_ds_calib.json'),str(calib/'euroc_config.json'),'/output/native'],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
previous=int(cameras[0][0][0])-100_000_000
cursor=0
counts=[]
started=time.monotonic()
try:
    for i,row in enumerate(cameras[0][:80]):
        timestamp=int(row[0]); assert int(cameras[1][i][0])==timestamp
        samples=[]
        while cursor<len(imu) and imu[cursor][0]<=timestamp:
            if imu[cursor][0]>previous: samples.append(imu[cursor])
            cursor+=1
        images=[cv2.imread(str(mav/f'cam{camera}/data'/cameras[camera][i][1].strip()),cv2.IMREAD_GRAYSCALE) for camera in range(2)]
        assert all(image is not None for image in images)
        height,width=images[0].shape
        metadata=json.dumps(dict(timestamp_ns=timestamp,width=width,height=height,imu=samples,initialization_imu=imu[cursor] if i==0 else None)).encode()
        process.stdin.write(struct.pack('<I',len(metadata))+metadata+b''.join(image.tobytes() for image in images));process.stdin.flush()
        line=process.stdout.readline();assert line,'Estimator exited before completing frames'
        result=json.loads(line)
        assert result['frame_id']==i
        assert all(math.isfinite(v) for v in result['position']+result['quaternion_xyzw'])
        points=result['map_points'];assert len(points)<=3000
        assert all(len(p)==4 and isinstance(p[0],int) and all(math.isfinite(v) for v in p[1:]) for p in points)
        counts.append(len(points));previous=timestamp
finally:
    process.stdin.close()
    try: process.wait(timeout=20)
    except subprocess.TimeoutExpired: process.kill();process.wait()
assert process.returncode==0 and len(counts)==80 and max(counts)>0
summary=dict(passed=True,frames=len(counts),min_active_map_points=min(counts),max_active_map_points=max(counts),wall_seconds=time.monotonic()-started)
Path('/output/validated.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
