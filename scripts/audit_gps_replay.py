#!/usr/bin/env python3
"""Verify matched replay input content/order from the immutable shared cache.

This reconstructs the deterministic publisher's image/IMU order and hashes it;
it is not a packet capture. Actual delivery is checked using zero drop counters
and bit-exact raw pose parity after the completed replays.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import numpy as np
from replay_sensor_source import open_source


def digest_stream(robot):
    source=open_source(robot);h=hashlib.sha256();index=0;images=0
    try:
        for timestamp,left,right in source.frames:
            end=min(len(source.imu),int(np.searchsorted(source.times,timestamp,side='left'))+1)
            for sample in source.imu[index:end]:h.update(struct.pack('<cq6d',b'I',int(sample[0])+robot['offset_ns'],*sample[1:7]))
            index=end
            for tag,raw in [(b'L',source.image(left)),(b'R',source.right_image(right))]:
                h.update(struct.pack('<cqIII',tag,int(timestamp)+robot['offset_ns'],raw.width,raw.height,raw.step));h.update(raw.encoding.encode());h.update(bytes([raw.is_bigendian]));h.update(np.asarray(raw.data,dtype=np.uint8).tobytes());images+=1
        return dict(sha256=h.hexdigest(),stereo_pairs=len(source.frames),images=images,imu_samples=index,first_timestamp_ns=source.frames[0][0]+robot['offset_ns'],last_timestamp_ns=source.frames[-1][0]+robot['offset_ns'])
    finally:source.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('control',type=Path);p.add_argument('gps',type=Path);p.add_argument('output',type=Path);p.add_argument('--inputs-only',action='store_true');a=p.parse_args()
    runs=[a.control.resolve(),a.gps.resolve()];missions=[json.loads((r/'mission.json').read_text()) for r in runs]
    robots=[m['robots'][0] for m in missions];configs=[json.loads(Path(r['config']).read_text()) for r in robots]
    for field in ('rate','epoch_ns'):assert missions[0][field]==missions[1][field]
    assert [{k:v for k,v in r.items() if k not in ('config','gps_records')} for r in robots][0]=={k:v for k,v in robots[1].items() if k not in ('config','gps_records')}
    assert {k:v for k,v in configs[0].items() if k not in ('output','gps')}=={k:v for k,v in configs[1].items() if k not in ('output','gps')}
    digests=[digest_stream(r) for r in robots];assert digests[0]==digests[1]
    report=dict(inputs_equal=True,input_streams=digests,note=__doc__.strip(),calibration_sha256=hashlib.sha256(Path(configs[0]['calibration']).read_bytes()).hexdigest())
    if not a.inputs_only:
        for run in runs:
            summary=json.loads((run/'replay_summary.json').read_text());robot=next(iter(summary['robots'].values()));assert robot['frames']==robot['frames_ingested']==digests[0]['stereo_pairs'];assert robot['dropped_images']==robot['dropped_keyframes']==0
            traffic=json.loads((run/'robots'/robots[0]['robot']/'communication.json').read_text());assert traffic['sensor_ingress_drops']==0
        parity=a.output.with_name(a.output.stem+'_poses.json')
        subprocess.run([sys.executable,str(Path(__file__).with_name('test_multi_robot_raw_parity.py')),*[str(r/'robots'/robots[0]['robot']/'trajectory.csv') for r in runs],'--output',str(parity)],check=True)
        report['raw_pose_parity']=json.loads(parity.read_text());report['no_additional_sensor_drops']=True
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
