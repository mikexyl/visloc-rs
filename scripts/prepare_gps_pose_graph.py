#!/usr/bin/env python3
"""Prepare GPS-only graph experiments without changing the VIO sensor cache.

NMEA measurement UTC is reconstructed from RMC dates and matching GGA epochs.
No PPK reference or trajectory is read here.
"""
import argparse
import bisect
import calendar
import collections
import datetime
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import numpy as np


def sentence(line):
    if not line.startswith('$') or '*' not in line:
        return None
    body, checksum = line[1:].strip().split('*', 1)
    value = 0
    for character in body:
        value ^= ord(character)
    if len(checksum) != 2 or value != int(checksum, 16):
        raise ValueError('NMEA checksum mismatch')
    return body.split(',')


def time_of_day_ns(text):
    if len(text)<6:
        raise ValueError('Malformed NMEA UTC time')
    hour,minute,second=int(text[:2]),int(text[2:4]),Decimal(text[4:])
    if not (0<=hour<24 and 0<=minute<60 and 0<=second<60):
        raise ValueError('Out-of-range NMEA UTC time (leap-second epochs require explicit handling)')
    return (hour*3600+minute*60)*10**9+int(second*10**9)


def utc_ns(date, tod):
    day = datetime.datetime.strptime(date, '%d%m%y')
    return calendar.timegm(day.timetuple())*10**9 + time_of_day_ns(tod)


def normalize(bag, robot):
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore
    fixes, gga, rmc = [], [], collections.defaultdict(list)
    with AnyReader([bag.resolve()], default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as reader:
        connections = [c for c in reader.connections if c.topic in ('/gps/fix', '/gps/nmea')]
        for connection, receipt, raw in reader.messages(connections=connections):
            message = reader.deserialize(raw, connection.msgtype)
            if connection.topic == '/gps/fix':
                fixes.append((receipt, message))
                continue
            for line in message.data.splitlines():
                fields = sentence(line)
                if not fields:
                    continue
                if fields[0].endswith('RMC') and fields[1] and fields[9]:
                    rmc[time_of_day_ns(fields[1])].append((receipt, utc_ns(fields[9], fields[1])))
                elif fields[0].endswith('GGA') and fields[1]:
                    gga.append((receipt, fields))
    times = [row[0] for row in gga]
    used = set()
    records, latencies, pairing = [], [], []
    for index, (receipt, message) in enumerate(fixes):
        where = bisect.bisect_left(times, receipt)
        choices = [i for i in range(max(0, where-2), min(len(gga), where+3)) if i not in used]
        if not choices:
            raise ValueError('Missing GGA for NavSatFix')
        chosen = min(choices, key=lambda i: abs(times[i]-receipt))
        if abs(times[chosen]-receipt) > 100_000_000:
            raise ValueError('Ambiguous GGA/NavSatFix receipt association')
        used.add(chosen)
        nmea_receipt, fields = gga[chosen]
        candidates = rmc[time_of_day_ns(fields[1])]
        if not candidates:
            raise ValueError('Missing RMC date for GGA epoch')
        rmc_receipt, measurement = min(candidates, key=lambda r: abs(r[0]-nmea_receipt))
        if abs(rmc_receipt-nmea_receipt) > 500_000_000:
            raise ValueError('RMC/GGA epoch association is too distant')
        quality = int(fields[6] or 0)
        hdop = float(fields[8]) if fields[8] else None
        lla = [float(message.latitude), float(message.longitude), float(message.altitude)]
        if not all(np.isfinite(lla[:2])):
            lla = None
        elif not np.isfinite(lla[2]):
            lla[2] = 0.  # Height is unused; preserve valid horizontal fixes.
        if quality and lla is not None:
            def degrees(value, sign):
                x = Decimal(value); d = int(x//100)
                return float(Decimal(d)+(x-100*d)/60) * (-1 if sign in ('S','W') else 1)
            expected = [degrees(fields[2],fields[3]),degrees(fields[4],fields[5])]
            if not np.allclose(lla[:2], expected, rtol=0, atol=1e-7):
                raise ValueError('GGA horizontal position disagrees with NavSatFix')
        covariance = message.position_covariance.tolist() if message.position_covariance_type else None
        if covariance is not None and not any(covariance):
            covariance = None
        header = int(message.header.stamp.sec)*10**9+int(message.header.stamp.nanosec)
        records.append(dict(key=dict(robot=robot,session='replay',id=index), timestamp_ns=measurement,
                            receipt_timestamp_ns=receipt,time_source='gga_rmc_utc',lla=lla,status=int(message.status.status),
                            quality=quality,hdop=hdop,covariance_enu=covariance))
        latencies.append((receipt-measurement)*1e-6)
        pairing.append(dict(id=index,header_timestamp_ns=header,receipt_timestamp_ns=receipt,
                            gga_receipt_ns=nmea_receipt,rmc_receipt_ns=rmc_receipt,measurement_timestamp_ns=measurement))
    if any(b['timestamp_ns'] <= a['timestamp_ns'] for a,b in zip(records,records[1:])):
        raise ValueError('GPS measurement times are not strictly increasing')
    audit = dict(records=len(records),qualities=dict(collections.Counter(r['quality'] for r in records)),
                 unknown_covariance=sum(r['covariance_enu'] is None for r in records),
                 receipt_minus_measurement_ms=dict(zip(['min','median','p95','max'],np.quantile(latencies,[0,.5,.95,1]).tolist())),
                 time_policy='RMC date + matching GGA UTC epoch; no fitted offset; camera calibration remains in immutable cache')
    return records, pairing, audit


def lever_arm(calibration):
    value = json.loads(calibration.read_text())['value0']['T_imu_cam'][0]
    x,y,z,w = (value[k] for k in ('qx','qy','qz','qw'))
    rotation = np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                         [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                         [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    # Nominal mounting datums, re-composed independently; no GNSS estimator code.
    camera_from_receiver = np.diag([-1.,-1.,1.])
    antenna_camera_cad = camera_from_receiver @ np.array([0.,.01045,.0525+.0006])+np.array([-.095,.087,.048])
    camera_cad_from_optical = np.array([[-1.,0.,0.],[0.,0.,-1.],[0.,-1.,0.]])
    optical = camera_cad_from_optical.T @ (antenna_camera_cad-np.array([.0475,-.02135,0.]))
    arm = rotation @ optical + np.array([value[k] for k in ('px','py','pz')])
    if not np.allclose(arm,[.111193231,-.094903010,-.078744703],rtol=0,atol=1e-9):
        raise ValueError('South-ece nominal lever disagrees with expected IMU calibration')
    return arm.tolist(),dict(optical_lever_m=optical.tolist(),imu_lever_m=arm.tolist(),sigma_m=.02,
                            nominal=True,gasket_gap_mm=.6,calibration=str(calibration),
                            calibration_sha256=hashlib.sha256(calibration.read_bytes()).hexdigest(),
                            provenance='Nominal D455f rear mount / CH7604A casing geometry; measured unrectified optical-to-IMU transform; phase centre not calibrated')


def prepare(source_mission, output, rate, frames=None):
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    source = json.loads(source_mission.read_text())
    if len(source['robots']) != 1:
        raise ValueError('South-ece preparation expects one robot')
    robot = source['robots'][0]
    original_config = json.loads(Path(robot['config']).read_text())
    arm, calibration = lever_arm(Path(original_config['calibration']))
    records, mapping, audit = normalize(Path(robot['bag']), robot['robot'])
    output.mkdir(parents=True)
    (output/'gps_records.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in records))
    (output/'gps_timestamp_mapping.json').write_text(json.dumps(mapping,indent=2)+'\n')
    (output/'gps_input_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    (output/'gps_calibration.json').write_text(json.dumps(calibration,indent=2)+'\n')
    gps=dict(enabled=True,lever_arms_m={robot['robot']:arm},lever_sigma_m=.02,
             unknown_horizontal_sigma_m=5.,max_hdop=1.,require_hdop=True,max_reported_horizontal_sigma_m=5.,residual_deadband_sigma=1.,robust_mode='switchable',switch_lambda=9.,huber_delta=3.)
    (output/'gps_backend_config.json').write_text(json.dumps(dict(gps=gps),indent=2)+'\n')
    for name, enabled in [('gps_off',False),('gps_on',True)]:
        run=output/name;run.mkdir()
        config=dict(original_config,output=str(run/'robots'/robot['robot']),gps=dict(enabled=enabled,normalized_input=True))
        (run/'robot.json').write_text(json.dumps(config,indent=2)+'\n')
        backend=json.loads(Path(source['backend_config']).read_text())
        backend.update(output=str(run/'backend'));backend.setdefault('pgo',{})['gps']=dict(gps,enabled=enabled)
        (run/'backend.json').write_text(json.dumps(backend,indent=2)+'\n')
        run_robot=dict(robot,config=str(run/'robot.json'))
        if frames:run_robot['frames']=min(frames,robot['frames'])
        if enabled:run_robot['gps_records']=str(output/'gps_records.jsonl')
        mission=dict(source,rate=rate,backend_config=str(run/'backend.json'),robots=[run_robot])
        (run/'mission.json').write_text(json.dumps(mission,indent=2)+'\n')
    (output/'source_mission.json').write_text(json.dumps(source,indent=2)+'\n')
    print(json.dumps(audit,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source_mission',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--rate',type=float,default=.25);p.add_argument('--frames',type=int);a=p.parse_args()
    prepare(a.source_mission.resolve(),a.output.resolve(),a.rate,a.frames)
