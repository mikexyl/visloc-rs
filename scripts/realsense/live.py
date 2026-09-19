#!/usr/bin/env python3
"""Live D455 stereo+IMU -> Rust Basalt VIO. Uses device-clock timestamps."""
import argparse
from collections import deque
import json
import os
from pathlib import Path
import select
import signal
import struct
import subprocess
import threading
import time

import cv2
import numpy as np
import pyrealsense2 as rs

from browser_stream import BrowserStream
from calibration import prepare
from sync import interpolate_imu



def connect_rerun(url, calibration):
    if not url:
        return None
    import rerun as rr
    import rerun.blueprint as rrb
    rr.init('visloc_D455F_live')
    rr.connect_grpc(url)
    rr.send_blueprint(rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin='world', name='Live VIO'),
            rrb.Vertical(
                rrb.Spatial2DView(origin='world/body/cam0/image', name='Left infrared'),
                rrb.Spatial2DView(origin='world/body/cam1/image', name='Right infrared'),
                rrb.TimeSeriesView(origin='metrics/tracks', name='Visual observations'),
                rrb.TimeSeriesView(origin='metrics/timing', name='Timing (ms)')),
            column_shares=[2, 1]),
        rrb.TimePanel(state='expanded', timeline='elapsed', play_state='Following')))
    rr.log('world', rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    for i, transform in enumerate(calibration['T_imu_cam']):
        path = f'world/body/cam{i}'
        rr.log(path, rr.Transform3D(
            translation=[transform[k] for k in ('px', 'py', 'pz')],
            quaternion=rr.Quaternion(xyzw=[transform[k] for k in ('qx', 'qy', 'qz', 'qw')])), static=True)
        intrinsics = calibration['intrinsics'][i]['intrinsics']
        rr.log(path + '/image', rr.Pinhole(
            focal_length=[intrinsics['fx'], intrinsics['fy']],
            principal_point=[intrinsics['cx'], intrinsics['cy']],
            resolution=calibration['resolution'][i], image_plane_distance=.15), static=True)
    print(f'Rerun SDK connected to {url}', flush=True)
    return rr


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration-dir',type=Path,default=Path('/opt/visloc-rs/configs/realsense/d455_1'))
    parser.add_argument('--config',type=Path,default=Path('/opt/visloc-rs/configs/basalt/euroc_config.json'))
    parser.add_argument('--output',type=Path,default=Path('/output'))
    parser.add_argument('--serial',default='425122300073')
    parser.add_argument('--duration',type=float,default=0,help='Seconds after warmup; zero runs until stopped')
    parser.add_argument('--vio-fps',type=float,default=5,help='Maximum processed stereo frame rate; all IMU samples retained')
    parser.add_argument('--camera-fps',type=int,default=30,choices=[15,30])
    parser.add_argument('--binary',default='basalt_stream_vio')
    parser.add_argument('--rerun-connect',default=os.environ.get('RERUN_CONNECT'),help='Rerun gRPC URL; unset disables visualization')
    parser.add_argument('--browser-connect', default=os.environ.get('BROWSER_CONNECT'), help='Online browser server URL, e.g. http://192.168.0.243:8090')
    args=parser.parse_args()
    if not 0 < args.vio_fps <= args.camera_fps or args.duration < 0:
        parser.error('Require 0 < vio-fps <= camera-fps and duration >= 0')
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'trajectory.tum').exists():
        parser.error('Output already contains a trajectory; choose a fresh directory')
    maps,offset,calibration=prepare(args.calibration_dir,args.output)
    context=rs.context()
    devices=[d for d in context.query_devices() if d.get_info(rs.camera_info.serial_number)==args.serial]
    if len(devices)!=1:
        raise RuntimeError(f'RealSense serial {args.serial} not found')
    device=devices[0]
    streams=set()
    for sensor in device.query_sensors():
        streams.update(p.stream_type() for p in sensor.get_stream_profiles())
        for option,value in [(rs.option.global_time_enabled,0),(rs.option.emitter_enabled,0)]:
            if sensor.supports(option): sensor.set_option(option,value)
    if not {rs.stream.gyro,rs.stream.accel}.issubset(streams):
        raise RuntimeError('No IMU exposed: use the RSUSB image; do not run stereo-only as VIO')
    lock=threading.Lock()
    gyro,accel,images=deque(),deque(),deque(maxlen=10)
    errors=[]
    counts={'gyro':0,'accel':0,'stereo':0}
    domains=set()
    last_motion={}
    stopped=threading.Event()
    clock_ready=threading.Event()
    class StartupClock(Exception):
        pass
    for sig in (signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,lambda *_: stopped.set())

    def stamp(frame):
        domain=frame.get_frame_timestamp_domain()
        if domain != rs.timestamp_domain.hardware_clock:
            if not clock_ready.is_set(): raise StartupClock()
            raise RuntimeError(f'Expected hardware clock, got {domain}')
        domains.add(str(domain))
        return round(frame.get_timestamp()*1e6)

    def callback(frame):
        try:
            with lock:
                if frame.is_motion_frame():
                    kind=frame.get_profile().stream_type()
                    key='gyro' if kind==rs.stream.gyro else 'accel'
                    t=stamp(frame)
                    if key in last_motion and t<=last_motion[key]:
                        raise RuntimeError(f'Non-monotonic {key} device clock')
                    last_motion[key]=t
                    v=frame.as_motion_frame().get_motion_data()
                    queue=gyro if key=='gyro' else accel
                    queue.append((t,(v.x,v.y,v.z)))
                    counts[key]+=1
                    if len(queue)>4000:
                        raise RuntimeError('IMU buffer overrun; estimator is not keeping up')
                elif frame.is_frameset():
                    fs=frame.as_frameset()
                    left,right=fs.get_infrared_frame(1),fs.get_infrared_frame(2)
                    if left and right:
                        lt,rt=stamp(left),stamp(right)
                        if abs(lt-rt)>1_000_000:
                            raise RuntimeError('Stereo timestamp skew exceeds 1 ms')
                        images.append((lt+offset,np.asanyarray(left.get_data()).copy(),np.asanyarray(right.get_data()).copy()))
                        counts['stereo']+=1
        except StartupClock:
            return
        except Exception as error:
            with lock:
                if not errors: errors.append(str(error))
            stopped.set()

    config=rs.config()
    config.enable_device(args.serial)
    config.enable_stream(rs.stream.infrared,1,640,480,rs.format.y8,args.camera_fps)
    config.enable_stream(rs.stream.infrared,2,640,480,rs.format.y8,args.camera_fps)
    config.enable_stream(rs.stream.gyro,rs.format.motion_xyz32f,200)
    config.enable_stream(rs.stream.accel,rs.format.motion_xyz32f,200)
    pipeline=rs.pipeline(context)
    process=None
    started=False
    processed=0
    processing_ms_total=0.0
    rr=None
    browser=None
    trail=deque(maxlen=1500)
    first_vio_ns=None
    metadata={'serial':args.serial,'device':device.get_info(rs.camera_info.name),
              'firmware':device.get_info(rs.camera_info.firmware_version),'camera_time_shift_ns':offset,
              'camera_fps':args.camera_fps,'vio_fps_limit':args.vio_fps,'imu_rate_hz':200,
              'emitter_enabled':False,'clock':'hardware_clock','calibration_dir':str(args.calibration_dir)}
    start=time.monotonic()
    try:
        profile=pipeline.start(config,callback); started=True
        # Configure the active pipeline sensor instances; discovery instances can
        # have separate global-time readers in librealsense.
        for sensor in profile.get_device().query_sensors():
            for option,value in [(rs.option.global_time_enabled,0),(rs.option.emitter_enabled,0),(rs.option.enable_auto_exposure,1)]:
                if sensor.supports(option): sensor.set_option(option,value)
        # Record SDK extrinsics for auditing the sensor coordinate convention.
        gp=profile.get_stream(rs.stream.gyro)
        ap=profile.get_stream(rs.stream.accel)
        ext=ap.get_extrinsics_to(gp)
        if not np.allclose(np.array(ext.rotation).reshape(3,3),np.eye(3),atol=1e-4):
            raise RuntimeError('Accel and gyro axes differ; explicit frame conversion required')
        metadata['accel_to_gyro_rotation']=list(ext.rotation)
        metadata['gyro_to_ir1_rotation']=list(gp.get_extrinsics_to(profile.get_stream(rs.stream.infrared,1)).rotation)
        (args.output/'device.json').write_text(json.dumps(metadata,indent=2))
        print(json.dumps(metadata),flush=True)
        # Warm up exposure and acquire bracketing samples before initializing gravity.
        stopped.wait(2.0)
        clock_ready.set()
        if stopped.is_set():
            raise RuntimeError('; '.join(errors) or 'Stopped before initialization')
        process=subprocess.Popen([args.binary,str(args.output/'basalt_calibration.json'),str(args.config),str(args.output)],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
        rr=connect_rerun(args.rerun_connect,calibration)
        browser=BrowserStream(args.browser_connect) if args.browser_connect else None
        previous=None
        capture_start=time.monotonic()
        last_success=capture_start
        with (args.output/'live.jsonl').open('w') as log:
            while not stopped.is_set() and (args.duration==0 or time.monotonic()-capture_start<args.duration):
                packet=None
                with lock:
                    if errors: raise RuntimeError(errors[0])
                    if images and len(gyro)>1 and len(accel)>1:
                        eligible=[im for im in images if im[0]<gyro[-1][0] and im[0]<accel[-1][0]
                                  and (previous is None or im[0]-previous>=int(1e9/args.vio_fps))]
                        if eligible:
                            t,left,right=eligible[-1]
                            start_ns=previous if previous is not None else t-100_000_000
                            gs,acs=list(gyro),list(accel)
                            samples=interpolate_imu(gs,acs,start_ns,t)
                            future=next((g for g in gs if g[0]>=t and g[0]<=acs[-1][0]),None)
                            if previous is not None or future is not None:
                                initialization=interpolate_imu(gs,acs,future[0]-1,future[0])[0] if previous is None else None
                                packet=(t,left,right,samples,initialization)
                                while gyro and gyro[0][0]<=t: gyro.popleft()
                                while len(accel)>1 and accel[1][0]<=t: accel.popleft()
                                while images and images[0][0]<=t: images.popleft()
                if packet is None:
                    if time.monotonic()-last_success>10:
                        raise RuntimeError('No synchronized stereo/IMU packet for 10 seconds')
                    time.sleep(.005); continue
                t,left,right,samples,initialization=packet
                h=json.dumps({'timestamp_ns':t,'width':640,'height':480,'imu':samples,'initialization_imu':initialization},allow_nan=False).encode()
                process.stdin.write(struct.pack('<I',len(h))+h)
                for i,raw in enumerate((left,right)):
                    image=cv2.remap(raw,*maps[i],cv2.INTER_LINEAR)
                    process.stdin.write(image.tobytes())
                    # Latest preview for operational checks, bounded storage.
                    cv2.imwrite(str(args.output/f'cam{i}_latest.jpg'),image)
                process.stdin.flush()
                if not select.select([process.stdout],[],[],30)[0]:
                    raise RuntimeError('VIO did not acknowledge a frame within 30 seconds')
                line=process.stdout.readline()
                if not line: raise RuntimeError(f'VIO exited unexpectedly ({process.poll()})')
                result=json.loads(line)
                with lock:
                    result['sensor_lag_ms']=(last_motion['gyro']-t)/1e6
                result['accel_norm_m_s2']=float(np.linalg.norm(np.array(samples)[:,4:].mean(axis=0)))
                result['gyro_norm_rad_s']=float(np.linalg.norm(np.array(samples)[:,1:4].mean(axis=0)))
                log.write(json.dumps(result)+'\n'); log.flush()
                (args.output/'status.json').write_text(json.dumps(result,indent=2))
                if browser is not None:
                    browser.publish(result, [(args.output/f'cam{i}_latest.jpg').read_bytes() for i in range(2)])
                if rr is not None:
                    if first_vio_ns is None: first_vio_ns=t
                    rr.set_time('elapsed',duration=(t-first_vio_ns)*1e-9)
                    rr.set_time('frame',sequence=result['frame_id'])
                    rr.log('world/body',rr.Transform3D(translation=result['position'],quaternion=rr.Quaternion(xyzw=result['quaternion_xyzw'])))
                    trail.append(result['position'])
                    if len(trail)>1: rr.log('world/trajectory',rr.LineStrips3D([list(trail)],colors=[40,160,255],radii=.002))
                    for i in range(2):
                        rr.log(f'world/body/cam{i}/image',rr.EncodedImage(contents=(args.output/f'cam{i}_latest.jpg').read_bytes(),media_type='image/jpeg'))
                        rr.log(f'metrics/tracks/cam{i}',rr.Scalars(result['observations'][i]))
                    for key in ('process_ms','sensor_lag_ms'):
                        rr.log(f'metrics/timing/{key}',rr.Scalars(result[key]))
                processed+=1; previous=t; last_success=time.monotonic()
                processing_ms_total+=result['process_ms']
                if processed==1 or processed%10==0: print(json.dumps(result),flush=True)
        if errors: raise RuntimeError(errors[0])
        if processed==0: raise RuntimeError('No VIO frames processed')
    except Exception as error:
        errors.append(str(error))
        raise
    finally:
        if started: pipeline.stop()
        if process is not None:
            if process.stdin: process.stdin.close()
            try: code=process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(); code=-9
            if code: errors.append(f'VIO exit code {code}')
        if browser is not None: browser.close()
        if rr is not None: rr.disconnect()
        summary={'frames_processed':processed,'sensor_counts':counts,'wall_seconds':time.monotonic()-start,
                 'process_ms_mean':processing_ms_total/processed if processed else None,'errors':errors,
                 'clock_domains':list(domains)}
        (args.output/'capture_summary.json').write_text(json.dumps(summary,indent=2))
        print(json.dumps(summary),flush=True)
    if errors: raise RuntimeError('; '.join(errors))


if __name__=='__main__':
    main()
