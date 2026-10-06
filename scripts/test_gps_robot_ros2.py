#!/usr/bin/env python3
"""Actual Rust GPS adapter restart, session history and duplicate/conflict checks.
No camera frames, IMU, GPU inference, backend or GPS hardware are used.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import rclpy
from visloc_msgs import msg as m,srv as s
from test_multi_robot_robot_ros2 import Fixture,REPO


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('robot_config',type=Path);a=p.parse_args()
    if os.environ.get('ROS_DOMAIN_ID')!='226':raise SystemExit('Use isolated ROS_DOMAIN_ID=226')
    rclpy.init();node=Fixture();process=None
    pub=node.create_publisher(m.GpsFix,'/queue_test/gps/normalized',32)
    history=node.create_client(s.GetGpsHistory,'/queue_test/slam/gps_history')
    with tempfile.TemporaryDirectory(prefix='visloc-gps-robot-') as d:
        root=Path(d);out=root/'robot';cfg=json.loads(a.robot_config.read_text())
        cfg.update(robot='queue_test',peers=['queue_test'],output=str(out),camera_mode='mono',preprocess=None,preprocess_right=None,
                   reliable_sensors=True,loop_enabled=False,imu_startup=None,gps={'enabled':True,'normalized_input':True})
        config=root/'config.json';config.write_text(json.dumps(cfg));log=(root/'robot.log').open('w')
        binary=Path(os.environ.get('VISLOC_ROS_INSTALL',REPO/'.runtime/ros2_install'))/'visloc_ros/lib/visloc_ros/robot'
        def launch():return subprocess.Popen([str(binary)],env=dict(os.environ,VISLOC_ROBOT_CONFIG=str(config)),stdout=log,stderr=subprocess.STDOUT)
        latest=None
        def count(n,rejected=0):
            nonlocal latest
            latest=node.call(history,s.GetGpsHistory.Request(cursor=0));return len(latest.fixes)==n and latest.rejected_inputs==rejected
        try:
            process=launch();node.until(lambda:node.status and node.status.ready,'robot ready')
            node.until(lambda:pub.get_subscription_count()>0,'GPS subscription')
            fix=m.GpsFix(key=m.Key(robot='input',session='unassigned',id=0),timestamp_ns=10**9,receipt_timestamp_ns=10**9+10_000_000,
                         time_source='gnss_utc',has_position=True,lla=[35.,33.,9999.],status=0,has_quality=True,quality=1,has_hdop=True,hdop=.8)
            pub.publish(fix);node.until(lambda:count(1),'GPS journal and history')
            old=latest.session;assert latest.fixes[0].key.session==old and latest.fixes[0].key.robot=='queue_test'
            pub.publish(fix);conflict=m.GpsFix(**{name:getattr(fix,name) for name in fix.get_fields_and_field_types()});conflict.lla=[35.1,33.,9999.];pub.publish(conflict)
            node.until(lambda:count(1,1),'duplicate idempotence and conflict rejection')
            process.terminate();process.wait(timeout=10)
            with (out/'gps_records.jsonl').open('a') as partial:partial.write('{partial')
            process=launch();node.until(lambda:node.status.session!=old and node.status.ready,'fresh estimator session')
            node.until(lambda:count(1) and latest.session!=old,'archived GPS recovered')
            pub.publish(fix);node.until(lambda:count(2),'reused sample id in new session')
            assert {r.key.session for r in latest.fixes}=={old,latest.session};assert latest.dropped_inputs==0
            report=dict(passed=True,records=2,sessions=2,ordinary_gps=True,duplicate_idempotent=True,conflict_rejected=True,partial_archive_recovered=True,old_gps_history_preserved=True)
            (REPO/'.runtime/gps_robot_integration_result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
        except Exception:
            print((root/'robot.log').read_text());raise
        finally:
            if process and process.poll() is None:process.terminate();process.wait(timeout=10)
            log.close();node.destroy_node();rclpy.shutdown()
if __name__=='__main__':main()
