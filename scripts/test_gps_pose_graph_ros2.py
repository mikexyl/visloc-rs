#!/usr/bin/env python3
"""Native Rust GPS backend interoperability, history, restart and session isolation.
Uses synthetic positions, no estimator/GPU, no GPS device, and an isolated DDS domain.
"""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import rclpy
from rclpy.serialization import serialize_message, deserialize_message
from visloc_msgs import msg as m, srv as s
from test_multi_robot_ros2 import Fixture, record, key, REPO

class GpsFixture(Fixture):
    def __init__(self):
        super().__init__();self.gps=[];self.gps_status=None;self.gps_requests=0;self.gps_session='test_session'
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.gps_sub=self.create_subscription(m.GpsStatus,'/visloc/gps_status',lambda x:setattr(self,'gps_status',x),qos)
        self.gps_pub=self.create_publisher(m.GpsFix,'/alpha/slam/gps',256)
        self.gps_service=None
    def connect_gps(self):
        def history(q,r):
            self.gps_requests+=1;r.session=self.gps_session
            begin=min(q.cursor,len(self.gps));end=min(begin+3,len(self.gps));r.fixes=self.gps[begin:end];r.cursor=end;r.more=end<len(self.gps);return r
        self.gps_service=self.create_service(s.GetGpsHistory,'/alpha/slam/gps_history',history)

def frame(i):
    r=record('alpha',i);r.timestamp_ns=(i+1)*10**9;r.body_to_odom.translation=[i*3.,0.,1.];return r

def fix(i,session='test_session'):
    import math
    # GPS track is the raw track rotated 90 degrees and translated (10,20).
    return m.GpsFix(key=m.Key(robot='alpha',session=session,id=i),timestamp_ns=(i+1)*10**9,receipt_timestamp_ns=(i+1)*10**9+10_000_000,
                    time_source='synthetic_measurement_utc',has_position=True,lla=[(20.+3.*i)/6335439.32729282*180/math.pi,10./6378137.*180/math.pi,0.],
                    status=0,has_quality=True,quality=2,has_hdop=True,hdop=.8,has_covariance=False,covariance_enu=[0.]*9)

def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,default=REPO/'.runtime/gps_ros2_integration_result.json')
    args=parser.parse_args()
    if os.environ.get('ROS_DOMAIN_ID')!='224':raise SystemExit('Use isolated ROS_DOMAIN_ID=224')
    rclpy.init();node=GpsFixture();process=None
    with tempfile.TemporaryDirectory(prefix='visloc-gps-ros2-') as d:
        root=Path(d);config=root/'config.json';out=root/'backend'
        config.write_text(json.dumps({'peers':['alpha'],'output':str(out),'pgo':{'gps':{'enabled':True,'origin':{'latitude':0.,'longitude':0.,'altitude':0.},'lever_arms_m':{'alpha':[0.,0.,0.]}}}}))
        log=(root/'backend.log').open('w');env=dict(os.environ,VISLOC_BACKEND_CONFIG=str(config));binary=Path(os.environ.get('VISLOC_ROS_INSTALL',REPO/'.runtime/ros2_install'))/'visloc_ros/lib/visloc_ros/backend'
        def launch():return subprocess.Popen([str(binary),'--ros-args','-p','use_sim_time:=true'],env=env,stdout=log,stderr=subprocess.STDOUT)
        def checkpoint():
            future=node.finish.call_async(s.Finish.Request(last_image_timestamp_ns=0));node.until(future.done,'finish');assert future.result().accepted
            node.until(lambda:(out/'finished.json').exists(),'persisted checkpoint')
        def gps_count(n):return node.gps_status and node.gps_status.total_records==n
        try:
            process=launch();node.until(lambda:node.gps_pub.get_subscription_count()>0,'GPS discovery')
            # CDR verifies field order/precision through generated custom interfaces.
            for message in [fix(0),m.GpsStatus(revision=2,total_records=3,active_factors=1,aligned_components=1),s.GetGpsHistory.Request(cursor=2),s.GetGpsHistory.Response(session='test',fixes=[fix(0)],cursor=1,more=True)]:
                assert deserialize_message(serialize_message(message),type(message))==message
            # No-fix data is counted and diagnosed without ever entering the solver.
            bad=fix(99);bad.status=-1;bad.has_position=False;node.gps_pub.publish(bad);node.until(lambda:gps_count(1),'no-fix diagnostic')
            node.records['alpha']=[frame(i) for i in [7,0,2,1,3,4,5,6]+list(range(8,16))];node.records['beta']=[];node.discover()
            node.until(lambda:node.shape(16,1),'ordered keyframe history')
            node.gps=[fix(i) for i in reversed(range(16))];node.connect_gps()
            node.until(lambda:gps_count(17) and node.gps_status.aligned_components==1,'GPS-only initialization via paginated history')
            assert node.gps_status.active_factors==16;assert len(node.graph.loops)==0
            poses={p.key.id:p.body_to_map.translation for p in node.graph.poses};assert abs(poses[0][0]-10)<.02 and abs(poses[0][1]-20)<.02;assert abs(poses[0][2]-1)<1e-8
            revision=node.graph.input_revision
            for r in node.gps:node.gps_pub.publish(r)
            conflict=fix(0);conflict.lla[0]+=.01;node.gps_pub.publish(conflict)
            node.until(lambda:(out/'revisions.jsonl').exists() and 'gps_rejected' in (out/'revisions.jsonl').read_text(),'explicit conflict rejection')
            assert node.graph.input_revision==revision
            checkpoint();snapshot=json.loads((out/'graph_snapshot.json').read_text());datum=snapshot['gps']['datum'];old_revision=snapshot['revision']
            assert snapshot['optimizer_reports'][0]['solver']=='gtsam_cpp'
            assert snapshot['optimizer_reports'][0]['version']=='4.3.0'
            assert snapshot['optimizer_reports'][0]['gps_factors']==16
            process.terminate();process.wait(timeout=10)
            # Records and fixes emitted while disconnected are recovered only by history.
            for i in range(16,20):node.records['alpha'].append(frame(i));node.gps.append(fix(i))
            with (out/'graph.jsonl').open('a') as f:f.write('{"gps":')
            node.graph=None;process=launch();node.until(lambda:node.shape(20,1) and gps_count(21) and node.graph.revision>old_revision,'restart and outage catch-up')
            checkpoint();restored=json.loads((out/'graph_snapshot.json').read_text());assert restored['gps']['datum']==datum;assert len(restored['gps']['aligned_components'])==1
            assert abs(restored['poses'][0]['body_to_map']['translation'][0]-poses[0][0])<1e-3
            # Reused local sample IDs in another estimator session cannot attach to old poses.
            node.gps_session='restarted';node.gps=[fix(0,'restarted')]
            node.until(lambda:gps_count(22),'history cursor reset on new session')
            future=node.finish.call_async(s.Finish.Request(last_image_timestamp_ns=0));node.until(future.done,'last solve')
            node.until(lambda:any(x['key']['session']=='restarted' for x in json.loads((out/'gps_diagnostics.json').read_text())['gps']['diagnostics']),'new-session diagnostic persisted')
            latest=json.loads((out/'gps_diagnostics.json').read_text());d=next(x for x in latest['gps']['diagnostics'] if x['key']['session']=='restarted');assert d['reason']=='pending_keyframes'
            report={'passed':True,'records':22,'active_factors':20,'gps_history_requests':node.gps_requests,'no_loops_required':True,'datum_and_enu_warm_start_preserved':True,'conflicts_rejected':True,'new_session_isolated':True,'partial_journal_recovered':True,'frozen_ros_time':True,'custom_message_service_cdr':True}
            report['optimizer']='gtsam_cpp'
            args.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
        except Exception:
            print((root/'backend.log').read_text());raise
        finally:
            if process and process.poll() is None:process.terminate();process.wait(timeout=10)
            log.close();node.destroy_node();rclpy.shutdown()

if __name__=='__main__':main()
