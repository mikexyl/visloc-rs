#!/usr/bin/env python3
"""Exercise the real native rclrs backend with Python-generated ROS2 records.

Tests delayed discovery, typed services, pagination, duplicate/out-of-order
records, a bridge between components, frozen/jumping /clock, missed messages,
persistent restart, and history recovery. No GPU or datasets are required.
"""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Path as PathMessage
from visloc_msgs.msg import Key, Transform, Keyframe, LoopConstraint, GraphSnapshot, Status
from visloc_msgs.srv import GetHistory, Finish

REPO=Path(__file__).resolve().parents[1]

def key(robot, local):
    return Key(robot=robot,session='test_session',id=local)

def pose(x):
    return Transform(translation=[float(x),0.,0.],rotation_xyzw=[0.,0.,0.,1.])

def record(robot, local):
    return Keyframe(key=key(robot,local),timestamp_ns=10**9+local*10**8,
                    body_to_odom=pose(local),has_previous=local>0,
                    previous=key(robot, max(0,local-1)))

class Fixture(Node):
    def __init__(self):
        super().__init__('multi_robot_contract_test')
        self.sessions={'alpha':'test_session','beta':'test_session'};self.paths={}
        self.records={'alpha':[], 'beta':[]}; self.loops=[];self.graph=None;self.requests=0;self.delayed_once=False
        qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.sub=self.create_subscription(GraphSnapshot,'/visloc/graph',lambda g:setattr(self,'graph',g),qos)
        self.kf_publishers={r:self.create_publisher(Keyframe,f'/{r}/slam/keyframes',256) for r in self.records}
        self.loop_pub=self.create_publisher(LoopConstraint,'/visloc/loops',256)
        self.clock=self.create_publisher(Clock,'/clock',10)
        self.finish=self.create_client(Finish,'/visloc/backend/finish')
        self.history_services=[]
        self.path_sub=self.create_subscription(PathMessage,'/alpha/slam/path',lambda m:self.paths.update({'alpha':m}),qos)
        self.status_pub=self.create_publisher(Status,'/alpha/slam/status',qos)
    def discover(self):
        for robot in self.records:
            self.history_services.append(self.create_service(GetHistory,f'/{robot}/slam/history',lambda q,r,n=robot:self.history(n,q,r)))
    def history(self,robot,q,r):
        self.requests+=1;r.session=self.sessions[robot]
        if not self.delayed_once:
            self.delayed_once=True;time.sleep(2.15)  # force a monotonic-time retry with /clock frozen
        # Two records/page forces actual paginated catch-up.
        a=min(q.keyframe_cursor,len(self.records[robot]));b=min(a+2,len(self.records[robot]))
        r.keyframes=self.records[robot][a:b] if q.include_keyframes else []
        r.keyframe_cursor=b;r.sequence_cursor=0
        r.loops=self.loops[q.loop_cursor:] if robot=='alpha' and q.include_loops else []
        r.loop_cursor=len(self.loops) if robot=='alpha' else 0;r.more=b<len(self.records[robot])
        return r
    def until(self,condition,reason,timeout=20):
        started=time.monotonic()
        while not condition():
            rclpy.spin_once(self,timeout_sec=.03)
            if time.monotonic()-started>timeout:raise TimeoutError(reason)
    def shape(self,poses,components):
        return self.graph is not None and len(self.graph.poses)==poses and self.graph.components==components

def main():
    if os.environ.get('ROS_DOMAIN_ID')!='219':
        raise SystemExit('Use isolated ROS_DOMAIN_ID=219 for this destructive-process test')
    rclpy.init();node=Fixture();process=None;log=None
    with tempfile.TemporaryDirectory(prefix='visloc-ros2-test-') as directory:
        root=Path(directory);config=root/'config.json'
        config.write_text(json.dumps({'peers':['alpha','beta'],'output':str(root/'backend')}))
        env=dict(os.environ,VISLOC_BACKEND_CONFIG=str(config));log=(root/'backend.log').open('w')
        def launch():
            return subprocess.Popen([str(Path(os.environ.get('VISLOC_ROS_INSTALL',REPO/'.runtime/ros2_install'))/'visloc_ros/lib/visloc_ros/backend'),'--ros-args','-p','use_sim_time:=true'],env=env,stdout=log,stderr=subprocess.STDOUT)
        try:
            process=launch()
            node.until(lambda:all(p.get_subscription_count() for p in node.kf_publishers.values()),'native subscriptions')
            # Missing predecessors can temporarily split a single session.
            # Its map frame must include the component's local root ID too.
            node.kf_publishers['alpha'].publish(record('alpha',2))
            node.until(lambda:node.shape(1,1),'isolated late keyframe')
            node.until(lambda:'alpha' in node.paths and node.paths['alpha'].header.frame_id=='map_alpha_test_session_2','fragment-specific correction frame')
            node.kf_publishers['alpha'].publish(record('alpha',0))
            node.until(lambda:node.shape(2,2),'same-session disconnected fragments')
            assert {p.component.id for p in node.graph.poses}=={0,2}
            # Services appear after backend startup, with deliberately shuffled records.
            for r in node.records:node.records[r]=[record(r,i) for i in [2,0,1]]
            node.discover()
            node.until(lambda:node.shape(6,2),'delayed/paginated history')
            revision=node.graph.input_revision
            for r,records in node.records.items():
                for k in records*2:node.kf_publishers[r].publish(k)
            loop=LoopConstraint(sequence_from=key('alpha',0),sequence_to=key('beta',0),from_key=key('alpha',1),to_key=key('beta',4),to_from=pose(-13.),similarity=.91)
            loop.information=[25. if i==j and i<3 else 625. if i==j else 0. for i in range(6) for j in range(6)]
            loop.verification.matches=40;loop.verification.two_d_inliers=35;loop.verification.pnp_inliers=30;loop.verification.reason='synthetic_fixture'
            node.loops.append(loop);node.loop_pub.publish(loop);node.loop_pub.publish(loop)
            # Bridge arrives before its endpoint. ROS time freezes and jumps backwards.
            for sec in [42,42,0]:
                c=Clock();c.clock.sec=sec;node.clock.publish(c)
            node.records['beta'].extend([record('beta',4),record('beta',3)])
            node.until(lambda:node.shape(8,1),'out-of-order bridge')
            assert node.graph.input_revision==revision+3, 'duplicate factors changed the graph'
            assert len(node.graph.loops)==1
            beta4=next(p for p in node.graph.poses if p.key.robot=='beta' and p.key.id==4)
            assert abs(beta4.body_to_map.translation[0]-14)<1e-5
            first_finish=node.finish.call_async(Finish.Request(last_image_timestamp_ns=0))
            node.until(first_finish.done,'pre-restart checkpoint');assert first_finish.result().accepted
            node.until(lambda:(root/'backend/finished.json').exists(),'pre-restart checkpoint on disk')
            traffic=json.loads((root/'backend/communication.json').read_text())
            assert traffic['service_attempts']>traffic['service_responses'], 'delayed service did not exercise timeout/retry'
            old_revision=node.graph.revision
            # Disconnect the backend, emit records while it is absent, restart
            # from the journal, then recover the missed tail through services.
            process.terminate();process.wait(timeout=10)
            for i in range(5,10):
                r=record('beta',i);node.records['beta'].append(r);node.kf_publishers['beta'].publish(r)
            # Simulate an interrupted final journal write as well as outage.
            with (root/'backend/graph.jsonl').open('a') as partial:partial.write('{"keyframe":')
            node.graph=None;process=launch()
            node.until(lambda:node.shape(13,1) and node.graph.revision>old_revision,'restart/catch-up')
            assert len(node.graph.loops)==1
            # An estimator restart reuses local IDs but creates a new session.
            node.sessions['alpha']='restarted';node.records['alpha']=[record('alpha',i) for i in range(2)]
            for r in node.records['alpha']:
                r.key.session='restarted';r.previous.session='restarted'
            node.status_pub.publish(Status(robot='alpha',session='restarted',ready=True))
            node.until(lambda:node.shape(15,2),'new estimator session/cursor reset')
            node.until(lambda:'alpha' in node.paths and node.paths['alpha'].header.frame_id=='map_alpha_restarted_0','active-session correction frame')
            future=node.finish.call_async(Finish.Request(last_image_timestamp_ns=0))
            node.until(future.done,'final solve');assert future.result().accepted
            node.until(lambda:(root/'backend/finished.json').exists(),'final persisted solve')
            assert node.graph.final_cost<=node.graph.initial_cost+1e-8
            report={'passed':True,'keyframes':15,'components':2,'unique_loops':1,'fragment_frame_ids_distinct':True,'history_requests':node.requests,'timed_out_attempts_before_restart':traffic['service_attempts']-traffic['service_responses'],'revision_before_restart':old_revision,'revision_after_restart':node.graph.revision}
            print(json.dumps(report,indent=2))
            (REPO/'.runtime/ros2_integration_result.json').write_text(json.dumps(report,indent=2))
        except Exception:
            print((root/'backend.log').read_text());raise
        finally:
            if process and process.poll() is None:process.terminate();process.wait(timeout=10)
            log.close();node.destroy_node();rclpy.shutdown()

if __name__=='__main__':main()
