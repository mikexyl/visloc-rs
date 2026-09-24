#!/usr/bin/env python3
"""Real TensorRT loop verification after peer outage, using archived mission data.

A Rust robot restores its old keyframes. A Python peer announces a real matching
sequence but withholds services until the initial timeout/retry budget expires.
After catch-up, repeated announcements must produce exactly one loop factor.
"""
import argparse,json,os,shutil,subprocess,tempfile,time
from pathlib import Path
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile,ReliabilityPolicy,DurabilityPolicy
from visloc_msgs import msg as m,srv as s

REPO=Path(__file__).resolve().parents[1]

def key(v):return m.Key(**v)
def transform(v):return m.Transform(**v)
def sequence(v):
    return m.SequenceAnnouncement(key=key(v['key']),members=[key(x) for x in v['members']],selected=[key(x) for x in v['selected']],
        start_ns=v['start_ns'],end_ns=v['end_ns'],model_id=v['model_id'],descriptor=v['descriptor'],excluded_keyframes=v['excluded_keyframes'])
def matrix(v):return m.FrameDescriptors(sequence=key(v['sequence']),frames=[key(x) for x in v['frames']],descriptors=v['descriptors'])
def feature(v):
    cam=v['camera'];camera=m.Camera(width=cam['width'],height=cam['height'],intrinsics=cam['intrinsics'],camera_to_body=transform(cam['camera_to_body']))
    fields=[m.Feature(track_id=tid,pixel=pixel,descriptor=v['descriptors'][i*64:(i+1)*64],has_landmark=point is not None,point_camera=point if point is not None else [0.,0.,0.])
            for i,(tid,pixel,point) in enumerate(zip(v['track_ids'],v['pixels'],v['points_camera']))]
    return m.FeatureFrame(key=key(v['key']),timestamp_ns=v['timestamp_ns'],camera=camera,features=fields)

class Fixture(Node):
    def __init__(self,owner,remote):
        super().__init__('visloc_reconnect_fixture');self.status=None;self.edges=[];self.service_handles=[];self.remote=remote
        qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.status_sub=self.create_subscription(m.Status,f'/{owner}/slam/status',lambda v:setattr(self,'status',v),qos)
        self.edge_sub=self.create_subscription(m.LoopConstraint,'/visloc/loops',self.edges.append,256)
        self.announcements=self.create_publisher(m.SequenceAnnouncement,'/visloc/sequences',256)
        self.finish=self.create_client(s.Finish,f'/{owner}/slam/finish')
    def until(self,f,reason,timeout=35):
        started=time.monotonic()
        while not f():
            rclpy.spin_once(self,timeout_sec=.03)
            if self.status and self.status.error:raise RuntimeError(self.status.error)
            if time.monotonic()-started>timeout:raise TimeoutError(reason)
    def connect(self):
        robot=self.remote['sequence']['key']['robot'];session=self.remote['sequence']['key']['session']
        seq=sequence(self.remote['sequence']);desc=matrix(self.remote['descriptors'])
        frames={v['key']['id']:feature(v) for v in self.remote['features']}
        def history(q,r):
            r.session=session;r.sequences=[seq] if q.sequence_cursor==0 else [];r.sequence_cursor=1;return r
        def descriptors(q,r):
            r.found=q.key==seq.key
            if r.found:r.descriptors=desc
            return r
        def features(q,r):
            r.found=q.key.session==session and q.key.id in frames
            if r.found:r.frame=frames[q.key.id]
            return r
        self.service_handles=[self.create_service(s.GetHistory,f'/{robot}/slam/history',history),
                              self.create_service(s.GetSequence,f'/{robot}/slam/sequence_frames',descriptors),
                              self.create_service(s.GetFeatures,f'/{robot}/slam/features',features)]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mission',type=Path);a=p.parse_args()
    if os.environ.get('ROS_DOMAIN_ID')!='221':raise SystemExit('Use isolated ROS_DOMAIN_ID=221')
    mission=json.loads(a.mission.read_text());base=a.mission.parent
    graph=json.loads((base/'backend/graph_snapshot.json').read_text());edge=next(e for e in graph['loops'] if e['from']['robot']!=e['to']['robot'])
    ka,kb=edge['pair'];owner=ka['robot'];remote_robot=kb['robot']
    local_path=base/'robots'/owner/'sequences'/f"{ka['id']:06}.json"
    remote=json.loads((base/'robots'/remote_robot/'sequences'/f"{kb['id']:06}.json").read_text())
    cfg=json.loads(Path(next(r['config'] for r in mission['robots'] if r['robot']==owner)).read_text())
    rclpy.init();node=Fixture(owner,remote);process=None
    with tempfile.TemporaryDirectory(prefix='visloc-reconnect-') as directory:
        root=Path(directory);archive=root/'robot';(archive/'sequences').mkdir(parents=True)
        shutil.copyfile(local_path,archive/'sequences'/local_path.name)
        # Only one local sequence; no existing attempt/loop journals in this
        # disposable fixture. The original experiment is never modified.
        shutil.copyfile(base/'robots'/owner/'keyframes.jsonl',archive/'keyframes.jsonl')
        (archive/'config.json').write_text('{}')
        cfg.update(output=str(archive),robot=owner,peers=[owner,remote_robot],loop_enabled=True)
        path=root/'config.json';path.write_text(json.dumps(cfg));log=(root/'robot.log').open('w')
        binary=Path(os.environ.get('VISLOC_ROS_INSTALL',REPO/'.runtime/ros2_install'))/'visloc_ros/lib/visloc_ros/robot'
        try:
            process=subprocess.Popen([str(binary)],env=dict(os.environ,VISLOC_ROBOT_CONFIG=str(path)),stdout=log,stderr=subprocess.STDOUT)
            node.until(lambda:node.status is not None and node.status.ready,'Rust model readiness',120)
            node.until(lambda:node.announcements.get_subscription_count()>0,'announcement discovery')
            announcement=sequence(remote['sequence']);node.announcements.publish(announcement)
            node.until(lambda:node.status.request_failures>=1,'peer request timeout budget')
            node.connect()
            for _ in range(8):node.announcements.publish(announcement)
            node.until(lambda:len(node.edges)>=1,'retried staged exchange and verification')
            future=node.finish.call_async(s.Finish.Request(last_image_timestamp_ns=0));node.until(future.done,'finish');assert future.result().accepted
            node.until(lambda:node.status.finished,'all verification output drained')
            for _ in range(8):node.announcements.publish(announcement)
            started=time.monotonic()
            while time.monotonic()-started<1:rclpy.spin_once(node,timeout_sec=.05)
            assert len(node.edges)==1,'duplicate matching/factors after retransmission'
            assert node.edges[0].sequence_from==key(ka) and node.edges[0].sequence_to==key(kb)
            active=json.loads((archive/'active_session.json').read_text());out=Path(active['output'])
            attempts=(out/'attempts.jsonl').read_text().splitlines();assert len(attempts)==1
            traffic=json.loads((out/'communication.json').read_text());assert traffic['service_attempts']-traffic['service_responses']>=4
            report={'passed':True,'timeout_batches':node.status.request_failures,'matching_attempts':1,'verified_constraints':1,
                    'pnp_inliers':node.edges[0].verification.pnp_inliers,'session_restore':node.status.session!=ka['session'],**traffic}
            (REPO/'.runtime/ros2_reconnect_result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
        except Exception:
            print((root/'robot.log').read_text());raise
        finally:
            if process and process.poll() is None:process.terminate();process.wait(timeout=10)
            log.close();node.destroy_node()
            if rclpy.ok():rclpy.shutdown()

if __name__=='__main__':main()
