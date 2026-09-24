#!/usr/bin/env python3
"""Test the real Rust sensor queue and estimator-session restart, without GPU."""
import argparse,array,json,os,subprocess,tempfile,time
from pathlib import Path
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile,ReliabilityPolicy,DurabilityPolicy
from sensor_msgs.msg import Image,Imu
from visloc_msgs.msg import Status
from visloc_msgs.srv import GetHistory,Finish

REPO=Path(__file__).resolve().parents[1]

def stamp(t,ns):t.sec,t.nanosec=divmod(ns,10**9)

class Fixture(Node):
    def __init__(self):
        super().__init__('visloc_sensor_contract_test');self.status=None
        self.image_pub=self.create_publisher(Image,'/queue_test/camera/image',32)
        self.imu_pub=self.create_publisher(Imu,'/queue_test/imu',2048)
        qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.sub=self.create_subscription(Status,'/queue_test/slam/status',lambda s:setattr(self,'status',s),qos)
        self.history=self.create_client(GetHistory,'/queue_test/slam/history');self.finish=self.create_client(Finish,'/queue_test/slam/finish')
    def until(self,f,reason,timeout=25):
        started=time.monotonic()
        while not f():
            rclpy.spin_once(self,timeout_sec=.02)
            if self.status and self.status.error:raise RuntimeError(self.status.error)
            if time.monotonic()-started>timeout:raise TimeoutError(reason)
    def call(self,client,request):
        self.until(lambda:client.service_is_ready(),'service discovery');future=client.call_async(request)
        self.until(future.done,'response');return future.result()
    def image(self,ns):
        m=Image(height=550,width=800,encoding='mono8',step=800,data=array.array('B',[0])*(800*550))
        stamp(m.header.stamp,ns);self.image_pub.publish(m)
    def imu(self,ns):
        m=Imu();stamp(m.header.stamp,ns);m.linear_acceleration.z=9.81;self.imu_pub.publish(m)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('robot_config',type=Path);a=p.parse_args()
    if os.environ.get('ROS_DOMAIN_ID')!='220':raise SystemExit('Use isolated ROS_DOMAIN_ID=220')
    rclpy.init();node=Fixture();process=None
    with tempfile.TemporaryDirectory(prefix='visloc-robot-test-') as directory:
        root=Path(directory);cfg=json.loads(a.robot_config.read_text())
        # This queue/restart fixture uses blank images and deliberately missing
        # sensor coverage. Explicitly bypass stationarity validation; real-image
        # default-startup parity is exercised by the GRACO replay regression.
        cfg.pop('scalar_mode', None)
        cfg.update(robot='queue_test',peers=['queue_test'],output=str(root/'robot'),preprocess=None,reliable_sensors=True,loop_enabled=False,imu_startup=None)
        config=root/'config.json';config.write_text(json.dumps(cfg));log=(root/'robot.log').open('w')
        binary=Path(os.environ.get('VISLOC_ROS_INSTALL',REPO/'.runtime/ros2_install'))/'visloc_ros/lib/visloc_ros/robot'
        def launch():return subprocess.Popen([str(binary)],env=dict(os.environ,VISLOC_ROBOT_CONFIG=str(config)),stdout=log,stderr=subprocess.STDOUT)
        try:
            process=launch();node.until(lambda:node.status is not None and node.status.ready,'initial readiness')
            node.until(lambda:node.image_pub.get_subscription_count()>0 and node.imu_pub.get_subscription_count()>0,'sensor discovery')
            for i in range(48):
                node.image(10**9+i*50_000_000);rclpy.spin_once(node,timeout_sec=.005)
            node.until(lambda:node.status.dropped_images==16,'bounded image buffer')
            assert node.status.frames==0
            for ns in range(990_000_000,3_360_000_000,5_000_000):node.imu(ns)
            node.until(lambda:node.status.frames==32,'remaining ordered camera frames')
            assert node.call(node.finish,Finish.Request(last_image_timestamp_ns=3_350_000_000)).accepted
            node.until(lambda:node.status.finished,'drain')
            old=node.status.session;history=node.call(node.history,GetHistory.Request(include_keyframes=True,include_loops=True));count=len(history.keyframes)
            assert count>0 and all(k.key.session==old for k in history.keyframes)
            process.terminate();process.wait(timeout=10);process=launch()
            node.until(lambda:node.status.session!=old and node.status.ready,'new estimator session')
            new=node.status.session;history=node.call(node.history,GetHistory.Request(include_keyframes=True,include_loops=True))
            assert history.session==new and len(history.keyframes)==count
            assert all(k.key.session==old for k in history.keyframes)
            node.until(lambda:node.image_pub.get_subscription_count()>0 and node.imu_pub.get_subscription_count()>0,'restart sensor discovery')
            node.imu(4_995_000_000);node.imu(5_005_000_000);node.image(5_000_000_000)
            node.until(lambda:node.status.frames==1,'new session frame')
            history=node.call(node.history,GetHistory.Request(include_keyframes=True,include_loops=True))
            assert len(history.keyframes)==count+1
            assert {k.key.session for k in history.keyframes if k.key.id==0}=={old,new}
            report={'passed':True,'input_images':48,'retained_images':32,'dropped_images':16,'archived_keyframes':count,'new_session_keyframes':1,'session_ids_distinct':old!=new}
            (REPO/'.runtime/ros2_robot_integration_result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
        except Exception:
            print((root/'robot.log').read_text());raise
        finally:
            if process and process.poll() is None:process.terminate();process.wait(timeout=10)
            log.close();node.destroy_node()
            if rclpy.ok():rclpy.shutdown()

if __name__=='__main__':main()
