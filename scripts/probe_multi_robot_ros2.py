#!/usr/bin/env python3
"""Read-only interoperability probe against running native Rust robot services."""
import argparse,json,time
from pathlib import Path
import numpy as np
import rclpy
from rclpy.node import Node
from visloc_msgs.srv import GetHistory, GetSequence, GetFeatures

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mission',type=Path);a=p.parse_args()
    mission=json.loads(a.mission.read_text());rclpy.init();node=Node('visloc_interface_probe');report={}
    def call(client,req):
        if not client.wait_for_service(timeout_sec=10):raise TimeoutError(client.srv_name)
        future=client.call_async(req);rclpy.spin_until_future_complete(node,future,timeout_sec=5)
        if not future.done():raise TimeoutError(client.srv_name)
        return future.result()
    try:
        for robot in mission['robots']:
            r=robot['robot'];h=node.create_client(GetHistory,f'/{r}/slam/history')
            seq=node.create_client(GetSequence,f'/{r}/slam/sequence_frames');features=node.create_client(GetFeatures,f'/{r}/slam/features')
            history=call(h,GetHistory.Request(include_keyframes=True,include_loops=True))
            entry={'session':history.session,'keyframes_in_page':len(history.keyframes),'sequences_in_page':len(history.sequences)}
            if history.sequences:
                s=history.sequences[0];matrix=call(seq,GetSequence.Request(key=s.key))
                assert matrix.found and len(matrix.descriptors.descriptors)==2560
                desc=np.array(matrix.descriptors.descriptors).reshape(5,512);assert np.allclose(np.linalg.norm(desc,axis=1),1,atol=1e-3)
                feature=call(features,GetFeatures.Request(key=matrix.descriptors.frames[0]))
                assert feature.found and len(feature.frame.features) in (0,128)
                entry.update(frame_matrix=[5,512],feature_count=len(feature.frame.features),model_id=s.model_id)
            report[r]=entry
        (a.mission.parent/'ros_interoperability.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    finally:node.destroy_node();rclpy.shutdown()

if __name__=='__main__':main()
