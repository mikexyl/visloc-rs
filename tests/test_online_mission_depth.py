"""Live DA3 uses causal, filtered keyframes and keeps map corrections display-only."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from online_da3 import Config
from online_mission_depth import FilteredPipeline, MissionDepth, DensePublisher, journal_keyframe
from test_online_da3 import FakeModel
from test_landmark_visualization import transform
from test_online_camera import FakeRerun


def item(directory, i, spacing=.2):
    key = dict(robot='r', session='s', id=i)
    camera = dict(width=80, height=60, intrinsics=[60.,60.,39.5,29.5], camera_to_body=transform())
    y,x=np.mgrid[5:60:10,5:80:10]
    world=np.column_stack(((x.ravel()-39.5)/6,(y.ravel()-29.5)/6,np.full(x.size,10.)))
    local=world-[i*spacing,0,0]
    pixels=local[:,:2]/local[:,2:]*60+[39.5,29.5]
    feature=dict(key=key,timestamp_ns=i*100000000, camera=camera,
                 track_ids=list(range(x.size)),pixels=pixels.tolist(),points_camera=local.tolist())
    pose=dict(key=key,timestamp_ns=feature['timestamp_ns'],body_to_odom=transform(t=[i*spacing,0,0]))
    name=f'{i}.png'
    cv2.imwrite(str(directory/name),np.full((60,80),100,np.uint8))
    frame=dict(robot='r',session='s',frame_id=i*7,timestamp_ns=feature['timestamp_ns'],
               images=[name],cameras=[camera],_directory=directory,body_to_odom=pose['body_to_odom'])
    return dict(feature=feature,pose=pose),frame


class MissionDepthTests(unittest.TestCase):
    def test_keyframe_uses_its_own_calibration_and_camera_to_body_transform(self):
        with tempfile.TemporaryDirectory() as folder:
            packet,camera=item(Path(folder),2)
            rb=Rotation.from_euler('z',.4).as_matrix()
            rc=Rotation.from_euler('y',-.2).as_matrix()
            packet['pose']['body_to_odom']=transform(rb,[3,4,5])
            packet['feature']['camera']['camera_to_body']=transform(rc,[.1,.2,.3])
            packet['feature']['camera']['intrinsics']=[70.,65.,38.,28.]
            frame=journal_keyframe(packet,camera)
            np.testing.assert_allclose(frame.camera_to_world[:3,:3],rb@rc)
            np.testing.assert_allclose(frame.camera_to_world[:3,3],rb@[.1,.2,.3]+[3,4,5])
            np.testing.assert_array_equal(frame.intrinsics,[[70,0,38],[0,65,28],[0,0,1]])

    def test_filter_rejects_inconsistent_tracks_and_outlying_observations_before_coverage(self):
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder)
            items=[item(directory,i) for i in range(5)]
            for i,(packet,_) in enumerate(items):
                packet['feature']['pixels'][0]=[10.+i*10,40. if i%2 else 10.]
                packet['feature']['track_ids'][-1]=2**64-1
            items[0][0]['feature']['pixels'][1]=[75.,55.]  # Bad observation of a supported track.
            frames=[journal_keyframe(*entry) for entry in items]
            pipeline=FilteredPipeline(Config(landmark_alignment=False),FakeModel(),directory/'out',lambda _:None)
            pipeline.owner=('r','s')
            filtered,report=pipeline.prepare_window(frames)
            try:
                self.assertTrue(all(0 not in f.track_ids for f in filtered))
                self.assertNotIn(1,filtered[0].track_ids)
                self.assertIn(1,filtered[-1].track_ids)
                self.assertIn(2**64-1,filtered[-1].track_ids)
                self.assertGreater(report['landmark_filter']['rejected']['inconsistent_reprojection'],0)
                for f in filtered:
                    np.testing.assert_allclose(f.points_world[:,2],10.,atol=1e-7)
                    pc=f.points_camera
                    np.testing.assert_allclose(pc[:,:2]/pc[:,2:]*60+[39.5,29.5],f.pixels,atol=1e-7)
                np.testing.assert_array_equal(frames[0].track_ids,[])  # No mutation/raw fallback.
            finally:
                pipeline.close()

    def test_low_parallax_and_missing_metric_seeds_prevent_inference(self):
        for mode in ('low_parallax','no_metric'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                directory=Path(folder);model=FakeModel()
                pipeline=FilteredPipeline(Config(landmark_alignment=False),model,directory/'out',lambda _:None)
                try:
                    for i in range(5):
                        packet,camera=item(directory,i,spacing=0 if mode=='low_parallax' else .2)
                        if mode=='no_metric': packet['feature']['points_camera']=[None]*48
                        pipeline.accept((packet,camera))
                    self.assertEqual(model.calls,0)
                    self.assertEqual(pipeline.stats['geometry_skips'],1)
                finally: pipeline.close()

    def test_exact_camera_join_and_input_pose_only_archive(self):
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder);events=[]
            pipeline=FilteredPipeline(Config(landmark_alignment=False),FakeModel(),directory/'out',events.append)
            try:
                for i in range(5): pipeline.accept(item(directory,i))
                self.assertEqual(len(events),1)
                self.assertEqual(events[0]['alignment']['method'],'input_poses')
                self.assertEqual(events[0]['frame_ids'],[0,7,14,21,28])
                with np.load(events[0]['archive']) as data:
                    self.assertGreater(len(data['landmark_ids']),100)
                    np.testing.assert_allclose(data['camera_to_world'][:,0,3],np.arange(5)*.2)
                    np.testing.assert_allclose(data['depth_m'],5.)
                packet,camera=item(directory,6)
                camera['timestamp_ns']+=1
                with self.assertRaises(ValueError): journal_keyframe(packet,camera)
                packet,camera=item(directory,6);packet['feature']['key']['session']='different'
                with self.assertRaises(ValueError): pipeline.accept((packet,camera))
            finally: pipeline.close()

    def test_delayed_images_duplicates_queue_bounds_and_missing_ordinals(self):
        class Worker:
            def __init__(self,*args,**kwargs): self.items=[]
            def submit(self,entry): self.items.append(entry)
            def finish(self): return dict(submitted=len(self.items))
            def poll(self): return []
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder)
            depth=MissionDepth(Config(),directory/'out',{},worker_factory=Worker)
            packet,camera=item(directory,0)
            depth.ingest([packet],[])
            self.assertFalse(depth.workers)
            depth.ingest([], [camera])
            depth.ingest([packet],[camera])
            self.assertEqual(len(depth.workers[('r','s')].items),1)
            for i in range(1,100):
                p,c=item(directory,i);depth.ingest([p],[])
            self.assertEqual(len(depth.pending),64)
            self.assertEqual(depth.counts['missing_image_drops'],35)
            summary=depth.finish()
            self.assertEqual(summary['join']['missing_image_drops'],99)
            # A missing actual keyframe clears the inference window.
            pipeline=FilteredPipeline(Config(),FakeModel(),directory/'gap',lambda _:None)
            try:
                for i in (0,1,3,4,5): pipeline.accept(item(directory,i))
                self.assertEqual(pipeline.stats['sequence_resets'],1)
                self.assertEqual(pipeline.stats['candidate_windows'],0)
            finally: pipeline.close()

    def test_dense_cloud_moves_with_pgo_without_changing_camera_samples(self):
        class Rerun(FakeRerun):
            def Points3D(self,positions,**kw): return dict(kind='points',positions=positions,**kw)
            def DepthImage(self,depth,**kw): return dict(kind='depth',depth=depth,**kw)
            def TextDocument(self,text): return text
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder);events=[]
            pipeline=FilteredPipeline(Config(landmark_alignment=False),FakeModel(),directory/'out',events.append)
            try:
                for i in range(5): pipeline.accept(item(directory,i))
            finally: pipeline.close()
            rr=Rerun();publisher=DensePublisher(rr,6)
            publisher.publish(events,dict(revision=0,poses=[]))
            before=publisher.points
            rr.logs=[]
            poses=[dict(key=k,component=dict(robot='r',session='s',id=0),
                        body_to_map=transform(t=[i*.2+10,0,0])) for i,k in enumerate(events[0]['keyframe_keys'])]
            publisher.publish([],dict(revision=1,poses=poses))
            self.assertEqual(publisher.points,before)
            transforms=[v[0] for _,v,_ in rr.logs if isinstance(v[0],dict) and v[0].get('kind')=='transform']
            self.assertEqual(len(transforms),5)
            np.testing.assert_allclose([v['translation'][0] for v in transforms],np.arange(5)*.2+10)
            self.assertFalse(any(path.endswith('/points') for path,_,_ in rr.logs))


if __name__=='__main__': unittest.main()
