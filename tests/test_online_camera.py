"""Camera/image synchronization, optical extrinsics, and causal PGO correction."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from online_camera import CameraPublisher, corrected_body
from test_landmark_visualization import transform


def packet(directory):
    cameras = [dict(width=640, height=480, intrinsics=[400+i,410+i,320.,240.],
                    camera_to_body=transform(Rotation.from_euler('y',.1).as_matrix(), [i*.095,0,0]))
               for i in range(2)]
    return dict(robot='robot', session='session', frame_id=7, timestamp_ns=1234567890,
                body_to_odom=transform(Rotation.from_euler('z',.2).as_matrix(), [1,2,3]),
                cameras=cameras, images=['left.jpg','right.jpg'], _directory=directory)


class FakeRerun:
    ViewCoordinates = type('Coordinates', (), {'RDF':'RDF'})
    def __init__(self): self.logs, self.times = [], {}
    def set_time(self, timeline, **kw): self.times[timeline] = kw
    def log(self, path, *values): self.logs.append((path, values, dict(self.times)))
    def Transform3D(self, **kw): return dict(kind='transform', **kw)
    def AnyValues(self, **kw): return dict(kind='attributes', **kw)
    def Pinhole(self, **kw): return dict(kind='pinhole', **kw)
    def EncodedImage(self, **kw): return dict(kind='image', **kw)
    def Clear(self, **kw): return dict(kind='clear', **kw)


class OnlineCameraTests(unittest.TestCase):
    def test_own_calibration_matched_images_and_full_extrinsic_chain(self):
        with tempfile.TemporaryDirectory() as folder:
            frame = packet(Path(folder))
            rr = FakeRerun()
            publisher = CameraPublisher(rr, 1000000000)
            publisher.publish([frame], dict(revision=0,poses=[]), 4.)
            root = publisher.active['robot']['body']
            body = next(values[0] for path,values,_ in rr.logs if path == root)
            np.testing.assert_allclose(body['translation'], [1,2,3])
            for i, camera in enumerate(frame['cameras']):
                path = root + f'/cam{i}'
                values = next(v for p,v,_ in rr.logs if p == path)
                extrinsic, pinhole = values
                np.testing.assert_allclose(extrinsic['translation'], camera['camera_to_body']['translation'])
                np.testing.assert_allclose(extrinsic['mat3x3'], Rotation.from_euler('y',.1).as_matrix())
                self.assertEqual(pinhole['focal_length'], camera['intrinsics'][:2])
                self.assertEqual(pinhole['resolution'], [640,480])
                self.assertEqual(pinhole['camera_xyz'], 'RDF')
                image = next(v for p,v,_ in rr.logs if p == path+'/image')
                self.assertEqual(image[0]['path'], Path(folder)/frame['images'][i])
                self.assertEqual(image[1]['frame_id'], 7)
            self.assertEqual(rr.times['sensor']['duration'], np.timedelta64(234567890,'ns'))
            self.assertEqual(publisher.image_count, 2)
            # Duplicate and out-of-order delivery must not rewind a camera.
            publisher.publish([frame, dict(frame,frame_id=6)], dict(revision=0,poses=[]), 5.)
            self.assertEqual(publisher.image_count, 2)
            follow, images = publisher.views()
            self.assertEqual((len(follow),len(images)), (1,2))

    def test_pgo_rotation_translation_and_no_future_or_other_session_correction(self):
        frame = packet(Path('/tmp'))
        rotation = Rotation.from_euler('z',.5).as_matrix()
        correction = transform(rotation, [10,0,0])
        pose = dict(key=dict(robot='robot',session='session',id=0),timestamp_ns=1000000000,
                    component=dict(robot='joined',session='map',id=3),map_from_odom=correction)
        c,r,t = corrected_body(frame,[pose])
        self.assertEqual(c,('joined','map',3))
        np.testing.assert_allclose(t,rotation @ [1,2,3] + [10,0,0])
        np.testing.assert_allclose(r,rotation @ Rotation.from_euler('z',.2).as_matrix())
        future=copy.deepcopy(pose);future['timestamp_ns']=2000000000
        other=copy.deepcopy(pose);other['key']['session']='old'
        c,r,t=corrected_body(frame,[future,other])
        np.testing.assert_allclose(t,[1,2,3])
        self.assertEqual(c,('robot','session',0))

    def test_final_graph_moves_held_camera_and_relogs_images_on_component_change(self):
        frame=packet(Path('/tmp'))
        rr=FakeRerun();publisher=CameraPublisher(rr,0)
        publisher.publish([frame],dict(revision=0,poses=[]),0.)
        old=publisher.active['robot']['body']
        graph=dict(revision=1,poses=[dict(key=dict(robot='robot',session='session',id=0),
            timestamp_ns=1000000000,component=dict(robot='joined',session='new',id=0),
            map_from_odom=transform(t=[1,0,0]))])
        publisher.publish([],graph,1.)
        self.assertNotEqual(old,publisher.active['robot']['body'])
        self.assertTrue(any(p==old and v[0]['kind']=='clear' for p,v,_ in rr.logs))
        self.assertEqual(publisher.image_frames,1)
        self.assertEqual(sum(v[0]['kind']=='image' for _,v,_ in rr.logs),4)


if __name__ == '__main__':
    unittest.main()
