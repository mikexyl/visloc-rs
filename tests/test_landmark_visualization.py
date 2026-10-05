"""Fixed-pose display-map geometry, namespace isolation, and quality gates."""
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from landmark_visualization import build_landmark_map, identity


def transform(r=None, t=None):
    return dict(rotation_xyzw=Rotation.from_matrix(np.eye(3) if r is None else r).as_quat().tolist(),
                translation=[0., 0., 0.] if t is None else list(t))


def scene(count=4, robot='a', session='first', component=None, baseline=.5):
    point = np.array([.3, .1, 5.])
    extrinsic_r = Rotation.from_euler('xyz', [.02, .05, -.03]).as_matrix()
    extrinsic_t = np.array([.08, -.01, .02])
    features, poses = [], {}
    component = component or dict(robot=robot, session=session, id=0)
    for i in range(count):
        key = dict(robot=robot, session=session, id=i)
        r = Rotation.from_euler('y', i * .01).as_matrix()
        t = np.array([baseline*i, 0., 0.])
        intrinsics = np.array([400.+i*15, 410.+i*10, 320., 240.])
        pc = extrinsic_r.T @ (r.T @ (point-t)-extrinsic_t)
        uv = pc[:2]/pc[2]*intrinsics[:2]+intrinsics[2:]
        camera = dict(intrinsics=intrinsics.tolist(), width=640, height=480,
                      camera_to_body=transform(extrinsic_r, extrinsic_t))
        # Biased historical depth estimates: correct viewing rays, wrong surfaces.
        features.append(dict(key=key, camera=camera, pixels=[uv.tolist()],
                             track_ids=[42], points_camera=[(pc*(1+.08*i)).tolist()]))
        poses[identity(key)] = dict(key=key, component=component, body_to_map=transform(r,t))
    return features, poses, point


class LandmarkVisualizationTests(unittest.TestCase):
    def test_one_refined_point_uses_corrected_poses_and_each_camera_calibration(self):
        features, poses, truth = scene()
        points, audit = build_landmark_map(features, poses)
        self.assertEqual(len(points), 1)
        np.testing.assert_allclose(points[0]['position'], truth, atol=1e-7)
        self.assertLess(points[0]['reprojection_px'], 1e-7)
        self.assertEqual(points[0]['inliers'], 4)
        self.assertEqual(audit['archived_metric_estimates'], 4)

    def test_bad_view_is_rejected_without_averaging_its_depth(self):
        features, poses, truth = scene(5)
        features[-1]['pixels'][0][0] += 45
        features[-1]['points_camera'][0] = [100., 40., 80.]
        points, _ = build_landmark_map(features, poses)
        self.assertEqual(points[0]['inliers'], 4)
        np.testing.assert_allclose(points[0]['position'], truth, atol=1e-7)

    def test_duplicate_archives_do_not_count_as_independent_views(self):
        features, poses, _ = scene(1)
        points, audit = build_landmark_map(features*4, poses)
        self.assertEqual(points, [])
        self.assertEqual(audit['duplicate_observations'], 3)
        self.assertEqual(audit['rejected']['insufficient_observations'], 1)

    def test_robot_session_and_disconnected_component_ids_never_merge(self):
        packets, graph = [], {}
        for robot, session, component in [('a','one',0),('b','one',0),('a','two',0)]:
            f,p,_=scene(2,robot,session,dict(robot='root',session='map',id=component))
            packets.extend(f);graph.update(p)
        # Same local identity split across components must not triangulate across them.
        f,p,_=scene(2,'c','one')
        for i,pose in enumerate(p.values()):pose['component']=dict(robot='root',session='map',id=i)
        packets.extend(f);graph.update(p)
        points,audit=build_landmark_map(packets,graph)
        self.assertEqual(len(points),3)
        self.assertEqual(audit['rejected']['insufficient_observations'],2)

    def test_low_parallax_and_single_views_are_hidden(self):
        features,poses,_=scene(3,baseline=1e-5)
        points,audit=build_landmark_map(features,poses)
        self.assertEqual(points,[])
        self.assertEqual(audit['rejected']['low_parallax'],1)

    def test_missing_metric_points_can_support_an_existing_landmark(self):
        features,poses,truth=scene(3)
        features[0]['points_camera'][0]=None
        features[1]['points_camera'][0]=None
        points,_=build_landmark_map(features,poses)
        np.testing.assert_allclose(points[0]['position'],truth,atol=1e-7)
        features[2]['points_camera'][0]=None
        points,audit=build_landmark_map(features,poses)
        self.assertEqual(points,[])
        self.assertEqual(audit['nonmetric_tracks_omitted'],1)
        self.assertEqual(audit['landmark_groups'],0)

    def test_invalid_input_and_missing_graph_pose_do_not_create_points(self):
        features,poses,_=scene(3)
        features[0]['pixels'][0]=[float('nan'),1.]
        features[1]['points_camera'][0]=[0,0,-5]
        del poses[identity(features[2]['key'])]
        points,audit=build_landmark_map(features,poses)
        self.assertEqual(points,[])
        self.assertEqual(audit['invalid_pixels'],1)
        self.assertEqual(audit['invalid_metric_estimates'],1)
        self.assertEqual(audit['missing_graph_pose'],1)

    def test_identity_order_is_deterministic_and_bad_arrays_fail(self):
        features,poses,_=scene()
        a,_=build_landmark_map(features,poses)
        b,_=build_landmark_map(reversed(features),poses)
        np.testing.assert_array_equal(a[0]['position'],b[0]['position'])
        features[0]['track_ids']=[]
        with self.assertRaisesRegex(ValueError,'misaligned'):
            build_landmark_map(features,poses)

    def test_backward_intersection_and_invalid_settings_are_rejected(self):
        features,poses,_=scene(3)
        behind=np.array([.3,.1,-5.])
        for feature in features:
            pose=poses[identity(feature['key'])]
            rb=Rotation.from_quat(pose['body_to_map']['rotation_xyzw'])
            cam=feature['camera'];rc=Rotation.from_quat(cam['camera_to_body']['rotation_xyzw'])
            pc=rc.inv().apply(rb.inv().apply(behind-np.array(pose['body_to_map']['translation']))-cam['camera_to_body']['translation'])
            intr=np.array(cam['intrinsics'])
            feature['pixels'][0]=(pc[:2]/pc[2]*intr[:2]+intr[2:]).tolist()
        points,audit=build_landmark_map(features,poses)
        self.assertEqual(points,[])
        self.assertEqual(audit['rejected']['inconsistent_reprojection'],1)
        with self.assertRaises(ValueError):
            build_landmark_map(features,poses,min_observations=1)
        with self.assertRaises(ValueError):
            build_landmark_map(features,poses,min_parallax_deg=float('nan'))


if __name__=='__main__': unittest.main()
