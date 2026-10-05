"""Matched live camera images, calibrated optical frames, and PGO corrections."""
import numpy as np

from landmark_visualization import _transform, identity


def corrected_body(frame, poses):
    """Use the newest available correction at/before this exposure, in its session."""
    candidates = [p for p in poses if (p['key']['robot'], p['key']['session']) ==
                  (frame['robot'], frame['session']) and p['timestamp_ns'] <= frame['timestamp_ns']]
    r, t = _transform(frame['body_to_odom'])
    component = (frame['robot'], frame['session'], 0)
    if candidates:
        pose = max(candidates, key=lambda p:p['timestamp_ns'])
        component = identity(pose['component'])
        if 'map_from_odom' in pose:
            rc, tc = _transform(pose['map_from_odom'])
            r, t = rc @ r, rc @ t + tc
    return component, r, t


class CameraPublisher:
    def __init__(self, rr, epoch_ns):
        self.rr, self.epoch_ns = rr, epoch_ns
        self.latest = {}
        self.active = {}
        self.calibration = {}
        self.graph_revision = None
        self.image_frames = 0
        self.image_count = 0

    def _log(self, frame, graph, elapsed, images=True):
        rr = self.rr
        rr.set_time('live', duration=elapsed)
        rr.set_time('sensor', duration=np.timedelta64(frame['timestamp_ns'] - self.epoch_ns, 'ns'))
        component, r, t = corrected_body(frame, graph['poses'])
        c = component
        root = f"components/{c[0]}_{c[1]}_{c[2]}/{frame['robot']}/camera/{frame['session']}/body"
        old = self.active.get(frame['robot'])
        moved = old is not None and old['body'] != root
        if moved:
            rr.log(old['body'], rr.Clear(recursive=True))
        rr.log(root, rr.Transform3D(translation=t, mat3x3=r),
               rr.AnyValues(frame_id=frame['frame_id'], timestamp_ns=str(frame['timestamp_ns'])))
        paths = []
        for i, camera in enumerate(frame['cameras']):
            path = f'{root}/cam{i}'
            paths.append(path)
            if self.calibration.get(path) != camera or moved:
                rc, tc = _transform(camera['camera_to_body'])
                rr.log(path, rr.Transform3D(translation=tc, mat3x3=rc),
                       rr.Pinhole(focal_length=camera['intrinsics'][:2],
                                  principal_point=camera['intrinsics'][2:],
                                  resolution=[camera['width'], camera['height']],
                                  camera_xyz=rr.ViewCoordinates.RDF, image_plane_distance=.4))
                self.calibration[path] = camera
            if images or moved:
                directory = frame['_directory'].resolve()
                source = (directory / frame['images'][i]).resolve()
                if not source.is_relative_to(directory):
                    raise ValueError('camera image must be inside its session directory')
                rr.log(path + '/image', rr.EncodedImage(path=source),
                       rr.AnyValues(frame_id=frame['frame_id'], timestamp_ns=str(frame['timestamp_ns'])))
        self.active[frame['robot']] = dict(body=root, cameras=paths, component=component,
                                         timestamp_ns=frame['timestamp_ns'])

    def publish(self, frames, graph, elapsed):
        updated = set()
        for frame in frames:
            if len(frame['images']) != len(frame['cameras']):
                raise ValueError('camera images/calibration count mismatch')
            previous = self.latest.get(frame['robot'])
            if (previous and previous['session'] == frame['session']
                    and frame['frame_id'] <= previous['frame_id']):
                continue
            self._log(frame, graph, elapsed)
            self.latest[frame['robot']] = frame
            updated.add(frame['robot'])
            self.image_frames += 1
            self.image_count += len(frame['images'])
        if graph['revision'] != self.graph_revision:
            # The robot may be stationary or replay may have drained when PGO
            # finishes. Move the last camera with the new map even without a new image.
            for robot, frame in self.latest.items():
                if robot not in updated:
                    self._log(frame, graph, elapsed, images=False)
            self.graph_revision = graph['revision']

    def views(self):
        import rerun.blueprint as rrb
        follow, images = [], []
        for robot, state in sorted(self.active.items()):
            c = state['component']
            origin = f'components/{c[0]}_{c[1]}_{c[2]}'
            follow.append(rrb.Spatial3DView(origin=origin, name=f'Follow camera: {robot}',
                eye_controls=rrb.EyeControls3D(tracking_entity=state['cameras'][0])))
            for i, path in enumerate(state['cameras']):
                images.append(rrb.Spatial2DView(origin=path, name=f'{robot}: {"left" if i == 0 else "right"}'))
        return follow, images
