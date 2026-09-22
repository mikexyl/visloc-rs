"""Pose-conditioned, five-keyframe TensorRT depth beside the GRACO VIO replay.

The bounded worker owns the native GPU session. It never changes VIO state.
Only successful depth windows establish coverage; all scheduling happens before
inference and uses geometry available at the time of the current keyframe.
"""
from collections import deque
from dataclasses import asdict, dataclass, fields
import hashlib
import json
from pathlib import Path
import queue
import threading
import time
import traceback

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from da3_geometry import Keyframe, CoveredView, align_depth, coverage, resize_intrinsics
from tensorrt_session import Session, DTYPES


@dataclass
class Config:
    engine: str = ''
    library: str = ''
    device: int = 0
    queue_capacity: int = 4
    new_area_threshold: float = .30
    history_depth_tolerance: float = .20
    min_coverage_cells: int = 80
    min_baseline_depth_ratio: float = .005
    min_interval_keyframes: int = 5
    max_windows: int = 256
    max_source_reprojection_px: float = 2.
    max_depth_m: float = 200.
    min_confidence: float = 1.
    min_fit_landmarks: int = 20
    min_holdout_landmarks: int = 5
    min_fit_samples: int = 30
    min_fit_inlier_ratio: float = .55
    max_fit_relative_error: float = .35
    max_holdout_median: float = .20
    max_holdout_p90: float = .40
    cloud_stride: int = 6

    @classmethod
    def from_path(cls, path):
        path = Path(path).resolve()
        data = json.loads(path.read_text())
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f'Unknown DA3 configuration fields: {sorted(unknown)}')
        result = cls(**data)
        for name in ('engine', 'library'):
            value = Path(getattr(result, name))
            if not getattr(result, name):
                raise ValueError(f'DA3 {name} is required')
            setattr(result, name, str((path.parent/value).resolve() if not value.is_absolute() else value))
        result.validate()
        return result

    def validate(self):
        for name in ('queue_capacity', 'min_coverage_cells', 'min_interval_keyframes', 'max_windows',
                     'min_fit_landmarks', 'min_holdout_landmarks', 'min_fit_samples', 'cloud_stride'):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if not 1 <= self.queue_capacity <= 32 or not 5 <= self.min_coverage_cells <= 240:
            raise ValueError('Invalid queue capacity or coverage support')
        if type(self.device) is not int or self.device < 0 or self.max_windows > 1000:
            raise ValueError('Invalid device or window capacity')
        for name in ('new_area_threshold', 'history_depth_tolerance', 'min_fit_inlier_ratio',
                     'max_fit_relative_error', 'max_holdout_median', 'max_holdout_p90'):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f'{name} must be finite and in (0,1]')
        for name in ('min_baseline_depth_ratio', 'max_source_reprojection_px', 'max_depth_m', 'min_confidence'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive and finite')


def keyframe_from_vio(result, gray, calibration, ordinal, config):
    """Join measured left pixels to VIO landmarks, using the camera's lever arm."""
    value = calibration['intrinsics'][0]['intrinsics']
    k = np.array([[value['fx'], 0, value['cx']], [0, value['fy'], value['cy']], [0, 0, 1.]])
    extr = calibration['T_imu_cam'][0]
    r_bc = Rotation.from_quat([extr[x] for x in ('qx', 'qy', 'qz', 'qw')]).as_matrix()
    t_bc = np.array([extr[x] for x in ('px', 'py', 'pz')])
    r_wb = Rotation.from_quat(result['quaternion_xyzw']).as_matrix()
    t_wb = np.asarray(result['position'])
    c2w = np.eye(4)
    c2w[:3, :3] = r_wb @ r_bc
    c2w[:3, 3] = r_wb @ t_bc + t_wb
    points = {int(p[0]): p[1:4] for p in result['map_points']}
    observations = [o for o in result['feature_tracks'] if o[0] == 0 and int(o[1]) in points]
    ids = np.array([o[1] for o in observations], dtype=np.int64)
    pixels = np.array([o[2:4] for o in observations], dtype=float).reshape(-1, 2)
    world = np.array([points[int(i)] for i in ids], dtype=float).reshape(-1, 3)
    pc = (world-c2w[:3, 3]) @ c2w[:3, :3]
    projected = pc @ k.T
    projected = projected[:, :2] / np.maximum(projected[:, 2:], 1e-12)
    h, w = gray.shape
    good = (np.isfinite(world).all(1) & np.isfinite(pixels).all(1)
            & (pc[:, 2] > .1) & (pc[:, 2] < config.max_depth_m)
            & (pixels[:, 0] >= 0) & (pixels[:, 0] < w)
            & (pixels[:, 1] >= 0) & (pixels[:, 1] < h)
            & (np.linalg.norm(projected-pixels, axis=1) <= config.max_source_reprojection_px))
    return Keyframe(int(result['frame_id']), ordinal, int(result['timestamp_ns']), gray.copy(),
                    k, c2w, ids[good], pixels[good], world[good])


def model_inputs(window, size):
    """The exporter normalizes extrinsics inside the graph, exactly once."""
    width, height = size
    images, intrinsics, extrinsics, gray = [], [], [], []
    for frame in window:
        image = cv2.resize(frame.gray, (width, height), interpolation=cv2.INTER_LINEAR)
        gray.append(image)
        rgb = np.repeat(image[None].astype(np.float32)/255., 3, axis=0)
        rgb = (rgb - np.array([.485, .456, .406], np.float32)[:, None, None]) / np.array([.229, .224, .225], np.float32)[:, None, None]
        images.append(rgb)
        h, w = frame.gray.shape
        intrinsics.append(resize_intrinsics(frame.intrinsics, (w, h), size))
        extrinsics.append(np.linalg.inv(frame.camera_to_world))
    inputs = dict(images=np.array(images, np.float32)[None],
                  input_extrinsics=np.array(extrinsics, np.float32)[None],
                  input_intrinsics=np.array(intrinsics, np.float32)[None])
    return inputs, np.asarray(gray)


class Model:
    def __init__(self, config):
        self.session = Session(config.engine, config.library, config.device)
        try:
            tensors = {t['name']: t for t in self.session.tensors if t['input']}
            if set(tensors) != {'images', 'input_extrinsics', 'input_intrinsics'}:
                raise ValueError('DA3 must be a pose-conditioned engine with images, input_extrinsics and input_intrinsics')
            shape = tensors['images']['shape']
            if len(shape) != 5 or shape[:3] != (1, 5, 3) or min(shape[-2:]) < 14:
                raise ValueError(f'A fixed, genuine five-view engine is required; got {shape}')
            if tensors['input_extrinsics']['shape'] != (1, 5, 4, 4) or tensors['input_intrinsics']['shape'] != (1, 5, 3, 3):
                raise ValueError('Unexpected DA3 pose/intrinsic tensor dimensions')
            self.size = (shape[-1], shape[-2])
            self.inputs = tensors
        except BaseException:
            self.session.close()
            raise

    def infer(self, window):
        inputs, gray = model_inputs(window, self.size)
        data = {key: value.astype(DTYPES[self.inputs[key]['dtype']], copy=False) for key, value in inputs.items()}
        started = time.monotonic()
        outputs = self.session.run(data)
        elapsed = (time.monotonic()-started)*1000
        def planes(name):
            array = np.asarray(outputs[name], dtype=np.float32)
            if array.shape == (1, 5, self.size[1], self.size[0], 1):
                array = array[..., 0]
            if array.shape != (1, 5, self.size[1], self.size[0]):
                raise ValueError(f'Unexpected DA3 {name} shape: {array.shape}')
            return array[0]
        return planes('depth'), planes('depth_conf'), inputs, gray, elapsed

    def close(self):
        self.session.close()


class Pipeline:
    def __init__(self, config, model, output, emit):
        self.config, self.model, self.output, self.emit = config, model, Path(output), emit
        self.window, self.history = deque(maxlen=5), []
        self.last_attempt = -10**12
        self.stats = dict(keyframes=0, candidate_windows=0, inferences=0, accepted_windows=0,
                          novelty_skips=0, geometry_skips=0, interval_skips=0, capacity_skips=0,
                          alignment_rejections=0, sequence_resets=0)
        self.output.mkdir(parents=True, exist_ok=True)
        self.events = (self.output/'events.jsonl').open('w')

    def event(self, record):
        self.events.write(json.dumps(record, allow_nan=False)+'\n')
        self.events.flush()

    def accept(self, frame):
        self.stats['keyframes'] += 1
        if self.window and frame.ordinal != self.window[-1].ordinal + 1:
            self.window.clear()
            self.stats['sequence_resets'] += 1
            self.event(dict(event='sequence_reset', ordinal=frame.ordinal, frame_id=frame.frame_id))
        self.window.append(frame)
        if len(self.window) < 5:
            return
        window = list(self.window)
        self.stats['candidate_windows'] += 1
        event = dict(event='window', frame_ids=[f.frame_id for f in window],
                     keyframe_ordinals=[f.ordinal for f in window], timestamp_ns=frame.timestamp_ns,
                     accepted_history_windows=self.stats['accepted_windows'])
        if self.stats['accepted_windows'] >= self.config.max_windows:
            self.stats['capacity_skips'] += 1
            self.event(dict(event, decision='capacity'))
            return
        if frame.ordinal-self.last_attempt < self.config.min_interval_keyframes:
            self.stats['interval_skips'] += 1
            self.event(dict(event, decision='minimum_interval'))
            return
        visible = coverage(window, self.history, self.config.history_depth_tolerance)
        event['coverage'] = visible
        supported_views = sum(v['cells'] >= 8 for v in visible['by_view'])
        if (visible['sampled_cells'] < self.config.min_coverage_cells or supported_views < 3
                or visible['baseline_depth_ratio'] < self.config.min_baseline_depth_ratio):
            self.stats['geometry_skips'] += 1
            self.event(dict(event, decision='insufficient_vio_geometry'))
            return
        if visible['new_fraction'] < self.config.new_area_threshold:
            self.stats['novelty_skips'] += 1
            self.event(dict(event, decision='already_covered'))
            return
        self.last_attempt = frame.ordinal
        self.stats['inferences'] += 1
        depth, confidence, inputs, gray, inference_ms = self.model.infer(window)
        if self.stats['inferences'] == 1:
            np.savez_compressed(self.output/'first_inference.npz', **inputs,
                                depth=depth, depth_conf=confidence,
                                frame_ids=np.array([f.frame_id for f in window]))
        scale, alignment = align_depth(window, depth, confidence, self.config)
        event.update(inference_ms=inference_ms, alignment=alignment)
        if scale is None:
            self.stats['alignment_rejections'] += 1
            self.event(dict(event, decision='alignment_rejected'))
            return
        aligned = depth * scale
        valid = (np.isfinite(aligned) & (aligned > .1) & (aligned < self.config.max_depth_m)
                 & np.isfinite(confidence) & (confidence >= self.config.min_confidence))
        aligned = np.where(valid, aligned, np.nan).astype(np.float32)
        if np.mean(valid) < .10:
            self.stats['alignment_rejections'] += 1
            self.event(dict(event, decision='too_little_valid_depth'))
            return
        number = self.stats['accepted_windows']
        archive = self.output/f'window_{number:04d}.npz'
        np.savez_compressed(archive, raw_depth=depth, depth_m=aligned, confidence=confidence,
                            gray=gray, intrinsics=inputs['input_intrinsics'][0],
                            input_world_to_camera=inputs['input_extrinsics'][0],
                            camera_to_world=np.array([f.camera_to_world for f in window]),
                            frame_ids=np.array([f.frame_id for f in window]),
                            timestamps_ns=np.array([f.timestamp_ns for f in window]),
                            keyframe_ordinals=np.array([f.ordinal for f in window]),
                            scale=np.array(scale),
                            metadata=np.array(json.dumps(event, allow_nan=False)))
        for i, f in enumerate(window):
            h, w = aligned[i].shape
            small_size = (max(2, w//5), max(2, h//5))
            small = cv2.resize(aligned[i], small_size, interpolation=cv2.INTER_NEAREST_EXACT)
            self.history.append(CoveredView(np.linalg.inv(f.camera_to_world),
                resize_intrinsics(inputs['input_intrinsics'][0, i], (w, h), small_size), small))
        self.stats['accepted_windows'] += 1
        event.update(decision='accepted', archive=str(archive), window_index=number)
        self.event(event)
        self.emit(event)

    def close(self):
        self.events.close()


class OnlineDepth:
    def __init__(self, config, output):
        config.validate()
        self.config, self.output = config, Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.queue = queue.Queue(config.queue_capacity)
        self.results = queue.Queue()
        self.stop, self.ready = threading.Event(), threading.Event()
        self.error, self.stats, self.dropped, self.submitted = None, {}, 0, 0
        self.thread = threading.Thread(target=self._run, name='da3-tensorrt', daemon=True)
        self.thread.start()
        if not self.ready.wait(60):
            self.stop.set()
            raise RuntimeError('DA3 initialization timed out')
        self.check()

    def _run(self):
        model = pipeline = None
        try:
            model = Model(self.config)
            (self.output/'config.json').write_text(json.dumps(asdict(self.config), indent=2)+'\n')
            manifest = dict(engine_sha256=hashlib.sha256(Path(self.config.engine).read_bytes()).hexdigest(),
                            engine=self.config.engine, tensors=model.session.tensors,
                            intrinsics='calibrated, scaled with resize pixel centers',
                            poses='VIO world-to-camera; T_world_body @ T_body_camera inverted',
                            extrinsic_normalization='official DA3 normalization inside exported graph',
                            scale_source='robust fit to VIO landmark camera-z; held-out track IDs',
                            inference='native TensorRT bridge, no PyTorch or CPU fallback')
            (self.output/'model.json').write_text(json.dumps(manifest, indent=2)+'\n')
            pipeline = Pipeline(self.config, model, self.output, self.results.put)
            self.ready.set()
            while not self.stop.is_set() or not self.queue.empty():
                try:
                    frame = self.queue.get(timeout=.05)
                except queue.Empty:
                    continue
                pipeline.accept(frame)
                self.stats = pipeline.stats.copy()
        except BaseException:
            self.error = traceback.format_exc()
            (self.output/'error.log').write_text(self.error)
        finally:
            if pipeline:
                pipeline.close()
            if model:
                model.close()
            self.ready.set()

    def check(self):
        if self.error:
            raise RuntimeError(f'DA3 worker failed:\n{self.error}')

    def submit(self, frame):
        self.check()
        if self.stop.is_set():
            raise RuntimeError('DA3 worker already stopped')
        self.submitted += 1
        try:
            self.queue.put_nowait(frame)
        except queue.Full:
            self.dropped += 1

    def poll(self):
        self.check()
        results = []
        while True:
            try:
                results.append(self.results.get_nowait())
            except queue.Empty:
                return results

    def finish(self):
        self.stop.set()
        self.thread.join(timeout=60)
        if self.thread.is_alive():
            raise RuntimeError('DA3 worker did not finish within 60 seconds')
        self.check()
        status = dict(self.stats, submitted_keyframes=self.submitted, dropped_keyframes=self.dropped)
        (self.output/'summary.json').write_text(json.dumps(status, indent=2)+'\n')
        return status

    def abort(self):
        self.stop.set()
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        self.thread.join(timeout=10)


def log_result(event, start_ns, cloud_stride):
    """Called only by the replay thread, keeping Rerun timelines deterministic."""
    import rerun as rr
    with np.load(event['archive'], allow_pickle=False) as data:
        depth, gray, intrinsics, c2w = data['depth_m'], data['gray'], data['intrinsics'], data['camera_to_world']
        rr.set_time('elapsed', duration=(event['timestamp_ns']-start_ns)/1e9)
        rr.log('da3/depth', rr.DepthImage(depth[-1], meter=1.))
        rr.log('da3/input', rr.Image(gray[-1]))
        rr.log('da3/status', rr.TextDocument(
            f"Five VIO keyframes: {event['frame_ids']}\n"
            f"New sampled area: {event['coverage']['new_fraction']:.1%}\n"
            f"Landmark depth scale: {event['alignment']['scale']:.4f}\n"
            f"Held-out median relative error: {event['alignment']['holdout_relative_median']:.1%}"))
        cloud, colors = [], []
        h, w = depth.shape[1:]
        y, x = np.mgrid[0:h:cloud_stride, 0:w:cloud_stride]
        for i in range(5):
            z = depth[i, y, x]
            good = np.isfinite(z) & (z > 0)
            k = intrinsics[i]
            pc = np.column_stack(((x[good]-k[0, 2])*z[good]/k[0, 0],
                                  (y[good]-k[1, 2])*z[good]/k[1, 1], z[good]))
            cloud.append(pc @ c2w[i, :3, :3].T + c2w[i, :3, 3])
            colors.append(np.repeat(gray[i, y, x][good, None], 3, axis=1))
        rr.log(f"world/da3/window_{event['window_index']:04d}",
               rr.Points3D(np.concatenate(cloud), colors=np.concatenate(colors), radii=.025))
        for name, value in [('new_area_fraction', event['coverage']['new_fraction']),
                            ('depth_scale', event['alignment']['scale']),
                            ('held_out_relative_error', event['alignment']['holdout_relative_median']),
                            ('inference_ms', event['inference_ms'])]:
            rr.log('metrics/da3/'+name, rr.Scalars(value))
