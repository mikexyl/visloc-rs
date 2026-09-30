"""Common read-only input for GRACO bags and prepared RealSense recordings."""
from pathlib import Path
import sqlite3
from types import SimpleNamespace
import cv2
import numpy as np


class SensorSource:
    def image_at(self, timestamp_ns):
        if not hasattr(self, '_rows'):
            self._rows = {t: row for t, row, _ in self.frames}
        return self.image(self._rows[timestamp_ns])


class GracoSource(SensorSource):
    def __init__(self, robot):
        from run_graco_vio import Bag, load_imu
        self.bag = Bag(Path(robot['bag']))
        self.frames = self.bag.stereo_index()[:robot['frames']]
        self.imu, self.times = load_imu(self.bag)

    def image(self, row):
        tid, _ = self.bag.topics['/camera_left/image_raw']
        data = self.bag.connection.execute('SELECT data FROM messages WHERE id=? AND topic_id=?', (row, tid)).fetchone()[0]
        return self.bag.types.deserialize_cdr(data, 'sensor_msgs/msg/Image')

    def close(self):
        self.bag.connection.close()


class RealSenseSource(SensorSource):
    def __init__(self, robot):
        root = Path(robot['sensor_cache'])
        self.connection = sqlite3.connect((root/'images.sqlite3').resolve().as_uri()+'?mode=ro', uri=True)
        stereo = robot.get('camera_mode', 'mono') == 'stereo'
        query = ('SELECT timestamp_ns FROM images INNER JOIN right_images USING(timestamp_ns)'
                 if stereo else 'SELECT timestamp_ns FROM images')
        # The prepared mission may select only the interval covered by IMU.
        # Keep excluded images in the immutable cache without replaying them.
        conditions, parameters = [], []
        for name, operator in [('original_first_ns', '>='), ('original_last_ns', '<=')]:
            if name in robot:
                conditions.append(f'timestamp_ns {operator} ?')
                parameters.append(int(robot[name]))
        if conditions:
            query += ' WHERE ' + ' AND '.join(conditions)
        self.frames = [(t, t, t if stereo else None) for (t,) in self.connection.execute(
            query+' ORDER BY timestamp_ns LIMIT ?', (*parameters, robot['frames']))]
        with np.load(root/'imu.npz') as source:
            self.times = source['timestamps'].copy()
            values = source['values'].copy()
        self.imu = [[int(t), *v.tolist()] for t, v in zip(self.times, values)]
        if not self.frames or self.times.dtype != np.int64 or np.any(np.diff(self.times) <= 0):
            raise ValueError('Invalid prepared RealSense timestamp sequence')

    def image(self, row):
        return self._image(row, 'images')

    def right_image(self, row):
        return self._image(row, 'right_images')

    def _image(self, row, table):
        data = self.connection.execute(f'SELECT data FROM {table} WHERE timestamp_ns=?', (row,)).fetchone()[0]
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None or image.dtype != np.uint8 or image.shape != (480, 640):
            raise ValueError('Expected calibrated 640x480 mono8 frame')
        return SimpleNamespace(width=640, height=480, step=640, encoding='mono8',
                               is_bigendian=0, data=image.ravel())

    def close(self):
        self.connection.close()


def open_source(robot):
    kind = robot.get('input_format', 'graco')
    if kind == 'realsense_cache':
        return RealSenseSource(robot)
    if kind == 'graco':
        return GracoSource(robot)
    raise ValueError(f'Unknown replay input format: {kind}')
