"""Session-scoped, spatially bounded visualization map (not estimator state)."""
from array import array
import math
import sys
import threading
import time


class MapStore:
    """Keep last-known positions after tracks leave the VIO window.

    One representative per voxel, with a bounded ID index for correcting retained
    points. At capacity, double voxel width and merge *all* occupied cells rather
    than evicting the oldest area. Serialize once per second in a worker, never
    once per browser or camera frame.
    """
    def __init__(self, max_points=30000, voxel_size=0.10, interval=1.0):
        if max_points < 8 or voxel_size <= 0 or not math.isfinite(voxel_size):
            raise ValueError('Invalid map budget')
        self.max_points = max_points
        self.initial_voxel_size = voxel_size
        self.voxel_size = voxel_size
        self.interval = interval
        self.stream = None
        self.cells = {}
        self.ids = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._dirty = False
        self._revision = 0
        self._snapshot = self._empty_snapshot()

    def _empty_snapshot(self):
        return dict(stream=self.stream, revision=self._revision, count=0,
                    voxel_size=self.voxel_size, data=b'', sampled_at=time.monotonic())

    def start(self):
        self._thread = threading.Thread(target=self._run, name='visualization-map', daemon=True)
        self._thread.start()
        return self

    def close(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def _run(self):
        while not self._stop.wait(self.interval):
            self.serialize()

    def _cell(self, point):
        return tuple(math.floor(v / self.voxel_size) for v in point[1:])

    def _compact(self):
        while len(self.cells) > self.max_points:
            self.voxel_size *= 2
            cells = {}
            for point in self.cells.values():
                cells.setdefault(self._cell(point), point)
            self.cells = cells
        self.ids = {point[0]: key for key, point in self.cells.items()}

    def update(self, stream, points):
        with self._lock:
            if stream != self.stream:
                self.stream = stream
                self.cells.clear()
                self.ids.clear()
                self.voxel_size = self.initial_voxel_size
                self._revision += 1
                self._snapshot = self._empty_snapshot()
                self._dirty = True
            for raw in points:
                point = (int(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))
                # Wire validation enforces this too; protect standalone use.
                if not all(math.isfinite(v) and abs(v) <= 1e7 for v in point[1:]):
                    continue
                key = self._cell(point)
                old_key = self.ids.get(point[0])
                if old_key is not None:
                    if self.cells[old_key] == point:
                        continue
                    del self.cells[old_key]
                    del self.ids[point[0]]
                    self._dirty = True
                if key not in self.cells:
                    self.cells[key] = point
                    self.ids[point[0]] = key
                    self._dirty = True
                if len(self.cells) > self.max_points:
                    self._compact()
            # Missing IDs are deliberately retained: marginalization is not deletion
            # from the explored map. Only a new publisher stream resets the map.

    def serialize(self):
        with self._lock:
            if not self._dirty:
                return
            coords = array('f', (v for point in self.cells.values() for v in point[1:]))
            if sys.byteorder != 'little':
                coords.byteswap()
            self._revision += 1
            self._snapshot = dict(stream=self.stream, revision=self._revision, count=len(self.cells),
                                  voxel_size=self.voxel_size, data=coords.tobytes(), sampled_at=time.monotonic())
            self._dirty = False

    def snapshot(self):
        with self._lock:
            # Immutable bytes are shared across clients rather than copied/repacked.
            return dict(self._snapshot)
