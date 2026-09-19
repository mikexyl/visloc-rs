"""Bounded, asynchronous browser output; a slow viewer never stalls VIO."""
import base64
import json
import queue
import threading
import urllib.request
import uuid


class BrowserStream:
    def __init__(self, url):
        self.url = url.rstrip('/') + '/api/live'
        self.stream = str(uuid.uuid4())
        self.first_ns = None
        self.pending = queue.Queue(maxsize=1)
        self.stopped = threading.Event()
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def publish(self, result, jpegs):
        if self.first_ns is None:
            self.first_ns = result['timestamp_ns']
        packet = dict(stream=self.stream, frame_id=result['frame_id'],
                      t=(result['timestamp_ns']-self.first_ns)*1e-9,
                      p=result['position'], q=result['quaternion_xyzw'],
                      observations=result['observations'], imu=result['imu_samples'],
                      map_points=result.get('map_points', []),
                      images=[base64.b64encode(data).decode() for data in jpegs])
        try:
            self.pending.get_nowait()
        except queue.Empty:
            pass
        self.pending.put_nowait(packet)

    def _run(self):
        failed = False
        while not self.stopped.is_set():
            try:
                packet = self.pending.get(timeout=.2)
            except queue.Empty:
                continue
            try:
                req = urllib.request.Request(self.url, json.dumps(packet, allow_nan=False).encode(),
                                             {'Content-Type': 'application/json'})
                with urllib.request.urlopen(req, timeout=1) as response:
                    response.read()
                failed = False
            except Exception as error:
                if not failed:
                    print(f'Browser viewer unavailable: {error}', flush=True)
                failed = True

    def close(self):
        self.stopped.set()
        self.worker.join(timeout=2)
