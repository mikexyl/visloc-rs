"""Check actual serial-descriptor ownership through the live HTTP server."""
import base64
import json
import os
from pathlib import Path
import pty
import socket
import subprocess
import sys
import time
import unittest
import urllib.error
import urllib.request


class GpsHttpTests(unittest.TestCase):
    def test_only_valid_active_mapping_frames_acquire_the_device(self):
        root = Path(__file__).resolve().parents[1]
        master, slave = pty.openpty()
        device = os.ttyname(slave)
        self.addCleanup(os.close, master)
        self.addCleanup(os.close, slave)
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            port = listener.getsockname()[1]
        process = subprocess.Popen([
            sys.executable, str(root/'tools/browser_viewer/live_server.py'),
            '--host', '127.0.0.1', '--port', str(port), '--gps-device', device,
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        def cleanup():
            process.terminate()
            process.wait(timeout=5)
        self.addCleanup(cleanup)
        url = f'http://127.0.0.1:{port}'

        def request(path, payload=None):
            body = None if payload is None else json.dumps(payload).encode()
            with urllib.request.urlopen(url+path, data=body, timeout=.5) as response:
                return json.loads(response.read())

        def owns_device():
            for fd in Path(f'/proc/{process.pid}/fd').iterdir():
                try:
                    if os.readlink(fd) == device:
                        return True
                except FileNotFoundError:
                    pass
            return False

        def wait_for(predicate, timeout=4):
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                try:
                    if predicate():
                        return
                except OSError:
                    pass
                time.sleep(.02)
            self.fail('Live server did not reach expected GPS state')

        wait_for(lambda: request('/api/session'))
        self.assertIn('waiting for mapping', request('/api/gps')['status'])
        self.assertFalse(owns_device())
        with self.assertRaises(urllib.error.HTTPError) as error:
            request('/api/live', {'stream': 'invalid'})
        self.assertEqual(error.exception.code, 400)
        self.assertFalse(owns_device())

        jpeg = base64.b64encode(b'\xff\xd8\xff\xd9').decode()
        packet = dict(stream='first', frame_id=0, t=0, p=[0, 0, 0], q=[0, 0, 0, 1],
                      observations=[1, 1], imu=1, images=[jpeg, jpeg], map_points=[])
        request('/api/live', packet)
        wait_for(owns_device)
        wait_for(lambda: not owns_device())
        self.assertIn('waiting for mapping', request('/api/gps')['status'])
        self.assertIsNone(request('/api/gps')['fix'])

        request('/api/live', dict(packet, stream='second'))
        wait_for(owns_device)


if __name__ == '__main__':
    unittest.main()
