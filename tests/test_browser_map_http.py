"""Exercise the deployed HTTP protocol using a disposable local viewer process."""
import base64
import json
from pathlib import Path
import socket
import struct
import subprocess
import sys
import time
import unittest
import urllib.error
import urllib.request


class MapHttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            port = listener.getsockname()[1]
        cls.url = f'http://127.0.0.1:{port}'
        cls.process = subprocess.Popen([sys.executable, str(root/'tools/browser_viewer/live_server.py'),
                                        '--host', '127.0.0.1', '--port', str(port)],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(100):
            try:
                with urllib.request.urlopen(cls.url+'/api/session', timeout=.2):
                    return
            except OSError:
                time.sleep(.05)
        cls.process.terminate()
        cls.process.wait(timeout=5)
        raise RuntimeError('Test viewer did not start')

    @classmethod
    def tearDownClass(cls):
        cls.process.terminate()
        cls.process.wait(timeout=5)

    def request(self, path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(self.url+path, data=data, headers={'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=2) as response:
                return response.status, response.headers, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.headers, error.read()

    def publish(self, stream, points):
        jpeg = base64.b64encode(b'\xff\xd8\xff\xd9').decode()
        return self.request('/api/live', dict(stream=stream, frame_id=0, t=0, p=[0,0,0], q=[0,0,0,1],
                                              observations=[1,1], imu=1, images=[jpeg,jpeg], map_points=points))

    def map_with_count(self, stream, count):
        for _ in range(60):
            response = self.request('/api/map?stream='+stream)
            if response[0] == 200 and response[1]['X-Map-Count'] == str(count):
                return response
            time.sleep(.05)
        self.fail(f'Map never reached {count} points')

    def test_retention_binary_cache_reset_and_input_validation(self):
        self.assertEqual(self.publish('first', [[1,1,2,3]])[0], 200)
        self.map_with_count('first', 1)
        self.assertEqual(self.publish('first', [[2,4,5,6]])[0], 200)
        status, headers, data = self.map_with_count('first', 2)
        self.assertEqual(set(struct.iter_unpack('<fff',data)), {(1,2,3),(4,5,6)})
        self.assertEqual(self.request('/api/map?stream=first&since='+headers['X-Map-Revision'])[0],304)
        self.assertEqual(self.request('/api/map?stream=other')[0],409)
        self.assertNotIn('map_points',json.loads(self.request('/api/live')[2])['packet'])
        self.assertEqual(self.publish('first', [[1.5,0,0,0]])[0],400)
        self.assertEqual(self.publish('first', [[1,1e300,0,0]])[0],400)
        self.assertEqual(self.publish('first', [[i,0,0,0] for i in range(3001)])[0],400)
        self.assertEqual(self.publish('second', [])[0],200)
        self.assertEqual(self.map_with_count('second',0)[2],b'')
        self.assertEqual(self.request('/api/map?stream=first')[0],409)
        for asset in ('map_layer.js','resources.js','app.js'):
            self.assertEqual(self.request('/'+asset)[0],200)


if __name__ == '__main__':
    unittest.main()
