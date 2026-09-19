#!/usr/bin/env python3
"""Receive online VIO frames and serve the browser viewer on the same port."""
import argparse
import base64
import json
import math
import secrets
from urllib.parse import urlsplit, parse_qs, quote
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from resource_monitor import ResourceMonitor
from map_store import MapStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8090)
    parser.add_argument('--controls', action='store_true', help='Enable local Docker start/stop controls')
    args = parser.parse_args()
    controller = None
    if args.controls:
        from controller import Controller
        controller = Controller()
    token = secrets.token_urlsafe(32)
    lock = threading.Lock()
    state = {'sequence': 0, 'packet': None, 'received': 0}
    assets = Path(__file__).parent
    resources = ResourceMonitor()
    map_store = MapStore()

    class Handler(BaseHTTPRequestHandler):
        def reply(self, data, mime='application/json', status=200, headers=None):
            self.send_response(status)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Frame-Options', 'DENY')
            self.send_header('X-Content-Type-Options', 'nosniff')
            for key, value in (headers or {}).items():
                self.send_header(key, str(value))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            path = self.path.split('?')[0]
            if path == '/api/resources':
                self.reply(json.dumps(resources.snapshot(), allow_nan=False).encode())
            elif path == '/api/map':
                query = parse_qs(urlsplit(self.path).query)
                snapshot = map_store.snapshot()
                if query.get('stream', [''])[0] != snapshot['stream']:
                    self.reply(b'{}', status=409)
                    return
                headers = {'X-Map-Stream': quote(snapshot['stream'], safe=''),
                           'X-Map-Revision': snapshot['revision'],
                           'X-Map-Count': snapshot['count'],
                           'X-Map-Voxel-Size': snapshot['voxel_size']}
                if query.get('since', [''])[0] == str(snapshot['revision']):
                    self.reply(b'', status=304, headers=headers)
                else:
                    self.reply(snapshot['data'], 'application/octet-stream', headers=headers)
            elif path == '/api/control' and controller:
                try:
                    self.reply(json.dumps(controller.status()).encode())
                except Exception:
                    self.reply(b'{"error":"Unable to read VIO status"}', status=503)
            elif path == '/api/live':
                with lock:
                    payload = dict(sequence=state['sequence'], packet=state['packet'],
                                   age=time.monotonic()-state['received'] if state['packet'] else None)
                self.reply(json.dumps(payload).encode())
            elif path == '/api/session':
                self.reply(json.dumps(dict(live=True, name='Online VIO', mode='Waiting for VIO', frames=[], controls=bool(controller))).encode())
            elif path in ('/', '/app.js', '/resources.js', '/map_layer.js', '/style.css'):
                name, mime = {'/': ('index.html', 'text/html'), '/app.js': ('app.js', 'text/javascript'),
                              '/resources.js': ('resources.js', 'text/javascript'),
                              '/map_layer.js': ('map_layer.js', 'text/javascript'),
                              '/style.css': ('style.css', 'text/css')}[path]
                data = (assets/name).read_bytes()
                if path == '/':
                    data = data.replace(b'__CSRF_TOKEN__', token.encode())
                self.reply(data, mime)
            else:
                self.reply(b'{}', status=404)

        def do_POST(self):
            if self.path in ('/api/start', '/api/stop', '/api/start-gaussian', '/api/stop-gaussian') and controller:
                origin = self.headers.get('Origin')
                if (self.headers.get('X-Control-Token') != token or
                        (origin and urlsplit(origin).netloc != self.headers.get('Host'))):
                    self.reply(b'{"error":"Reload the viewer and try again"}', status=403)
                    return
                try:
                    controller.submit(self.path.rsplit('/', 1)[-1])
                    self.reply(b'{"accepted":true}', status=202)
                except (RuntimeError, ValueError) as error:
                    self.reply(json.dumps(dict(error=str(error))).encode(), status=409)
                return
            if self.path != '/api/live':
                self.reply(b'{}', status=404)
                return
            try:
                size = int(self.headers.get('Content-Length', 0))
                if not 0 < size <= 2_000_000:
                    raise ValueError('Invalid packet size')
                self.connection.settimeout(3)
                packet = json.loads(self.rfile.read(size))
                points = packet.get('map_points', [])
                if not isinstance(points, list) or len(points) > 3000:
                    raise ValueError('Invalid map size')
                for point in points:
                    if (not isinstance(point, list) or len(point) != 4 or
                            not isinstance(point[0], int) or isinstance(point[0], bool) or
                            not 0 <= point[0] <= 2**64 - 1 or
                            not all(math.isfinite(float(x)) and abs(float(x)) <= 1e7 for x in point[1:])):
                        raise ValueError('Invalid map point')
                for key, n in [('p', 3), ('q', 4), ('observations', 2)]:
                    if len(packet[key]) != n or not all(math.isfinite(float(x)) for x in packet[key]):
                        raise ValueError(key)
                for key in ('t', 'imu', 'frame_id'):
                    if not math.isfinite(float(packet[key])):
                        raise ValueError(key)
                if not isinstance(packet['stream'], str) or len(packet['stream']) > 100:
                    raise ValueError('stream')
                if len(packet['images']) != 2:
                    raise ValueError('images')
                for data in packet['images']:
                    if not base64.b64decode(data, validate=True).startswith(b'\xff\xd8'):
                        raise ValueError('JPEG required')
                with lock:
                    map_store.update(packet['stream'], points)
                    # Full map has its own cached, binary, rate-limited endpoint.
                    # Do not resend even active landmarks with every pose/image poll.
                    packet.pop('map_points', None)
                    state.update(sequence=state['sequence']+1, packet=packet, received=time.monotonic())
                self.reply(b'{}')
            except (ValueError, KeyError, TypeError, OverflowError):
                self.reply(b'{"error":"Invalid frame packet"}', status=400)

        def log_message(self, *_):
            pass

    print(f'Online viewer: http://127.0.0.1:{args.port} (waiting for VIO)', flush=True)
    with ThreadingHTTPServer((args.host, args.port), Handler) as server:
        resources.start()
        map_store.start()
        try:
            server.serve_forever()
        finally:
            resources.close()
            map_store.close()


if __name__ == '__main__':
    main()
