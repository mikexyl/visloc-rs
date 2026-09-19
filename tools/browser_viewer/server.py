#!/usr/bin/env python3
"""Local browser playback of a visloc trajectory and its EuRoC stereo images."""
import argparse
import csv
import io
import json
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from PIL import Image
from resource_monitor import ResourceMonitor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--euroc-dir', type=Path, required=True)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8090)
    args = parser.parse_args()
    with (args.run_dir / 'trajectory.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        parser.error('Trajectory is empty')
    first = int(rows[0]['timestamp_ns'])
    frames = [dict(t=(int(r['timestamp_ns'])-first)/1e9,
                   p=[float(r[k]) for k in ('tx','ty','tz')],
                   q=[float(r[k]) for k in ('qx','qy','qz','qw')],
                   observations=[int(r[f'cam{i}_observations']) for i in range(2)],
                   imu=int(r['imu_samples'])) for r in rows]
    manifests = []
    for i in range(2):
        folder = args.euroc_dir / 'mav0' / f'cam{i}'
        with (folder / 'data.csv').open() as stream:
            manifest = {int(r[0]): folder/'data'/r[1].strip() for r in csv.reader(stream)
                        if r and not r[0].startswith('#')}
        manifests.append(manifest)
    payload = json.dumps(dict(name=args.euroc_dir.name, mode='Recorded VIO replay', frames=frames), allow_nan=False).encode()
    assets = Path(__file__).parent
    resources = ResourceMonitor()

    @lru_cache(maxsize=100)
    def image(camera, index):
        path = manifests[camera][int(rows[index]['timestamp_ns'])]
        with Image.open(path) as source:
            source.thumbnail((640, 480))
            out = io.BytesIO()
            source.convert('L').save(out, format='JPEG', quality=80)
        return out.getvalue()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlparse(self.path)
            try:
                if url.path == '/api/resources':
                    data, mime = json.dumps(resources.snapshot(), allow_nan=False).encode(), 'application/json'
                elif url.path == '/api/session':
                    data, mime = payload, 'application/json'
                elif url.path == '/api/image':
                    query = parse_qs(url.query)
                    camera, index = int(query['camera'][0]), int(query['frame'][0])
                    if camera not in (0,1) or not 0 <= index < len(rows):
                        raise ValueError('Invalid frame')
                    data, mime = image(camera, index), 'image/jpeg'
                elif url.path in ('/', '/app.js', '/resources.js', '/map_layer.js', '/style.css'):
                    filename = {'/':'index.html','/app.js':'app.js','/resources.js':'resources.js','/map_layer.js':'map_layer.js','/style.css':'style.css'}[url.path]
                    mime = {'/':'text/html','/app.js':'text/javascript','/resources.js':'text/javascript','/map_layer.js':'text/javascript','/style.css':'text/css'}[url.path]
                    data = (assets/filename).read_bytes()
                elif url.path == '/favicon.ico':
                    self.send_response(204); self.end_headers(); return
                else:
                    self.send_error(404); return
                self.send_response(200)
                self.send_header('Content-Type', mime)
                self.send_header('Content-Length', str(len(data)))
                self.send_header('Cache-Control', 'no-cache' if mime != 'image/jpeg' else 'private, max-age=3600')
                self.end_headers()
                self.wfile.write(data)
            except (ValueError, KeyError, IndexError):
                self.send_error(400, 'Invalid camera or frame')
            except FileNotFoundError:
                self.send_error(404, 'Image not found')
            except (BrokenPipeError, ConnectionResetError):
                pass
        def log_message(self, *_):
            pass

    print(f'Viewer: http://127.0.0.1:{args.port} — {len(frames)} frames of {args.euroc_dir.name}', flush=True)
    with ThreadingHTTPServer((args.host, args.port), Handler) as server:
        resources.start()
        try:
            server.serve_forever()
        finally:
            resources.close()


if __name__ == '__main__':
    main()
