"""Read-only NMEA USB GPS overlay. Never feeds the estimator.

Pairs by host receipt time (not hardware synchronized). Freeze unit-scale yaw
and translation after an initial 10 m baseline; never refit away later drift.
"""
import glob
import math
import os
import select
import termios
import threading
import time
import tty
from collections import deque


def parse_gga(line):
    try:
        body, checksum = line.strip().rsplit('*', 1)
        if not body.startswith('$') or len(checksum) != 2:
            return None
        check = 0
        for c in body[1:]:
            check ^= ord(c)
        if check != int(checksum, 16):
            return None
        f = body[1:].split(',')
        if len(f) < 15 or f[0] not in ('GNGGA', 'GPGGA'):
            return None
        quality = int(f[6])
        result = dict(quality=quality, satellites=int(f[7]), hdop=float(f[8]), utc=f[1])
        if not math.isfinite(result['hdop']):
            return None
        if quality not in (1, 2, 4, 5):
            return result
        def degrees(value, hemisphere, positive, negative, limit):
            raw = float(value)
            whole = int(raw // 100)
            minutes = raw - whole * 100
            if hemisphere not in (positive, negative) or not 0 <= minutes < 60:
                raise ValueError('coordinate')
            value = whole + minutes / 60
            if not 0 <= value <= limit:
                raise ValueError('coordinate')
            return value if hemisphere == positive else -value
        result.update(lat=degrees(f[2], f[3], 'N', 'S', 90),
                      lon=degrees(f[4], f[5], 'E', 'W', 180),
                      altitude=float(f[9]) + float(f[11]))
        if f[10] != 'M' or f[12] != 'M' or not math.isfinite(result['altitude']):
            return None
        return result
    except (ValueError, IndexError, OverflowError):
        return None


def ecef(fix):
    lat, lon = map(math.radians, (fix['lat'], fix['lon']))
    n = 6378137 / math.sqrt(1 - 6.69437999014e-3 * math.sin(lat)**2)
    return ((n + fix['altitude']) * math.cos(lat) * math.cos(lon),
            (n + fix['altitude']) * math.cos(lat) * math.sin(lon),
            (n * (1 - 6.69437999014e-3) + fix['altitude']) * math.sin(lat))


def enu(fix, origin):
    x, y, z = [a-b for a, b in zip(ecef(fix), ecef(origin))]
    lat, lon = map(math.radians, (origin['lat'], origin['lon']))
    return [-math.sin(lon)*x + math.cos(lon)*y,
            -math.sin(lat)*math.cos(lon)*x - math.sin(lat)*math.sin(lon)*y + math.cos(lat)*z,
            math.cos(lat)*math.cos(lon)*x + math.cos(lat)*math.sin(lon)*y + math.sin(lat)*z]


class GpsMonitor:
    # Match the viewer's stopped-stream threshold. A crashed/disconnected
    # publisher must release the receiver without requiring a browser client.
    IDLE_TIMEOUT = 3.0

    def __init__(self, device='auto'):
        self.device = device
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.wake = threading.Event()
        self.released = threading.Event()
        self.released.set()
        self.thread = None
        self.status = self.idle_status()
        self.fix = None
        self.received = None
        self.stream = None
        self.pose = None
        self.origin = None
        self.pairs = deque(maxlen=3000)
        self.calibration = deque(maxlen=300)
        self.transform = None
        self.last_utc = None
        self.paused_stream = None

    def idle_status(self):
        return 'GPS idle · waiting for mapping' if self.device != 'off' else 'GPS disabled'

    def active(self, now):
        # Caller holds self.lock.
        return (not self.stop.is_set() and self.pose is not None
                and 0 <= now-self.pose[0] < self.IDLE_TIMEOUT)

    def start(self):
        if self.device != 'off' and self.thread is None and not self.stop.is_set():
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()

    def pause(self):
        with self.lock:
            # Ignore late packets from a stopped run; a new publisher has a
            # new stream ID. Automatic idle expiry still permits reconnection.
            self.paused_stream = self.stream
            self.pose = self.fix = self.received = None
            self.status = self.idle_status()
        self.wake.set()
        self.released.wait(timeout=2)

    def close(self):
        self.stop.set()
        self.wake.set()
        if self.thread:
            self.thread.join(timeout=2)

    def pose_update(self, stream, p, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            if self.stop.is_set() or stream == self.paused_stream:
                return
            if stream != self.stream:
                self.stream = stream
                self.origin = self.transform = self.last_utc = None
                self.pairs.clear()
                self.calibration.clear()
            self.pose = (now, list(map(float, p)))
        self.wake.set()

    def accept(self, fix, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            if not self.active(now):
                return
            self.fix, self.received = fix, now
            self.status = 'Receiving GPS'
            if ('lat' not in fix or fix['hdop'] > 3 or not self.pose or
                    not 0 <= now-self.pose[0] <= .5 or fix['utc'] == self.last_utc):
                return
            self.last_utc = fix['utc']
            if self.origin is None:
                self.origin = fix.copy()
            g, v = enu(fix, self.origin), self.pose[1]
            self.pairs.append((g, v, now))
            if self.transform is None:
                self.calibration.append((g, v))
                pairs = self.calibration
                if len(pairs) < 8:
                    return
                if (math.dist(g[:2], pairs[0][0][:2]) < 10 or
                        math.dist(v[:2], pairs[0][1][:2]) < 10):
                    return
                gm = [sum(p[0][i] for p in pairs)/len(pairs) for i in range(3)]
                vm = [sum(p[1][i] for p in pairs)/len(pairs) for i in range(3)]
                dot = cross = 0
                for a, b in pairs:
                    x, y = a[0]-gm[0], a[1]-gm[1]
                    u, w = b[0]-vm[0], b[1]-vm[1]
                    dot += x*u + y*w
                    cross += x*w - y*u
                angle = math.atan2(cross, dot)
                c, s = math.cos(angle), math.sin(angle)
                transform = (c, s, vm[0]-c*gm[0]+s*gm[1], vm[1]-s*gm[0]-c*gm[1], vm[2]-gm[2])
                residual = math.sqrt(sum(math.dist(self.apply(a, transform)[:2], b[:2])**2 for a, b in pairs)/len(pairs))
                if residual <= 3:
                    self.transform = transform

    @staticmethod
    def apply(p, transform):
        c, s, x, y, z = transform
        return [c*p[0]-s*p[1]+x, s*p[0]+c*p[1]+y, p[2]+z]

    def snapshot(self):
        with self.lock:
            now = time.monotonic()
            age = None if self.received is None else now-self.received
            points = [self.apply(g, self.transform) for g, _, _ in self.pairs] if self.transform else []
            deviation = None
            if points and now-self.pairs[-1][2] <= 3 and age is not None and age <= 3 and self.fix and 'lat' in self.fix and self.fix['hdop'] <= 3:
                deviation = math.dist(points[-1][:2], self.pairs[-1][1][:2])
            return dict(stream=self.stream, status=self.status, age=age, fix=self.fix,
                        aligned=self.transform is not None, points=points,
                        horizontal_difference_m=deviation, pairs=len(self.pairs))

    def run(self):
        while not self.stop.is_set():
            self.wake.clear()
            with self.lock:
                active = self.active(time.monotonic())
                if not active:
                    self.status = self.idle_status()
                    self.fix = self.received = None
            if not active:
                # Do not even discover/probe serial devices while mapping is
                # idle. pose_update wakes this worker on the first valid frame.
                self.wake.wait(1)
                continue
            fd = None
            try:
                devices = sorted(glob.glob('/dev/serial/by-id/*u-blox*')) if self.device == 'auto' else [self.device]
                if not devices:
                    raise OSError('USB GPS not connected')
                with self.lock:
                    # A Stop action can arrive while discovery is in progress.
                    if not self.active(time.monotonic()):
                        continue
                    fd = os.open(devices[0], os.O_RDONLY | os.O_NONBLOCK | os.O_NOCTTY)
                    self.released.clear()
                # Raw mode prevents terminal echo from sending GPS output back as commands.
                tty.setraw(fd, termios.TCSANOW)
                attrs = termios.tcgetattr(fd)
                attrs[4] = attrs[5] = termios.B38400
                termios.tcsetattr(fd, termios.TCSANOW, attrs)
                with self.lock:
                    self.status = 'Connected · waiting for NMEA GGA'
                buffer = b''
                while not self.stop.is_set():
                    with self.lock:
                        remaining = (self.pose[0] + self.IDLE_TIMEOUT - time.monotonic()
                                     if self.pose else 0)
                    if remaining <= 0:
                        break
                    if not select.select([fd], [], [], min(.25, remaining))[0]:
                        continue
                    chunk = os.read(fd, 4096)
                    if not chunk:
                        raise OSError('USB GPS disconnected')
                    buffer += chunk
                    while b'\n' in buffer:
                        line, buffer = buffer.split(b'\n', 1)
                        fix = parse_gga(line.decode('ascii', errors='replace'))
                        if fix is not None:
                            self.accept(fix)
                    if len(buffer) > 8192:
                        buffer = b''
            except (OSError, termios.error) as error:
                with self.lock:
                    self.status = str(error) if self.active(time.monotonic()) else self.idle_status()
            finally:
                if fd is not None:
                    os.close(fd)
                self.released.set()
                with self.lock:
                    if not self.active(time.monotonic()):
                        self.status = self.idle_status()
                        self.fix = self.received = None
            # Bound retries for a missing/disconnected receiver even when
            # mapping frames arrive continuously.
            self.stop.wait(1)
