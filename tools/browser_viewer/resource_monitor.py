"""Bounded, host-wide Linux resource telemetry; no third-party dependencies."""
import copy
import csv
import io
import math
import os
from pathlib import Path
import socket
import subprocess
import threading
import time
from collections import deque


def cpu_counters(text):
    """Exclude guest counters (already included in user/nice). Iowait is idle."""
    line = next(line for line in text.splitlines() if line.startswith('cpu '))
    values = [int(value) for value in line.split()[1:9]]
    if len(values) < 4 or any(value < 0 for value in values):
        raise ValueError('Invalid CPU counters')
    return sum(values), values[3] + (values[4] if len(values) > 4 else 0)


def memory_usage(text):
    values = {}
    for line in text.splitlines():
        key, value = line.split(':', 1)
        values[key] = int(value.split()[0]) * 1024
    total, available = values['MemTotal'], values['MemAvailable']
    if total <= 0 or available < 0 or available > total:
        raise ValueError('Invalid memory counters')
    used = total - available
    return dict(used_bytes=used, total_bytes=total, percent=100 * used / total)


def number(text):
    try:
        value = float(text.strip())
        return value if math.isfinite(value) and value >= 0 else None
    except ValueError:
        return None


def parse_nvidia(text):
    devices = []
    for row in csv.reader(io.StringIO(text), skipinitialspace=True):
        if len(row) != 6:
            continue
        uuid, name, utilization, used, total, temperature = [value.strip() for value in row]
        utilization, used, total, temperature = map(number, (utilization, used, total, temperature))
        memory = None
        if used is not None and total is not None and total > 0 and used <= total:
            memory = dict(used_bytes=int(used * 1024**2), total_bytes=int(total * 1024**2), percent=100 * used / total)
        devices.append(dict(id=uuid, name=name, percent=utilization if utilization is not None and utilization <= 100 else None,
                            memory=memory, memory_kind='dedicated', temperature_c=temperature, source='nvidia-smi'))
    return devices


class ResourceMonitor:
    """One sampler per server, independent of VIO and number of connected clients."""
    def __init__(self, interval=1.0, history_size=120, proc=Path('/proc'), sys=Path('/sys')):
        self.interval = interval
        self.proc, self.sys = Path(proc), Path(sys)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._history = deque(maxlen=history_size)
        self._previous_cpu = None
        self._sampled_at = None
        self._sequence = 0
        self.hostname = socket.gethostname()

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name='resource-monitor', daemon=True)
            self._thread.start()
        return self

    def close(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def _run(self):
        while not self._stop.is_set():
            start = time.monotonic()
            self.sample()
            self._stop.wait(max(0.05, self.interval - (time.monotonic() - start)))

    def gpu_usage(self):
        # Jetson's nvhost/nvgpu sysfs load is in tenths of a percent. Memory is
        # shared with the CPU; never label whole-system RAM as dedicated VRAM.
        model_path = self.sys / 'firmware/devicetree/base/model'
        model = model_path.read_text().strip('\x00\n') if model_path.exists() else ''
        if 'jetson' in model.lower() or 'tegra' in model.lower():
            candidates = [self.sys / 'devices/gpu.0/load']
            for pattern in ('bus/platform/drivers/nvgpu/*/load', 'bus/platform/devices/*gpu/load',
                            'devices/platform/*gpu/load', 'devices/platform/*/*gpu/load',
                            'class/devfreq/*gpu*/device/load'):
                candidates.extend(sorted(self.sys.glob(pattern)))
            for path in candidates:
                try:
                    load = number(path.read_text())
                    if load is not None and load <= 1000:
                        return [dict(id='jetson-integrated', name='Jetson integrated GPU', percent=load / 10,
                                     memory=None, memory_kind='shared', temperature_c=None, source='sysfs')], None
                except OSError:
                    continue
            return [dict(id='jetson-integrated', name='Jetson integrated GPU', percent=None,
                         memory=None, memory_kind='shared', temperature_c=None, source='sysfs')], 'GPU utilization unavailable on this Jetson'
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=uuid,name,utilization.gpu,memory.used,memory.total,temperature.gpu',
                '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=0.8, check=True)
            devices = parse_nvidia(result.stdout)
            return devices, None if devices else 'No NVIDIA GPU metrics available'
        except FileNotFoundError:
            return [], 'NVIDIA monitoring unavailable (nvidia-smi not installed)'
        except subprocess.TimeoutExpired:
            return [], 'NVIDIA monitoring timed out'
        except (OSError, subprocess.CalledProcessError):
            return [], 'NVIDIA monitoring unavailable (check driver/device access)'

    def sample(self):
        errors = []
        cpu, memory, devices = None, None, []
        try:
            current = cpu_counters((self.proc / 'stat').read_text())
            if self._previous_cpu:
                total, idle = (a - b for a, b in zip(current, self._previous_cpu))
                if total > 0 and 0 <= idle <= total:
                    cpu = 100 * (total - idle) / total
            self._previous_cpu = current
        except (OSError, ValueError, StopIteration):
            self._previous_cpu = None
            errors.append('CPU metrics unavailable')
        try:
            memory = memory_usage((self.proc / 'meminfo').read_text())
        except (OSError, ValueError, KeyError, IndexError):
            errors.append('Memory metrics unavailable')
        try:
            devices, gpu_error = self.gpu_usage()
            if gpu_error:
                errors.append(gpu_error)
        except (OSError, ValueError):
            errors.append('GPU metrics unavailable')
        with self._lock:
            self._sequence += 1
            self._history.append(dict(sequence=self._sequence, timestamp=time.time(), cpu_percent=cpu,
                                      memory=memory, gpus=devices, errors=errors))
            self._sampled_at = time.monotonic()

    def snapshot(self):
        # Copy while locked; HTTP serialization must not race sampler mutations.
        with self._lock:
            return dict(hostname=self.hostname, scope='viewer-host', interval_seconds=self.interval,
                        logical_cpus=os.cpu_count(), history_seconds=self.interval * self._history.maxlen,
                        age_seconds=None if self._sampled_at is None else time.monotonic() - self._sampled_at,
                        samples=copy.deepcopy(list(self._history)))
