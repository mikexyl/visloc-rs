#!/usr/bin/env python3
"""Fixed VIO lifecycle actions for the integrated browser viewer."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[2]


class Controller:
    def __init__(self, root=ROOT):
        self.root = root
        self.lock = threading.Lock()
        self.operation = None
        self.message = ''
        self.error = ''
        self.gaussian_root = Path(os.environ.get('STIPPLE_ROOT', str(Path.home() / 'workspaces/stipple-deployment/stipple-rs')))
        helper = self.gaussian_root / 'scripts/jetson/pipeline_control.py'
        self.gaussian = None
        if helper.is_file():
            spec = importlib.util.spec_from_file_location('pipeline_control', helper)
            self.gaussian = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.gaussian)

    def command(self, *args, timeout=30, env=None):
        result = subprocess.run(args, cwd=self.root, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        if result.returncode:
            raise RuntimeError((result.stderr or result.stdout).strip()[-1500:])
        return result.stdout

    def containers(self):
        raw = self.command('docker', 'container', 'ls', '-a', '--format', '{{json .}}')
        return {row['Names']: row for row in (json.loads(line) for line in raw.splitlines())}

    @staticmethod
    def running(containers, name):
        return containers.get(name, {}).get('State') == 'running'

    def status(self):
        containers = self.containers()
        gaussian_running = bool(self.gaussian and self.gaussian.running(self.gaussian_root))
        tracking_ok, tracking_lost = None, False
        if gaussian_running:
            try:
                folder = Path((self.gaussian_root / 'runs/latest_realsense_run.txt').read_text().strip()).resolve()
                if folder.is_relative_to((self.gaussian_root / 'runs').resolve()):
                    status = json.loads((folder / 'status.json').read_text())
                    tracking_ok = status.get('mapping_tracking_ok')
                    tracking_lost = status.get('mapping_tracking_lost', False)
            except (OSError, ValueError):
                pass
        with self.lock:
            return dict(vio=self.running(containers, 'visloc-realsense'),
                        gaussian_available=bool(self.gaussian),
                        gaussian=gaussian_running, mapping_tracking_ok=tracking_ok, mapping_tracking_lost=tracking_lost,
                        operation=self.operation, message=self.message, error=self.error)

    def submit(self, action):
        if action not in ('start', 'stop', 'start-gaussian', 'stop-gaussian'):
            raise ValueError('Unknown action')
        with self.lock:
            if self.operation:
                raise RuntimeError('Another operation is in progress')
            self.operation = action
            self.message = ''
            self.error = ''
        threading.Thread(target=self._work, args=(action,), daemon=True).start()

    def _work(self, action):
        try:
            self.perform(action)
        except Exception as error:
            with self.lock:
                self.error = str(error)
        finally:
            with self.lock:
                self.operation = None

    def perform(self, action):
        containers = self.containers()
        if action.endswith('-gaussian'):
            if not self.gaussian:
                raise RuntimeError('Gaussian pipeline is not installed')
            if action == 'stop-gaussian':
                self.gaussian.stop(self.gaussian_root)
                self.command('sudo', '-n', 'systemctl', 'stop', 'stipple-realsense.service')
                message = 'Gaussian pipeline stopped; map saved. Viewer remains available.'
            else:
                if self.running(containers, 'visloc-realsense'):
                    raise RuntimeError('Stop VIO before starting the Gaussian pipeline; both use the same camera.')
                if self.gaussian.running(self.gaussian_root):
                    message = 'Gaussian pipeline is already running.'
                else:
                    pointer = self.gaussian_root / 'runs/latest_realsense_run.txt'
                    previous_run = pointer.read_text().strip() if pointer.exists() else None
                    self.command('sudo', '-n', 'systemctl', 'start', 'stipple-viewer.service')
                    self.command('sudo', '-n', 'systemctl', 'start', 'stipple-realsense.service')
                    deadline = time.monotonic() + 180
                    while True:
                        state = self.command('systemctl', 'show', 'stipple-realsense.service', '--property=ActiveState', '--value').strip()
                        if state in ('failed', 'inactive'):
                            raise RuntimeError('Gaussian startup failed; inspect runs/panel-pipeline.log')
                        pointer = self.gaussian_root / 'runs/latest_realsense_run.txt'
                        if (pointer.exists() and pointer.read_text().strip() != previous_run
                                and self.gaussian.running(self.gaussian_root)):
                            status = Path(pointer.read_text().strip()) / 'status.json'
                            try:
                                if json.loads(status.read_text()).get('frame_id', 0) > 0:
                                    break
                            except (OSError, ValueError):
                                pass
                        if time.monotonic() >= deadline:
                            raise RuntimeError('Gaussian startup has not produced VIO frames; inspect runs/panel-pipeline.log')
                        time.sleep(1)
                    message = 'Gaussian pipeline is running (FFS depth + color).'
            with self.lock:
                self.message = message
            return
        if action == 'stop':
            if self.running(containers, 'visloc-realsense'):
                self.command('docker', 'stop', '--time', '20', 'visloc-realsense')
            with self.lock:
                self.message = 'Visloc stopped. The viewer remains available.'
            return
        if self.gaussian and self.gaussian.running(self.gaussian_root):
            raise RuntimeError('Stop the Gaussian pipeline before starting VIO; both use the same camera.')
        if self.running(containers, 'visloc-realsense'):
            with self.lock:
                self.message = 'Visloc is already running.'
            return
        if 'visloc-realsense' in containers:
            self.command('docker', 'rm', 'visloc-realsense')
        env = dict(os.environ, VISLOC_DETACH='1')
        self.command('bash', 'docker/jetson/run_browser.sh', timeout=45, env=env)
        # Wait for camera warmup; report a startup failure rather than a false success.
        time.sleep(4)
        if not self.running(self.containers(), 'visloc-realsense'):
            raise RuntimeError('Visloc exited during camera startup. Check the latest results folder on the Jetson.')
        with self.lock:
            self.message = 'Visloc is running.'
