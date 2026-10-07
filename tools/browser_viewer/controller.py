#!/usr/bin/env python3
"""Fixed VIO lifecycle actions for the integrated browser viewer."""
import json
import os
from pathlib import Path
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[2]


class Controller:
    def __init__(self, root=ROOT, on_mapping_stopped=None):
        self.root = root
        self.lock = threading.Lock()
        self.operation = None
        self.message = ''
        self.error = ''
        self.on_mapping_stopped = on_mapping_stopped

    def command(self, *args, timeout=30, env=None):
        result = subprocess.run(args, cwd=self.root, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        if result.returncode:
            raise RuntimeError((result.stderr or result.stdout).strip()[-1500:])
        return result.stdout

    def containers(self):
        # Listing unrelated containers is expensive on the Jetson runtime.
        # Lifecycle decisions only require this exact container name.
        raw = self.command('docker', 'container', 'ls', '-a',
                           '--filter', 'name=^/visloc-realsense$', '--format', '{{json .}}')
        return {row['Names']: row for row in (json.loads(line) for line in raw.splitlines())}

    @staticmethod
    def running(containers, name):
        return containers.get(name, {}).get('State') == 'running'

    def status(self):
        containers = self.containers()
        with self.lock:
            return dict(vio=self.running(containers, 'visloc-realsense'),
                        operation=self.operation, message=self.message, error=self.error)

    def submit(self, action):
        if action not in ('start', 'stop'):
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
            if action == 'stop' and self.on_mapping_stopped:
                self.on_mapping_stopped()
        except Exception as error:
            with self.lock:
                self.error = str(error)
        finally:
            with self.lock:
                self.operation = None

    def perform(self, action):
        containers = self.containers()
        if action == 'stop':
            if self.running(containers, 'visloc-realsense'):
                self.command('docker', 'stop', '--time', '20', 'visloc-realsense')
            with self.lock:
                self.message = 'Visloc stopped. The viewer remains available.'
            return
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
