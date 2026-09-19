"""Run with python3 -m unittest discover -s tests -p test_browser_resources.py."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('browser_resources', Path(__file__).resolve().parents[1] / 'tools/browser_viewer/resource_monitor.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ResourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.proc, self.sys = self.root / 'proc', self.root / 'sys'
        self.proc.mkdir()
        self.sys.mkdir()
        (self.proc / 'stat').write_text('cpu 100 0 100 800 0 0 0 0 90 0\n')
        (self.proc / 'meminfo').write_text('MemTotal: 1000 kB\nMemAvailable: 700 kB\nMemFree: 50 kB\n')
        self.monitor = m.ResourceMonitor(proc=self.proc, sys=self.sys, history_size=2)

    def test_cpu_delta_excludes_guest_and_memory_excludes_cache(self):
        with patch.object(self.monitor, 'gpu_usage', return_value=([], None)):
            self.monitor.sample()
            self.assertIsNone(self.monitor.snapshot()['samples'][0]['cpu_percent'])
            (self.proc / 'stat').write_text('cpu 150 0 100 850 0 0 0 0 140 0\n')
            self.monitor.sample()
        sample = self.monitor.snapshot()['samples'][-1]
        self.assertEqual(sample['cpu_percent'], 50)
        self.assertEqual(sample['memory'], dict(used_bytes=300*1024, total_bytes=1000*1024, percent=30))

    def test_gpu_csv_multiple_devices_and_na(self):
        devices = m.parse_nvidia('GPU-a, "NVIDIA GPU, test", 75, 1024, 8192, 65\nGPU-b, Other, [N/A], [N/A], [N/A], N/A\n')
        self.assertEqual(len(devices), 2)
        self.assertEqual(devices[0]['percent'], 75)
        self.assertEqual(devices[0]['memory']['percent'], 12.5)
        self.assertIsNone(devices[1]['percent'])
        self.assertIsNone(devices[1]['memory'])
        self.assertIsNone(m.number('nan'))
        self.assertIsNone(m.number('inf'))
        self.assertIsNone(m.number('-1'))

    def test_timeout_and_missing_driver_do_not_stop_cpu_sampling(self):
        for error in (subprocess.TimeoutExpired('nvidia-smi', .8), FileNotFoundError(), subprocess.CalledProcessError(9, 'nvidia-smi')):
            with patch.object(m.subprocess, 'run', side_effect=error):
                self.monitor.sample()
            sample = self.monitor.snapshot()['samples'][-1]
            self.assertEqual(sample['gpus'], [])
            self.assertTrue(sample['errors'])
            self.assertEqual(sample['memory']['percent'], 30)
        json.dumps(self.monitor.snapshot(), allow_nan=False)

    def test_history_is_bounded_and_snapshots_are_independent(self):
        with patch.object(self.monitor, 'gpu_usage', return_value=([], None)):
            for _ in range(4):
                self.monitor.sample()
        snapshot = self.monitor.snapshot()
        self.assertEqual([s['sequence'] for s in snapshot['samples']], [3, 4])
        snapshot['samples'][0]['memory']['percent'] = -1
        self.assertEqual(self.monitor.snapshot()['samples'][0]['memory']['percent'], 30)
        self.assertLess(self.monitor.snapshot()['age_seconds'], 1)

    def test_counter_reset_and_unreadable_proc(self):
        with patch.object(self.monitor, 'gpu_usage', return_value=([], None)):
            self.monitor.sample()
            (self.proc / 'stat').write_text('cpu 1 0 1 8\n')
            self.monitor.sample()
            self.assertIsNone(self.monitor.snapshot()['samples'][-1]['cpu_percent'])
            (self.proc / 'stat').unlink()
            (self.proc / 'meminfo').write_text('invalid')
            self.monitor.sample()
        sample = self.monitor.snapshot()['samples'][-1]
        self.assertIsNone(sample['memory'])
        self.assertIn('CPU metrics unavailable', sample['errors'])

    def test_jetson_uses_sysfs_and_labels_shared_memory(self):
        model = self.sys / 'firmware/devicetree/base/model'
        model.parent.mkdir(parents=True)
        model.write_text('NVIDIA Jetson Orin Nano\x00')
        load = self.sys / 'devices/platform/17000000.gpu/load'
        load.parent.mkdir(parents=True)
        load.write_text('425\n')
        with patch.object(m.subprocess, 'run') as run:
            devices, error = self.monitor.gpu_usage()
            run.assert_not_called()
        self.assertIsNone(error)
        self.assertEqual(devices[0]['percent'], 42.5)
        self.assertEqual(devices[0]['memory_kind'], 'shared')
        self.assertIsNone(devices[0]['memory'])
        load.unlink()
        devices, error = self.monitor.gpu_usage()
        self.assertIsNone(devices[0]['percent'])
        self.assertIsNotNone(error)

    def test_jetpack_bus_nested_gpu_load(self):
        model = self.sys / 'firmware/devicetree/base/model'
        model.parent.mkdir(parents=True)
        model.write_text('NVIDIA Jetson Orin NX Engineering Reference Developer Kit Super\x00')
        load = self.sys / 'devices/platform/bus@0/17000000.gpu/load'
        load.parent.mkdir(parents=True)
        load.write_text('123')
        devices, error = self.monitor.gpu_usage()
        self.assertIsNone(error)
        self.assertEqual(devices[0]['percent'], 12.3)

    def test_start_stop(self):
        with patch.object(self.monitor, 'gpu_usage', return_value=([], None)):
            self.monitor.start()
            self.monitor.close()
        self.assertFalse(self.monitor._thread.is_alive())


if __name__ == '__main__':
    unittest.main()
