import importlib.util
from pathlib import Path
import unittest
import tempfile
import json
from unittest.mock import patch, Mock

spec = importlib.util.spec_from_file_location('browser_controller', Path(__file__).resolve().parents[1]/'tools/browser_viewer/controller.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class BrowserControllerTests(unittest.TestCase):
    def controller(self, **containers):
        controller = module.Controller()
        controller.gaussian = None
        state = {name: {'State': value} for name, value in containers.items()}
        controller.containers = lambda: state
        calls = []

        def command(*args, **kwargs):
            calls.append(args)
            if args[:2] == ('docker', 'stop'):
                state.pop(args[-1], None)
            if args == ('bash', 'docker/jetson/run_browser.sh'):
                state['visloc-realsense'] = {'State': 'running'}
            return ''
        controller.command = command
        return controller, calls

    @patch.object(module.time, 'sleep')
    def test_running_dpvo_does_not_block_start_or_get_stopped(self, _):
        c, calls = self.controller(**{'dpvo-online-robot0':'running'})
        c.perform('start')
        self.assertEqual(calls, [('bash','docker/jetson/run_browser.sh')])
        self.assertTrue(c.running(c.containers(), 'dpvo-online-robot0'))

    @patch.object(module.time, 'sleep')
    def test_start_only_launches_vio(self, _):
        c, calls = self.controller()
        c.perform('start')
        self.assertEqual(calls, [('bash','docker/jetson/run_browser.sh')])

    def test_other_container_names_not_exposed(self):
        c, _ = self.controller(**{'dpvo-online-robot0':'running'})
        self.assertNotIn('dpvo', c.status())
        with self.assertRaises(ValueError):
            c.submit('takeover')

    def test_stop_only_stops_vio(self):
        c, calls = self.controller(**{'visloc-realsense':'running','dpvo-online-robot0':'running'})
        c.perform('stop')
        self.assertEqual(calls, [('docker','stop','--time','20','visloc-realsense')])

    def test_start_is_idempotent(self):
        c, calls = self.controller(**{'visloc-realsense':'running'})
        c.perform('start')
        self.assertEqual(calls, [])

    def test_concurrent_operations_rejected(self):
        c, _ = self.controller()
        c.operation = 'start'
        with self.assertRaisesRegex(RuntimeError, 'in progress'):
            c.submit('stop')

    @patch.object(module.time, 'sleep')
    def test_failed_start_is_reported(self, _):
        c, _ = self.controller()
        c.command = lambda *args, **kwargs: ''
        with self.assertRaisesRegex(RuntimeError, 'exited during camera startup'):
            c.perform('start')

    def test_gaussian_blocks_vio_camera_conflict(self):
        c, calls = self.controller()
        c.gaussian = Mock()
        c.gaussian.running.return_value = True
        with self.assertRaisesRegex(RuntimeError, 'same camera'):
            c.perform('start')
        self.assertEqual(calls, [])

    def test_vio_blocks_gaussian_camera_conflict(self):
        c, calls = self.controller(**{'visloc-realsense': 'running'})
        c.gaussian = Mock()
        with self.assertRaisesRegex(RuntimeError, 'same camera'):
            c.perform('start-gaussian')
        self.assertEqual(calls, [])

    def test_gaussian_stop_waits_for_save(self):
        c, calls = self.controller()
        c.gaussian = Mock()
        c.perform('stop-gaussian')
        c.gaussian.stop.assert_called_once_with(c.gaussian_root)
        self.assertIn('map saved', c.message)
        self.assertEqual(calls, [('sudo', '-n', 'systemctl', 'stop', 'stipple-realsense.service')])

    def test_gaussian_stop_failure_is_not_success(self):
        c, calls = self.controller()
        c.gaussian = Mock()
        c.gaussian.stop.side_effect = RuntimeError('save failed')
        c.operation = 'stop-gaussian'
        c._work('stop-gaussian')
        self.assertEqual(c.error, 'save failed')
        self.assertEqual(c.message, '')
        self.assertIsNone(c.operation)
        self.assertEqual(calls, [])

    def test_existing_manual_gaussian_run_is_not_started_twice(self):
        c, calls = self.controller()
        c.gaussian = Mock()
        c.gaussian.running.return_value = True
        c.perform('start-gaussian')
        self.assertEqual(calls, [])

    def test_tracking_warning_exposed_only_for_active_gaussian_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, _ = self.controller()
            c.gaussian_root = Path(tmp)
            folder = Path(tmp) / 'runs/test'
            folder.mkdir(parents=True)
            (Path(tmp) / 'runs/latest_realsense_run.txt').write_text(str(folder))
            (folder / 'status.json').write_text(json.dumps({'mapping_tracking_ok': False, 'mapping_tracking_lost': True}))
            c.gaussian = Mock()
            c.gaussian.running.return_value = True
            self.assertTrue(c.status()['mapping_tracking_lost'])
            self.assertFalse(c.status()['mapping_tracking_ok'])
            c.gaussian.running.return_value = False
            self.assertFalse(c.status()['mapping_tracking_lost'])
            self.assertIsNone(c.status()['mapping_tracking_ok'])


if __name__ == '__main__':
    unittest.main()
