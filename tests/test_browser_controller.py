import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch, Mock

spec = importlib.util.spec_from_file_location('browser_controller', Path(__file__).resolve().parents[1]/'tools/browser_viewer/controller.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class BrowserControllerTests(unittest.TestCase):
    def test_status_query_is_limited_to_visloc(self):
        c = module.Controller()
        c.command = Mock(return_value='{"Names":"visloc-realsense","State":"running"}\n')
        self.assertTrue(c.running(c.containers(), 'visloc-realsense'))
        c.command.assert_called_once_with('docker', 'container', 'ls', '-a',
                                         '--filter', 'name=^/visloc-realsense$',
                                         '--format', '{{json .}}')

    def controller(self, **containers):
        controller = module.Controller()
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

    def test_successful_stop_releases_mapping_resources(self):
        c, _ = self.controller(**{'visloc-realsense': 'running'})
        c.on_mapping_stopped = Mock()
        c._work('stop')
        c.on_mapping_stopped.assert_called_once_with()
        self.assertEqual(c.error, '')

    def test_failed_stop_does_not_mark_mapping_as_stopped(self):
        c, _ = self.controller(**{'visloc-realsense': 'running'})
        c.command = Mock(side_effect=RuntimeError('stop failed'))
        c.on_mapping_stopped = Mock()
        c._work('stop')
        c.on_mapping_stopped.assert_not_called()
        self.assertEqual(c.error, 'stop failed')

    def test_start_does_not_acquire_mapping_resources(self):
        c, _ = self.controller(**{'visloc-realsense': 'running'})
        c.on_mapping_stopped = Mock()
        c._work('start')
        c.on_mapping_stopped.assert_not_called()

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



if __name__ == '__main__':
    unittest.main()
