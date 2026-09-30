import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch, Mock

spec = importlib.util.spec_from_file_location('browser_controller', Path(__file__).resolve().parents[1]/'tools/browser_viewer/controller.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class BrowserControllerTests(unittest.TestCase):
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
