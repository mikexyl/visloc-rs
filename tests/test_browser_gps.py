#!/usr/bin/env python3
import math
import os
from pathlib import Path
import pty
import sys
import time
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools/browser_viewer'))
from gps_monitor import GpsMonitor, parse_gga


def sentence(body):
    check = 0
    for c in body:
        check ^= ord(c)
    return f'${body}*{check:02X}\r\n'


class GpsTests(unittest.TestCase):
    def wait_for(self, predicate, timeout=3):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(.01)
        self.fail('GPS lifecycle condition did not become true')

    def test_idle_reader_does_not_probe_or_open_devices(self):
        m = GpsMonitor()
        with patch('gps_monitor.glob.glob') as discover, patch('gps_monitor.os.open') as opened:
            m.start()
            thread = m.thread
            m.start()
            try:
                self.assertIs(m.thread, thread)
                time.sleep(.05)
                discover.assert_not_called()
                opened.assert_not_called()
                self.assertIn('waiting for mapping', m.snapshot()['status'])
            finally:
                m.close()
            self.assertFalse(thread.is_alive())

    def test_device_lifetime_follows_mapping_and_releases_on_stop(self):
        master, slave = pty.openpty()
        device = os.ttyname(slave)
        m = GpsMonitor(device)
        m.IDLE_TIMEOUT = .4
        opened, closed = [], []
        real_open, real_close = os.open, os.close
        def record_open(path, flags):
            fd = real_open(path, flags)
            opened.append(fd)
            return fd
        def record_close(fd):
            real_close(fd)
            closed.append(fd)
        try:
            with patch('gps_monitor.os.open', side_effect=record_open), patch('gps_monitor.os.close', side_effect=record_close):
                m.start()
                m.pose_update('first', [0, 0, 0])
                self.wait_for(lambda: len(opened) == 1)
                # Publisher disappears without an explicit Stop action.
                self.wait_for(lambda: len(closed) == 1)
                self.assertEqual(closed, opened)
                self.assertTrue(m.released.is_set())
                self.assertIsNone(m.snapshot()['fix'])
                # The same still-running publisher can reconnect after a gap.
                m.IDLE_TIMEOUT = 3
                m.pose_update('first', [1, 0, 0])
                self.wait_for(lambda: len(opened) == 2)
                m.pause()
                self.assertEqual(closed, opened)
                self.assertTrue(m.released.is_set())
                # A delayed packet from an explicitly stopped session must
                # not reacquire the receiver. A new mapping session may do so.
                m.pose_update('first', [2, 0, 0])
                self.assertIsNone(m.pose)
                m.pose_update('second', [0, 0, 0])
                self.wait_for(lambda: len(opened) == 3)
                m.close()
                self.assertEqual(closed, opened)
                self.assertFalse(m.thread.is_alive())
        finally:
            m.close()
            os.close(master)
            os.close(slave)

    def test_disconnected_device_is_not_retried_while_idle(self):
        m = GpsMonitor('/missing-gps')
        m.IDLE_TIMEOUT = .05
        with patch('gps_monitor.os.open', side_effect=OSError('disconnected')) as opened:
            m.start()
            try:
                m.pose_update('first', [0, 0, 0])
                self.wait_for(lambda: opened.call_count == 1)
                self.wait_for(lambda: 'waiting for mapping' in m.snapshot()['status'])
                self.assertEqual(opened.call_count, 1)
                m.IDLE_TIMEOUT = 3
                m.pose_update('second', [0, 0, 0])
                self.wait_for(lambda: opened.call_count == 2)
            finally:
                m.close()

    def test_disabled_reader_never_opens_after_mapping_starts(self):
        m = GpsMonitor('off')
        with patch('gps_monitor.os.open') as opened:
            m.start()
            m.pose_update('first', [0, 0, 0])
            self.assertIsNone(m.thread)
            opened.assert_not_called()
            m.close()

    def test_parser(self):
        fix = parse_gga(sentence('GNGGA,120000,4807.038,N,01131.000,E,4,12,0.8,545.4,M,46.9,M,,') )
        self.assertEqual(fix['quality'], 4)
        self.assertAlmostEqual(fix['lat'], 48.1173)
        self.assertAlmostEqual(fix['altitude'], 592.3)
        self.assertIsNone(parse_gga('$GNGGA,,,,,,0,00,99.99,,,,,,*00'))
        self.assertNotIn('lat', parse_gga(sentence('GNGGA,,,,,,0,00,99.99,,,,,,')))
        self.assertIsNone(parse_gga(sentence('GNGGA,120000,4899.0,N,01131.000,E,1,12,0.8,545.4,M,46.9,M,,')))

    def test_alignment_freezes_and_preserves_drift(self):
        m = GpsMonitor('off')
        now = time.monotonic()
        def feed(i, drift=0):
            # Equatorial eastward motion, VIO rotated 90 degrees and translated.
            m.pose_update('a', [3, 4+i+drift, 2], now+i)
            m.accept(dict(lat=0, lon=math.degrees(i/6378137), altitude=0,
                          hdop=.7, satellites=12, quality=4, utc=str(i)), now+i+.1)
        for i in range(12):
            feed(i)
        self.assertIsNotNone(m.transform)
        locked = m.transform
        feed(20, 5)
        self.assertEqual(m.transform, locked)
        g, v, _ = m.pairs[-1]
        self.assertAlmostEqual(math.dist(m.apply(g, locked)[:2], v[:2]), 5, places=4)
        m.pose_update('b', [0, 0, 0], now+21)
        self.assertIsNone(m.transform)
        self.assertFalse(m.pairs)

    def test_bad_or_unsynchronized_fixes_are_not_paired(self):
        m = GpsMonitor('off')
        m.pose_update('a', [0, 0, 0], 10)
        fix = dict(lat=0, lon=0, altitude=0, hdop=.8, satellites=12, quality=1, utc='1')
        m.accept(fix, 11)
        self.assertFalse(m.pairs)
        m.accept(dict(fix, hdop=9), 10.1)
        self.assertFalse(m.pairs)
        m.accept(fix, 10.1)
        m.accept(fix, 10.2)
        self.assertEqual(len(m.pairs), 1)

    def test_serial_reader(self):
        master, slave = pty.openpty()
        m = GpsMonitor(os.ttyname(slave))
        m.start()
        m.pose_update('serial-test', [0, 0, 0])
        try:
            deadline = time.monotonic()+3
            while time.monotonic() < deadline:
                os.write(master, sentence('GNGGA,,,,,,0,00,99.99,,,,,,').encode())
                if m.snapshot()['fix']:
                    break
                time.sleep(.05)
            self.assertEqual(m.snapshot()['fix']['quality'], 0)
        finally:
            m.close()
            os.close(master)
            os.close(slave)


if __name__ == '__main__':
    unittest.main()
