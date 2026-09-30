"""Prepared sensor replay preserves integer timestamps and mono image pixels."""
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from replay_sensor_source import open_source


class ReplaySourceTests(unittest.TestCase):
    def recording(self, root, dtype=np.int64):
        t = 1790415501233623047  # exceeds exact float64 integer precision
        image = np.arange(480*640, dtype=np.uint8).reshape(480, 640)
        ok, data = cv2.imencode('.png', image)
        self.assertTrue(ok)
        with sqlite3.connect(root/'images.sqlite3') as db:
            db.execute('CREATE TABLE images(timestamp_ns INTEGER PRIMARY KEY, data BLOB)')
            db.executemany('INSERT INTO images VALUES (?,?)', [(t, data.tobytes()), (t+33333333, data.tobytes())])
        np.savez(root/'imu.npz', timestamps=np.array([t-1,t+1,t+5000000], dtype=dtype),
                 values=np.array([[1,2,3,4,5,6]]*3, dtype=np.float64))
        return dict(input_format='realsense_cache', sensor_cache=str(root), frames=2), t, image

    def test_exact_timestamps_pixels_and_read_only_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, t, image = self.recording(Path(tmp))
            source = open_source(config)
            try:
                self.assertEqual(source.frames, [(t,t,None),(t+33333333,t+33333333,None)])
                self.assertEqual([r[0] for r in source.imu], [t-1,t+1,t+5000000])
                self.assertEqual(source.imu[0][1:], [1,2,3,4,5,6])
                raw = source.image_at(t)
                self.assertEqual((raw.width, raw.height, raw.step, raw.encoding), (640,480,640,'mono8'))
                np.testing.assert_array_equal(raw.data.reshape(480,640), image)
                with self.assertRaises(sqlite3.OperationalError):
                    source.connection.execute('DELETE FROM images')
            finally:
                source.close()

    def test_frame_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, t, _ = self.recording(Path(tmp))
            source = open_source(dict(config, frames=1))
            try:
                self.assertEqual(len(source.frames), 1)
                with self.assertRaises(KeyError):
                    source.image_at(t+33333333)
            finally:
                source.close()

    def test_rejects_timestamp_precision_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, _, _ = self.recording(Path(tmp), dtype=np.float64)
            with self.assertRaises(ValueError):
                open_source(config)

    def test_unknown_format(self):
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            open_source(dict(input_format='unsupported'))

    def test_selected_imu_covered_interval_preserves_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, t, _ = self.recording(root)
            with sqlite3.connect(root/'images.sqlite3') as db:
                db.execute('CREATE TABLE right_images AS SELECT * FROM images')
            for mode in ['mono', 'stereo']:
                source = open_source(dict(config, camera_mode=mode, frames=1,
                                          original_first_ns=t+1, original_last_ns=t+33333333))
                try:
                    self.assertEqual([row[0] for row in source.frames], [t+33333333])
                    self.assertEqual(source.connection.execute('SELECT count(*) FROM images').fetchone()[0], 2)
                finally:
                    source.close()
            # An end bound is inclusive and excludes later frames independently.
            source = open_source(dict(config, original_last_ns=t))
            try:
                self.assertEqual([row[0] for row in source.frames], [t])
            finally:
                source.close()

    def test_stereo_uses_exact_pairs_and_distinct_right_pixels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, t, left = self.recording(root)
            right = np.flip(left, axis=1).copy()
            ok, data = cv2.imencode('.png', right)
            self.assertTrue(ok)
            with sqlite3.connect(root/'images.sqlite3') as db:
                db.execute('CREATE TABLE right_images(timestamp_ns INTEGER PRIMARY KEY, data BLOB)')
                db.executemany('INSERT INTO right_images VALUES (?,?)', [(t, data.tobytes()), (t+33333334, data.tobytes())])
            source = open_source(dict(config, camera_mode='stereo'))
            try:
                # A one-nanosecond mismatch is not silently reassigned.
                self.assertEqual(source.frames, [(t,t,t)])
                np.testing.assert_array_equal(source.image(t).data.reshape(480,640), left)
                np.testing.assert_array_equal(source.right_image(t).data.reshape(480,640), right)
            finally:
                source.close()


if __name__ == '__main__':
    unittest.main()
