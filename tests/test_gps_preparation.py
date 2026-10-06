import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('gps_prepare',Path(__file__).resolve().parents[1]/'scripts/prepare_gps_pose_graph.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class GpsPreparationTests(unittest.TestCase):
    def test_exact_utc_and_replay_roundtrip(self):
        t=module.utc_ns('270926','094904.600000001')
        self.assertEqual(t,1790502544600000001)
        offset=-1790502543606403113
        self.assertEqual((t+offset)-offset,t)
        self.assertEqual(module.utc_ns('280926','000000.0')-module.utc_ns('270926','235959.8'),200000000)
    def test_checksum_and_kind(self):
        raw='$GNRMC,094904.60,A,3508.55107,N,03324.48611,E,0.027,,270926,,,D,V*1E'
        self.assertEqual(module.sentence(raw)[0],'GNRMC')
        with self.assertRaises(ValueError):module.sentence(raw[:-2]+'00')

    def test_invalid_utc_is_rejected(self):
        for value in ('240000','096000','095960','09'):
            with self.assertRaises(ValueError):module.time_of_day_ns(value)
    def test_mounting_geometry_uses_measured_camera_to_imu_transform(self):
        import tempfile,json
        values=dict(px=-.030887849037458916,py=.006890377543619629,pz=.01904559767519483,
                    qx=-.001375972032525672,qy=.0005955286880858952,qz=-.001490337716126886,qw=.9999977654675232)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'calibration.json';p.write_text(json.dumps({'value0':{'T_imu_cam':[values]}}))
            arm,report=module.lever_arm(p)
            for a,b in zip(arm,[.111193231,-.094903010,-.078744703]):self.assertAlmostEqual(a,b,places=9)
            self.assertEqual(report['nominal'],True)
            values['px']+=.01;p.write_text(json.dumps({'value0':{'T_imu_cam':[values]}}))
            with self.assertRaises(ValueError):module.lever_arm(p)
