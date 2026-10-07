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

    def test_backend_selection_keeps_ba_observation_transport_and_gps_pair_consistent(self):
        import contextlib, io, json, tempfile
        from unittest.mock import patch
        record=dict(timestamp_ns=1790502544600000001, receipt_timestamp_ns=1790502544620000001)
        for source_mode, override, expected in [
            ('pose_graph', None, 'global_bundle_adjustment'),
            ('global_bundle_adjustment', 'pose_graph', 'pose_graph'),
        ]:
            with self.subTest(source_mode=source_mode, override=override), tempfile.TemporaryDirectory() as d:
                root=Path(d); robot=root/'robot.json'; backend=root/'backend.json'; source=root/'mission.json'
                robot.write_text(json.dumps(dict(calibration='calibrated.json',
                    bundle_adjustment_enabled=source_mode == 'global_bundle_adjustment')))
                backend.write_text(json.dumps(dict(pgo=dict(mode=source_mode))))
                source.write_text(json.dumps(dict(backend_config=str(backend),robots=[dict(
                    robot='ucy01', config=str(robot), bag='recording', frames=100,
                    sensor_cache='immutable_cache', offset_ns=123, gps_records='stale_source_records.jsonl')])))
                originals=[p.read_bytes() for p in (robot,backend,source)]
                with patch.object(module,'lever_arm',return_value=([.1,.2,.3],{})), \
                     patch.object(module,'normalize',return_value=([record],[record],dict(records=1))), \
                     contextlib.redirect_stdout(io.StringIO()):
                    module.prepare(source,root/'out',.25,frames=20,backend_mode=override)
                for name,enabled in [('gps_off',False),('gps_on',True)]:
                    run=root/'out'/name
                    config=json.loads((run/'robot.json').read_text())
                    pgo=json.loads((run/'backend.json').read_text())['pgo']
                    mission=json.loads((run/'mission.json').read_text())
                    self.assertEqual(pgo['mode'],expected)
                    self.assertEqual(config['bundle_adjustment_enabled'],expected == 'global_bundle_adjustment')
                    self.assertEqual(config['gps']['enabled'],enabled)
                    self.assertTrue(config['gps']['normalized_input'])
                    self.assertEqual(pgo['gps']['enabled'],enabled)
                    self.assertEqual(config['calibration'],'calibrated.json')
                    self.assertEqual(mission['robots'][0]['sensor_cache'],'immutable_cache')
                    self.assertEqual(mission['robots'][0]['offset_ns'],123)
                    self.assertEqual(mission['robots'][0]['frames'],20)
                    self.assertEqual('gps_records' in mission['robots'][0],enabled)
                self.assertEqual(json.loads((root/'out/gps_records.jsonl').read_text()),record)
                self.assertEqual(originals,[p.read_bytes() for p in (robot,backend,source)])

    def test_missing_gps_uses_visual_backend_and_never_invents_a_lever_arm(self):
        import json, tempfile
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with patch.object(module,'normalize',return_value=([],[],dict(records=0))), \
                 patch.object(module,'lever_arm') as lever:
                robot,backend,path=module.prepare_inputs(Path('bag'),'other',Path('calibration'),root)
            self.assertFalse(robot['enabled'])
            self.assertFalse(backend['enabled'])
            self.assertIsNone(path)
            lever.assert_not_called()
            self.assertEqual(json.loads((root/'gps_input_audit.json').read_text())['status'],
                             'no_receiver_records_visual_only')
            with patch.object(module,'normalize') as normalize:
                robot,backend,path=module.prepare_inputs(Path('bag'),'other',Path('calibration'),root,enabled=False)
            normalize.assert_not_called()
            self.assertFalse(backend['enabled'])

    def test_no_gps_topics_produces_empty_audit_without_scanning_other_messages(self):
        from unittest.mock import MagicMock, patch
        reader=MagicMock();reader.connections=[]
        with patch('rosbags.highlevel.AnyReader') as factory:
            factory.return_value.__enter__.return_value=reader
            records,mapping,audit=module.normalize(Path('bag'),'robot')
        self.assertEqual(records,[])
        self.assertEqual(mapping,[])
        self.assertEqual(audit['receipt_minus_measurement_ms'],{})
        reader.messages.assert_not_called()
