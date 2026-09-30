#!/usr/bin/env python3
"""Read a running Rust robot's typed GNSS topics, including late-join ephemerides."""
import argparse
import json
import time
from pathlib import Path
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from visloc_msgs.msg import GnssEpoch, GnssEphemeris, GnssStatus


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot', default='ucy01'); parser.add_argument('--seconds', type=float, default=10)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); rclpy.init(); node = rclpy.create_node('gnss_contract_check')
    results = dict(epochs=0, ephemerides=0, statuses=0, systems=[], satellite_ids=[], last_status=None)
    systems, satellites = set(), set()
    def epoch(e):
        assert 0 <= e.tow_s < 604800 and e.week > 0 and e.leap_seconds_valid
        for o in e.observations:
            assert o.gnss_id in (0, 2) and o.sv_id > 0 and o.pseudorange_std_m > 0 and o.doppler_std_hz > 0
            systems.add(o.gnss_id)
        results['epochs'] += 1
    def ephemeris(e):
        assert e.gnss_id in (0, 2) and 4000 < e.sqrt_a < 6000 and 0 <= e.eccentricity < 1
        results['ephemerides'] += 1; satellites.add((e.gnss_id, e.sv_id))
    def status(e):
        results['statuses'] += 1
        results['last_status'] = dict(initialized=e.initialized, status=e.status, timestamp_ns=e.timestamp_ns,
            time_offset_s=e.time_offset_s, timing_std_s=e.timing_std_s, epochs=e.epochs,
            accepted_dopplers=e.accepted_dopplers, accepted_pseudoranges=e.accepted_pseudoranges,
            lever_arm_imu_m=list(e.lever_arm_imu_m))
    reliable = QoSProfile(depth=128, reliability=ReliabilityPolicy.RELIABLE)
    retained = QoSProfile(depth=128, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    subscriptions = [node.create_subscription(GnssEpoch, f'/{args.robot}/gnss/epochs', epoch, reliable),
        node.create_subscription(GnssEphemeris, f'/{args.robot}/gnss/ephemerides', ephemeris, retained),
        node.create_subscription(GnssStatus, f'/{args.robot}/gnss/status', status, retained)]
    deadline = time.monotonic() + args.seconds
    while time.monotonic() < deadline: rclpy.spin_once(node, timeout_sec=.1)
    assert results['epochs'] > 0 and results['ephemerides'] > 0 and results['statuses'] > 0, results
    assert systems == {0, 2}, systems
    results['systems'] = sorted(systems); results['satellite_ids'] = sorted(satellites)
    args.output.write_text(json.dumps(results, indent=2)+'\n'); print(json.dumps(results, indent=2))
    node.destroy_node(); rclpy.shutdown()


if __name__ == '__main__': main()
