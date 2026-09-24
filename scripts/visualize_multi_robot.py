#!/usr/bin/env python3
"""Render final raw/optimized trajectories, sparse maps, and refined frame pairs."""
import argparse
import csv
import json
from pathlib import Path
import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from scipy.spatial.transform import Rotation
from run_graco_vio import Bag

def identity(key):
    return key['robot'], key['session'], key['id']

def apply(transform, points):
    return Rotation.from_quat(transform['rotation_xyzw']).apply(points) + transform['translation']

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mission', type=Path); p.add_argument('--connect')
    args = p.parse_args(); root = args.mission.parent
    mission = json.loads(args.mission.read_text()); graph = json.loads((root / 'backend/graph_snapshot.json').read_text())
    accuracy = ''
    if (root / 'evaluation.json').exists():
        evaluation = json.loads((root / 'evaluation.json').read_text())
        accuracy = '\nMetric ATE, raw -> corrected (m):\n' + '\n'.join(
            f"{name}: {value['raw']['ate_translation_se3_m']['rmse']:.2f} -> {value['corrected']['ate_translation_se3_m']['rmse']:.2f}"
            for name, value in evaluation['robots'].items())
    rr.init('visloc multi-robot Rust ROS2 SLAM', spawn=False)
    sinks=[rr.FileSink(root / 'playback.rrd')]
    if args.connect: sinks.append(rr.GrpcSink(args.connect))
    rr.set_sinks(*sinks)
    colors = [[60,160,255],[255,140,50],[90,220,130],[210,100,240]]
    pose_by_key = {identity(p['key']): p for p in graph['poses']}
    components = {}
    for pose in graph['poses']:
        components.setdefault(identity(pose['component']), set()).add(pose['key']['robot'])
    paths = {k: f'components/{k[0]}_{k[1]}_{k[2]}' for k in components}
    views = [rrb.Spatial3DView(origin=paths[k], name='Component: '+', '.join(sorted(robots)))
             for k, robots in components.items()]
    rr.send_blueprint(rrb.Blueprint(rrb.Vertical(rrb.Horizontal(*views), rrb.Horizontal(
        rrb.Spatial2DView(origin='verification/from', name='Refined frame A'),
        rrb.Spatial2DView(origin='verification/to', name='Refined frame B'),
        rrb.TextDocumentView(origin='status', name='SLAM status')), row_shares=[.62,.38]),
        rrb.TimePanel(timeline='verification', fps=2, play_state='paused'), collapse_panels=True))
    archives, records, bags, diagnostics = {}, {}, {}, {}
    for path in paths.values():
        rr.log(path, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    for robot, color in zip(mission['robots'], colors):
        name = robot['robot']
        with (root / 'robots' / name / 'trajectory.csv').open() as f:
            rows = list(csv.DictReader(f))
        raw = np.array([[float(row[c]) for c in ('tx','ty','tz')] for row in rows])
        poses = sorted((p for p in graph['poses'] if p['key']['robot']==name), key=lambda p:p['timestamp_ns'])
        if poses:
            origin = paths[identity(poses[0]['component'])]
            rr.log(f'{origin}/{name}/optimized', rr.LineStrips3D([[p['body_to_map']['translation'] for p in poses]], colors=color), static=True)
            rr.log(f'{origin}/{name}/raw', rr.LineStrips3D([apply(poses[0]['map_from_odom'],raw)], colors=[*color,80]), static=True)
        for path in sorted((root / 'robots' / name / 'sequences').glob('*.json')):
            for feature in json.loads(path.read_text())['features']:
                k = identity(feature['key']); archives[k] = feature
                if k in pose_by_key:
                    points = np.array([v for v in feature['points_camera'] if v is not None]).reshape(-1,3)
                    if len(points):
                        body = apply(feature['camera']['camera_to_body'], points)
                        cloud = apply(pose_by_key[k]['body_to_map'], body)
                        rr.log(f'{paths[identity(pose_by_key[k]["component"])]}/{name}/landmarks/{k[2]}', rr.Points3D(cloud, colors=color, radii=rr.Radius.ui_points(1.3)), static=True)
        for line in (root / 'robots' / name / 'retrieval.jsonl').read_text().splitlines():
            e = json.loads(line)
            if e.get('event') == 'refinement':
                records[(identity(e['from']), identity(e['to']))] = e
        for line in (root / 'robots' / name / 'events.jsonl').read_text().splitlines():
            e = json.loads(line)
            if e.get('event') == 'verification':
                diagnostics[tuple(identity(k) for k in e['pair'])] = e['diagnostic']
        bags[name] = (Bag(Path(robot['bag'])), robot)
    edges={}
    for edge in graph['loops']:
        a,b=pose_by_key.get(identity(edge['from'])),pose_by_key.get(identity(edge['to']))
        if a and b and identity(a['component']) == identity(b['component']):
            edges.setdefault(paths[identity(a['component'])],[]).append([a['body_to_map']['translation'], b['body_to_map']['translation']])
    for origin, lines in edges.items():
        rr.log(f'{origin}/loop_edges', rr.LineStrips3D(lines, colors=[255,50,170]), static=True)
    for index, ((ka,kb), event) in enumerate(records.items()):
        rr.set_time('verification', sequence=index)
        for side,k in [('from',ka),('to',kb)]:
            feature=archives.get(k)
            if not feature:
                continue
            bag,robot=bags[k[0]]; original=feature['timestamp_ns']-robot['offset_ns']
            tid,_=bag.topics['/camera_left/image_raw']
            row=bag.connection.execute('SELECT data FROM messages WHERE topic_id=? AND timestamp=?',(tid,original)).fetchone()
            if row is None:
                continue
            raw=bag.types.deserialize_cdr(row[0],'sensor_msgs/msg/Image')
            image=np.asarray(raw.data).reshape(raw.height,raw.step)[:,:raw.width]
            config=json.loads(Path(robot['config']).read_text()); prep=config['preprocess'];cam=feature['camera']
            fx,fy,cx,cy=prep['intrinsics']; K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
            maps=cv2.initUndistortRectifyMap(K,np.array(prep['distortion']),np.eye(3),K,(cam['width'],cam['height']),cv2.CV_32FC1)
            image=cv2.remap(cv2.resize(image,(cam['width'],cam['height']),interpolation=cv2.INTER_AREA),*maps,cv2.INTER_LINEAR)
            rr.log(f'verification/{side}/image',rr.Image(image).compress(jpeg_quality=85))
            rr.log(f'verification/{side}/observations',rr.Points2D(feature['pixels'],colors=[255,200,40],radii=2))
        diagnostic=diagnostics.get(tuple(identity(k) for k in event['pair']),{})
        outcome=f"Verification: {diagnostic.get('reason','unavailable')}\nMatches / F inliers / PnP inliers: {diagnostic.get('matches',0)} / {diagnostic.get('two_d_inliers',0)} / {diagnostic.get('pnp_inliers',0)}"
        if diagnostic.get('reason')=='verified':
            outcome+=f"\nMean reprojection error: {diagnostic['reprojection_px']:.3f} px\nPnP direction: {'reverse' if diagnostic['reverse'] else 'forward'}"
        rr.log('status',rr.TextDocument(f"Components: {graph['components']}\nComponent membership: {[sorted(v) for v in components.values()]}\nVerified constraints: {len(graph['loops'])}\nSelected frames: {ka[0]}:{ka[2]} and {kb[0]}:{kb[2]}\nJIST sequence similarity: {event['similarity']:.4f}\nFrame similarity: {event['frame_similarity']:.4f}\n{outcome}{accuracy}"))
    if not records:
        rr.log('status',rr.TextDocument(f"Components: {graph['components']}\nVerified constraints: {len(graph['loops'])}\nNo frame pairs reached refinement.{accuracy}"),static=True)
    for bag,_ in bags.values():
        bag.connection.close()
    print(root / 'playback.rrd')

if __name__ == '__main__':
    main()
