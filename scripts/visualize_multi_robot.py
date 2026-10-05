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
from replay_sensor_source import open_source
from landmark_visualization import build_landmark_map

def identity(key):
    return key['robot'], key['session'], key['id']

def apply(transform, points):
    return Rotation.from_quat(transform['rotation_xyzw']).apply(points) + transform['translation']

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mission', type=Path); p.add_argument('--connect')
    p.add_argument('--output', type=Path, help='recording destination (default: mission/playback.rrd)')
    p.add_argument('--landmark-min-observations', type=int, default=2)
    p.add_argument('--landmark-min-parallax-deg', type=float, default=1.0)
    p.add_argument('--landmark-max-reprojection-px', type=float, default=3.0)
    p.add_argument('--show-archived-landmarks', action='store_true',
                   help='also log unfused per-keyframe clouds as a diagnostic layer')
    args = p.parse_args(); root = args.mission.parent
    output = args.output or root / 'playback.rrd'
    output.parent.mkdir(parents=True, exist_ok=True)
    mission = json.loads(args.mission.read_text()); graph = json.loads((root / 'backend/graph_snapshot.json').read_text())
    accuracy = ''
    if (root / 'evaluation.json').exists():
        evaluation = json.loads((root / 'evaluation.json').read_text())
        accuracy = '\nMetric ATE, raw -> corrected (m):\n' + '\n'.join(
            f"{name}: {value['raw']['ate_translation_se3_m']['rmse']:.2f} -> {value['corrected']['ate_translation_se3_m']['rmse']:.2f}"
            for name, value in evaluation['robots'].items())
    rr.init('visloc multi-robot Rust ROS2 SLAM', spawn=False)
    sinks=[rr.FileSink(output)]
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
    robot_colors = {robot['robot']: colors[i % len(colors)] for i, robot in enumerate(mission['robots'])}
    for robot in mission['robots']:
        name = robot['robot']
        color = robot_colors[name]
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
                if args.show_archived_landmarks and k in pose_by_key:
                    points = np.array([v for v in feature['points_camera'] if v is not None]).reshape(-1,3)
                    if len(points):
                        body = apply(feature['camera']['camera_to_body'], points)
                        cloud = apply(pose_by_key[k]['body_to_map'], body)
                        rr.log(f'{paths[identity(pose_by_key[k]["component"])]}/{name}/archived_landmarks/{k[1]}/{k[2]}', rr.Points3D(cloud, colors=[*color,60], radii=rr.Radius.ui_points(.8)), static=True)
        for line in (root / 'robots' / name / 'retrieval.jsonl').read_text().splitlines():
            e = json.loads(line)
            if e.get('event') == 'refinement':
                records[(identity(e['from']), identity(e['to']))] = e
        for line in (root / 'robots' / name / 'events.jsonl').read_text().splitlines():
            e = json.loads(line)
            if e.get('event') == 'verification':
                diagnostics[tuple(identity(k) for k in e['pair'])] = e['diagnostic']
        bags[name] = (open_source(robot), robot)
    landmarks, landmark_audit = build_landmark_map(
        archives.values(), pose_by_key,
        min_observations=args.landmark_min_observations,
        min_parallax_deg=args.landmark_min_parallax_deg,
        reprojection_px=args.landmark_max_reprojection_px)
    clouds = {}
    for point in landmarks:
        clouds.setdefault((point['component'], point['robot'], point['session']), []).append(point)
    for (component, name, session), points in clouds.items():
        entity = f'{paths[component]}/{name}/landmarks/{session}'
        rr.log(entity,
               rr.Points3D([p['position'] for p in points], colors=robot_colors[name],
                           radii=rr.Radius.ui_points(1.0)),
               rr.AnyValues(landmark_id=[str(p['track_id']) for p in points],
                            observing_keyframes=[p['observations'] for p in points],
                            inlier_views=[p['inliers'] for p in points],
                            reprojection_rmse_px=[p['reprojection_px'] for p in points],
                            parallax_deg=[p['parallax_deg'] for p in points]), static=True)
    landmark_audit['graph_revision'] = graph['revision']
    landmark_audit['recording'] = str(output)
    output.with_suffix('.landmarks.json').write_text(json.dumps(landmark_audit, indent=2)+'\n')
    with output.with_suffix('.landmarks.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['component_robot','component_session','component_id','robot','session',
                         'landmark_id','x','y','z','observing_keyframes','inlier_views',
                         'reprojection_rmse_px','parallax_deg'])
        for point in landmarks:
            writer.writerow([*point['component'],point['robot'],point['session'],point['track_id'],
                             *point['position'],point['observations'],point['inliers'],
                             point['reprojection_px'],point['parallax_deg']])
    landmark_status = (f"\nDisplay landmarks: {len(landmarks)} / {landmark_audit['landmark_groups']} identities"
                       f"\nFixed-pose multi-view refinement; >= {args.landmark_min_observations} views, "
                       f">= {args.landmark_min_parallax_deg:g} deg parallax, <= {args.landmark_max_reprojection_px:g} px."
                       "\nLandmarks are refined for display, not jointly optimized by the backend.")
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
            try:
                raw=bag.image_at(original)
            except KeyError:
                continue
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
        rr.log('status',rr.TextDocument(f"Components: {graph['components']}\nComponent membership: {[sorted(v) for v in components.values()]}\nVerified constraints: {len(graph['loops'])}\nSelected frames: {ka[0]}:{ka[2]} and {kb[0]}:{kb[2]}\nJIST sequence similarity: {event['similarity']:.4f}\nFrame similarity: {event['frame_similarity']:.4f}\n{outcome}{accuracy}{landmark_status}"))
    if not records:
        rr.log('status',rr.TextDocument(f"Components: {graph['components']}\nVerified constraints: {len(graph['loops'])}\nNo frame pairs reached refinement.{accuracy}{landmark_status}"),static=True)
    for bag,_ in bags.values():
        bag.close()
    print(json.dumps(landmark_audit, indent=2))
    print(output)

if __name__ == '__main__':
    main()
