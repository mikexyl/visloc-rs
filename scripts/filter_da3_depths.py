#!/usr/bin/env python3
"""WIP: apply confidence/reprojection filters to saved DA3 windows and compare in Rerun.

This reuses archived network depth, metric scale, poses, intrinsics and confidence. It
does not rerun VIO/inference, fit scale, change cameras, or reschedule windows.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from da3_geometry import filter_depth_confidence, filter_depth_reprojection
from online_da3 import Config


def cloud(depth, gray, intrinsics, poses, stride):
    y,x = np.mgrid[0:depth.shape[1]:stride,0:depth.shape[2]:stride]
    points,colors = [],[]
    for i in range(5):
        z = depth[i,y,x]
        valid = np.isfinite(z) & (z > 0)
        k = intrinsics[i]
        p = np.column_stack(((x[valid]-k[0,2])*z[valid]/k[0,0],
                             (y[valid]-k[1,2])*z[valid]/k[1,1],z[valid]))
        points.append(p @ poses[i,:3,:3].T + poses[i,:3,3])
        colors.append(np.repeat(gray[i,y,x][valid,None],3,axis=1))
    return np.concatenate(points),np.concatenate(colors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True,help='Existing replay directory containing da3/window_*.npz')
    parser.add_argument('--output',type=Path,required=True,help='New output directory')
    parser.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'configs/graco/da3_five_view.json')
    parser.add_argument('--trajectory',type=Path,help='Optional original VIO trajectory when source is a previous filtering output')
    args = parser.parse_args()
    config = Config.from_path(args.config)
    if not config.reprojection_filter:
        parser.error('The supplied config must enable reprojection_filter')
    paths = sorted((args.source/'da3').glob('window_*.npz'))
    if not paths:
        parser.error('No archived DA3 windows found')
    args.output.mkdir(parents=True,exist_ok=False)
    archives = args.output/'da3'
    archives.mkdir()
    (args.output/'config.json').write_text(json.dumps(asdict(config),indent=2)+'\n')
    rr.init('DA3 depth WIP filter comparison')
    rr.save(str(args.output/'filter_comparison.rrd'))
    for branch in ('before','after'):
        rr.log(branch,rr.ViewCoordinates.RIGHT_HAND_Z_UP,static=True)
    rows,extents = [],[]
    try:
        for index,path in enumerate(paths):
            with np.load(path,allow_pickle=False) as archive:
                data = dict(archive)
            depth = data['depth_m']
            metric_depth = data['raw_depth'] * float(data['scale'])
            metadata = json.loads(str(data['metadata']))
            started = time.monotonic()
            confident,confidence_report = filter_depth_confidence(
                metric_depth,data['confidence'],min_confidence=config.min_confidence,
                percentile=config.confidence_percentile,max_depth_m=config.max_depth_m)
            filtered,support,report = filter_depth_reprojection(
                confident,data['intrinsics'],data['camera_to_world'],
                max_error_px=config.max_reprojection_error_px,
                max_relative_depth=config.max_reprojection_depth_error,
                min_consistent_views=config.min_consistent_views)
            report.update(filter_ms=(time.monotonic()-started)*1000,
                          source_archive=str(path.resolve()),frame_ids=data['frame_ids'].tolist(),
                          baseline_pixels=int(np.isfinite(depth).sum()),confidence_filter=confidence_report,
                          online_would_accept=bool(np.isfinite(filtered).mean() >= .10))
            # A filter must never adjust retained depths or turn invalid inputs into points.
            kept = np.isfinite(filtered)
            np.testing.assert_array_equal(filtered[kept],metric_depth[kept])
            assert not np.any(kept & ~np.isfinite(confident))
            metadata.update(consistency=report,confidence_filter=confidence_report,postprocess_only=True)
            data.update(depth_unfiltered_m=metric_depth,depth_m=filtered,depth_support=support,
                        confidence_mask=np.isfinite(confident),
                        consistency_mask=kept,metadata=np.array(json.dumps(metadata,allow_nan=False)))
            np.savez_compressed(archives/path.name,**data)
            counts = []
            for branch,values in (('before',depth),('after',filtered)):
                points,colors = cloud(values,data['gray'],data['intrinsics'],data['camera_to_world'],config.cloud_stride)
                counts.append(len(points))
                rr.log(f'{branch}/window_{index:04d}',
                       rr.Points3D(points,colors=colors,radii=rr.Radius.ui_points(1.1)),static=True)
                if branch == 'before' and len(points):
                    extents.append(np.quantile(points,[.01,.99],axis=0))
            report.update(displayed_before=counts[0],displayed_after=counts[1])
            rows.append(report)
            print(f'{index+1}/{len(paths)}: retained {report["retained_fraction"]:.1%}; {report["filter_ms"]:.1f} ms',flush=True)
        trajectory = args.trajectory if args.trajectory is not None else args.source/'trajectory.csv'
        if trajectory.is_file():
            values = np.genfromtxt(trajectory,delimiter=',',names=True)
            xyz = np.column_stack([values[name] for name in ('tx','ty','tz')])
            for branch in ('before','after'):
                rr.log(branch+'/vio',rr.LineStrips3D([xyz],colors=[40,160,255],radii=rr.Radius.ui_points(1.2)),static=True)
        total_in = sum(r['confidence_filter']['input_pixels'] for r in rows)
        total_out = sum(r['retained_pixels'] for r in rows)
        total_baseline = sum(r['baseline_pixels'] for r in rows)
        report = dict(source=str(args.source.resolve()),mode='offline_filter_same_windows',
                      windows=len(rows),input_pixels=total_in,retained_pixels=total_out,
                      retained_fraction=total_out/total_in if total_in else 0.,
                      baseline_pixels=total_baseline,
                      retained_fraction_of_baseline=total_out/total_baseline if total_baseline else 0.,
                      confident_pixels=sum(r['confidence_filter']['retained_pixels'] for r in rows),
                      displayed_before=sum(r['displayed_before'] for r in rows),
                      displayed_after=sum(r['displayed_after'] for r in rows),
                      filter_ms_median=float(np.median([r['filter_ms'] for r in rows])),
                      windows_below_online_minimum=sum(not r['online_would_accept'] for r in rows),
                      camera_poses_and_scale='unchanged from input archives',by_window=rows)
        (args.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        overview = (f"DA3 depth — WIP. Same {len(rows)} windows; camera poses and depth scale unchanged. "
                    f"Retained {report['retained_fraction']:.1%} of original valid pixels.\n"
                    f"Confidence >= max({config.min_confidence:g}, window percentile {config.confidence_percentile:g}) before reprojection. "
                    f"At least {config.min_consistent_views} other views must agree: "
                    f"round-trip error <= {config.max_reprojection_error_px} px, "
                    f"relative depth error <= {config.max_reprojection_depth_error:.0%}. "
                    "This checks agreement within each five-keyframe window.")
        rr.log('notes',rr.TextDocument(overview),static=True)
        if extents:
            extents = np.concatenate(extents)
            center = (extents.min(0)+extents.max(0))/2
            span = max(float(np.ptp(extents,axis=0).max()),1.)
            eye = rrb.EyeControls3D(position=center+[0,-.6*span,span],look_target=center,eye_up=[0,0,1])
        else:
            eye = None
        rr.send_blueprint(rrb.Blueprint(rrb.Vertical(rrb.Horizontal(
            rrb.Spatial3DView(origin='before',name='Before: saved DA3 map',line_grid=False,background=[25,28,33],eye_controls=eye),
            rrb.Spatial3DView(origin='after',name='After: confidence + reprojection filters',line_grid=False,background=[25,28,33],eye_controls=eye)),
            rrb.TextDocumentView(origin='notes',name='Filter settings'),row_shares=[9,1]),
            rrb.BlueprintPanel(expanded=False),rrb.SelectionPanel(expanded=False)))
        print(json.dumps({k:v for k,v in report.items() if k != 'by_window'},indent=2))
    finally:
        rr.disconnect()


if __name__ == '__main__':
    main()
