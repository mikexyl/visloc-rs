#!/usr/bin/env python3
"""External evo evaluation of frozen GPS-PGO ablations. PPK is evaluation-only."""
import argparse
import csv
from decimal import Decimal
import importlib.metadata
import io
import json
import subprocess
import sys
from pathlib import Path
import zipfile
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def tum(path, times, positions, rotations):
    with path.open('w') as stream:
        for ns,p,q in zip(times,positions,rotations):
            sec,nsec=divmod(int(ns),10**9)
            stream.write(f'{sec}.{nsec:09d} '+' '.join(f'{v:.12f}' for v in [*p,*q])+'\n')


def enu_rotation(lat,lon):
    lat,lon=np.radians([lat,lon]);s,c=np.sin(lat),np.cos(lat);sl,cl=np.sin(lon),np.cos(lon)
    return np.array([[-sl,cl,0],[-s*cl,-s*sl,c],[c*cl,c*sl,s]])


def ecef(lla):
    lat,lon=np.radians(lla[:2]);h=lla[2];e2=6.6943799901413165e-3;n=6378137/np.sqrt(1-e2*np.sin(lat)**2)
    return np.array([(n+h)*np.cos(lat)*np.cos(lon),(n+h)*np.cos(lat)*np.sin(lon),(n*(1-e2)+h)*np.sin(lat)])


def associate(times, reference):
    pairs=[];used=set()
    for j,t in enumerate(reference):
        i=int(np.searchsorted(times,t));choices=[k for k in [i-1,i] if 0<=k<len(times) and k not in used]
        if not choices:continue
        i=min(choices,key=lambda k:abs(int(times[k])-int(t)))
        if abs(int(times[i])-int(t))<=20_000_000:pairs.append((i,j));used.add(i)
    return np.array(pairs,dtype=int).T


def run_evo(output,reference,estimate,name):
    result=output/f'{name}.zip'
    command=[str(Path(sys.executable).parent/'evo_ape'),'tum',str(reference),str(estimate),'--align','--t_max_diff','0.020','--save_results',str(result),'--no_warnings']
    completed=subprocess.run(command,text=True,capture_output=True,check=True)
    (output/f'{name}.log').write_text(completed.stdout+completed.stderr)
    with zipfile.ZipFile(result) as archive:
        stats=json.loads(archive.read('stats.json'))
        alignment=np.load(io.BytesIO(archive.read('alignment_transformation_sim3.npy')))
        errors=np.load(io.BytesIO(archive.read('error_array.npy')))
    assert np.allclose(alignment[:3,:3].T@alignment[:3,:3],np.eye(3),atol=1e-9)
    assert np.isclose(np.linalg.det(alignment[:3,:3]),1.,atol=1e-9)
    return stats,alignment,errors,command


def displacement_rpe(times,points,truth,seconds):
    segments=np.cumsum(np.r_[0,np.diff(times)>500_000_000]);errors=[]
    for i,t in enumerate(times):
        j=int(np.searchsorted(times,t+int(seconds*1e9)))
        choices=[k for k in [j-1,j] if i<k<len(times) and segments[k]==segments[i]]
        if not choices:continue
        j=min(choices,key=lambda k:abs(int(times[k]-t)-int(seconds*1e9)))
        if abs(int(times[j]-t)-int(seconds*1e9))>50_000_000:continue
        errors.append(np.linalg.norm((points[j]-points[i])-(truth[j]-truth[i])))
    return dict(pairs=len(errors),rmse_m=float(np.sqrt(np.mean(np.square(errors)))) if errors else None)


def evaluate(root,reference_csv,include_online=False,online_label='Online loops + GPS (final graph)'):
    out=root/('evaluation_online' if include_online else 'evaluation');out.mkdir(exist_ok=False)
    source=json.loads((root/'source_mission.json').read_text());robot=source['robots'][0];original=Path(robot['config']).parent
    raw=list(csv.DictReader((original/'robots'/robot['robot']/'trajectory.csv').open()))
    times=np.array([int(r['timestamp_ns']) for r in raw],dtype=np.int64)
    original_times=times-int(robot['offset_ns'])
    positions=np.array([[float(r[k]) for k in ('tx','ty','tz')] for r in raw])
    rotations=Rotation.from_quat([[float(r[k]) for k in ('qx','qy','qz','qw')] for r in raw])
    gt=list(csv.DictReader(reference_csv.open()));rt=np.array([int(Decimal(r['t'])*10**9) for r in gt],dtype=np.int64)
    xyz=np.array([[float(r[k]) for k in ('ecef_x_m','ecef_y_m','ecef_z_m')] for r in gt])
    rotation=enu_rotation(float(gt[0]['lat']),float(gt[0]['lon']));ref=(xyz-xyz[0])@rotation.T
    reference=out/'ppk_reference.tum';tum(reference,rt,ref,np.tile([0.,0.,0.,1.],(len(rt),1)))
    arm=np.array(json.loads((root/'gps_calibration.json').read_text())['imu_lever_m'])
    labels={'raw':'Raw VIO','loops':'Loops','gps_huber':'GPS Huber','gps_switchable':'GPS switchable','loops_gps_huber':'Loops + GPS Huber','loops_gps_switchable':'Loops + GPS switchable'}
    if include_online:labels['online_gps_switchable']=online_label
    report={'evo_version':importlib.metadata.version('evo'),'reference_rows':len(rt),'strict_quality_rows':sum(r['quality_accepted']=='True' for r in gt),
            'reference_coverage_s':[float((rt[0]-original_times[0])*1e-9),float((rt[-1]-original_times[0])*1e-9)],
            'unreferenced_tail_s':float((original_times[-1]-rt[-1])*1e-9),'alignment':'Rigid SE3, scale=1; no fitted time offset; 20 ms association',
            'reference_limitations':'Partial PPK reference from the same rover observations; no supplied rows pass the stricter quality flag. Nominal lever arm, not a calibrated phase centre.',
            'rpe_definition':'1 s and 5 s ENU displacement-vector error after the ATE rigid alignment; no reference orientations are available; do not cross reference gaps.',
            'full':{},'keyframes':{},'uncompensated':{},'commands':[]}
    curves={};gps_diagnostics={};common={}
    for name,label in labels.items():
        graph_path=(root/'gps_guarded/backend/graph_snapshot.json') if name=='online_gps_switchable' else (root/'frozen'/name/'graph_snapshot.json')
        graph=json.loads(graph_path.read_text());poses=sorted(graph['poses'],key=lambda p:p['timestamp_ns'])
        knots=np.array([p['timestamp_ns'] for p in poses],dtype=np.int64);u=(times-knots[0])*1e-9;k=(knots-knots[0])*1e-9
        cr=Rotation.from_quat([p['map_from_odom']['rotation_xyzw'] for p in poses]);ct=np.array([p['map_from_odom']['translation'] for p in poses])
        correction=Slerp(k,cr)(np.clip(u,k[0],k[-1]));translation=np.column_stack([np.interp(u,k,ct[:,d]) for d in range(3)])
        body=correction.apply(positions)+translation;q=correction*rotations;antenna=body+q.apply(arm)
        kbody=np.array([p['body_to_map']['translation'] for p in poses]);kq=Rotation.from_quat([p['body_to_map']['rotation_xyzw'] for p in poses]);kant=kbody+kq.apply(arm)
        for mode,ts,points,quats in [('full',original_times,antenna,q.as_quat()),('keyframes',knots-int(robot['offset_ns']),kant,kq.as_quat()),('uncompensated',original_times,body,q.as_quat())]:
            path=out/f'{name}_{mode}.tum';tum(path,ts,points,quats)
            stats,align,errors,command=run_evo(out,reference,path,f'{name}_{mode}_ape');report['commands'].append(command)
            i,j=associate(ts,rt)
            if mode in common:assert np.array_equal(common[mode],j)
            common[mode]=j
            aligned=points@align[:3,:3].T+align[:3,3];residual=aligned[i]-ref[j]
            assert len(errors)==len(i) and np.allclose(errors,np.linalg.norm(residual,axis=1),atol=1e-7)
            stats.update(samples=len(i),horizontal_rmse_m=float(np.sqrt(np.mean(np.sum(residual[:,:2]**2,axis=1)))),vertical_rmse_m=float(np.sqrt(np.mean(residual[:,2]**2))),alignment_SE3=align.tolist(),max_association_ms=float(np.max(abs(ts[i]-rt[j]))/1e6))
            if mode=='full':
                stats['rpe_1s']=displacement_rpe(rt[j],aligned[i],ref[j],1)
                stats['rpe_5s']=displacement_rpe(rt[j],aligned[i],ref[j],5)
                deformation=cr[0].inv().apply(body-ct[0])-positions
                stats['deformation_median_m']=float(np.median(np.linalg.norm(deformation,axis=1)))
                stats['deformation_max_m']=float(np.max(np.linalg.norm(deformation,axis=1)))
                datum=graph.get('gps',{}).get('datum')
                if datum and graph['gps']['aligned_components']:
                    d=[datum['latitude'],datum['longitude'],datum['altitude']];absolute_ref=(xyz-ecef(d))@enu_rotation(d[0],d[1]).T
                    stats['global_horizontal_rmse_m']=float(np.sqrt(np.mean(np.sum((points[i,:2]-absolute_ref[j,:2])**2,axis=1))))
                curves[name]=(aligned,rt[j],errors,stats['rmse'])
            report[mode][name]=stats
        gps_diagnostics[name]=graph.get('gps',{}).get('diagnostics',[])
    report['matched_full_reference_rows']=len(common['full']);report['matched_keyframe_reference_rows']=len(common['keyframes'])
    report['gps']={name:dict(reasons=dict(__import__('collections').Counter(d['reason'] for d in ds)),downweighted=sum(d.get('robust_weight') is not None and d['robust_weight']<.25 for d in ds)) for name,ds in gps_diagnostics.items()}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,axs=plt.subplots(2,2,figsize=(15,10));colors=plt.cm.tab10(np.linspace(0,1,len(labels)))
    for (name,label),color in zip(labels.items(),colors):
        aligned,ct,errors,rmse=curves[name]
        axs[0,0].plot(aligned[:,0],aligned[:,1],color=color,lw=1,label=label)
        cuts=np.r_[0,np.flatnonzero(np.diff(ct)>500_000_000)+1,len(ct)]
        for segment,(a,b) in enumerate(zip(cuts[:-1],cuts[1:])):
            axs[0,1].plot((ct[a:b]-original_times[0])*1e-9,errors[a:b],color=color,lw=1,label=f'{label}: {rmse:.3f} m' if segment==0 else None)
        ds=[d for d in gps_diagnostics[name] if d.get('robust_weight') is not None]
        if ds and name in ('gps_huber','gps_switchable'):
            axs[1,0].plot([(d['timestamp_ns']-times[0])*1e-9 for d in ds],[d['robust_weight'] for d in ds],'.-',color=color,markersize=2,label=label)
    gaps=np.flatnonzero(np.diff(rt)>.5e9)
    cut=np.r_[0,gaps+1,len(rt)]
    for a,b in zip(cut[:-1],cut[1:]):axs[0,0].plot(ref[a:b,0],ref[a:b,1],color='black',lw=2)
    axs[0,0].set(title='Full final-graph trajectories (rigidly aligned)',xlabel='East (m)',ylabel='North (m)');axs[0,0].axis('equal');axs[0,0].legend(fontsize=8)
    axs[0,1].set(title='evo ATE — antenna compensation, scale fixed to 1',xlabel='Recorded time (s)',ylabel='Position error (m)');axs[0,1].legend(fontsize=8)
    axs[0,1].axvspan((rt[-1]-original_times[0])*1e-9,(original_times[-1]-original_times[0])*1e-9,color='gray',alpha=.15)
    axs[1,0].set(title='GPS robustness (tolerance applied separately)',xlabel='Recorded time (s)',ylabel='Robust-loss weight',ylim=(-.02,1.02));axs[1,0].legend(fontsize=8)
    names=list(labels);axs[1,1].barh([labels[n] for n in names],[report['full'][n]['rmse'] for n in names],color=colors)
    axs[1,1].set(title='Matched full trajectory ATE RMSE',xlabel='metres')
    for ax in axs.flat:ax.grid(alpha=.15)
    fig.suptitle('South-ece GPS pose-graph experiment — frozen VIO and loop factors',fontsize=15)
    fig.text(.5,.015,f"{report['matched_full_reference_rows']} matched reference rows; incomplete, correlated PPK reference. Grey area: no reference. GPS uncertainty is assumed.",ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.035,1,.95));fig.savefig(out/'comparison.png',dpi=160);fig.savefig(out/'comparison.pdf');plt.close(fig)
    lines=['# South-ece GPS pose-graph ablation','',report['alignment'],'','| Method | Full ATE (m) | Keyframe ATE (m) | 1 s displacement RPE (m) | Global horizontal error (m) |','|---|---:|---:|---:|---:|']
    for name,label in labels.items():
        f=report['full'][name];g=f.get('global_horizontal_rmse_m');lines.append(f"| {label} | {f['rmse']:.3f} | {report['keyframes'][name]['rmse']:.3f} | {f['rpe_1s']['rmse_m']:.3f} | {g:.3f} |" if g is not None else f"| {label} | {f['rmse']:.3f} | {report['keyframes'][name]['rmse']:.3f} | {f['rpe_1s']['rmse_m']:.3f} | Unreferenced |")
    lines+=['',f"Coverage: {report['matched_full_reference_rows']}/{len(rt)} supplied reference rows for full trajectories; {report['matched_keyframe_reference_rows']} at keyframe timestamps.",'',report['reference_limitations'],'',report['rpe_definition'],'','Full trajectories apply the final graph correction to every raw VIO pose. They are not historical online estimates. Each method uses identical VIO, keyframes, loop candidates, and reference associations.','',f"Uncompensated continuity check: raw {report['uncompensated']['raw']['rmse']:.6f} m, loops {report['uncompensated']['loops']['rmse']:.6f} m.",'','![Comparison](comparison.png)']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({name:{k:v for k,v in row.items() if k in ('rmse','samples','global_horizontal_rmse_m','rpe_1s','deformation_max_m')} for name,row in report['full'].items()},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('result',type=Path);p.add_argument('reference',type=Path);p.add_argument('--include-online',action='store_true');p.add_argument('--online-label',default='Online loops + GPS (final graph)');a=p.parse_args();evaluate(a.result.resolve(),a.reference.resolve(),a.include_online,a.online_label)
