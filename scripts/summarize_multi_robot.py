#!/usr/bin/env python3
"""Summarize completed mission accuracy, geometry, timing and payload accounting."""
import argparse,json
from collections import Counter
from pathlib import Path
import numpy as np

def json_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

def key(value):return value['robot'],value['session'],value['id']
def pair(value):return tuple(key(v) for v in value)

def timing(events,field):
    values=[e[field] for e in events if field in e]
    return {'samples':len(values),'median_ms':float(np.median(values)),'p95_ms':float(np.percentile(values,95))} if values else None

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mission',type=Path)
    p.add_argument('--backfill-archive',action='store_true',help='Derive missing durable records from older experiment journals; never overwrite existing records')
    p.add_argument('--allow-incomplete-traffic',action='store_true',help='Report unavailable service counters explicitly for legacy runs with interrupted diagnostic writes')
    a=p.parse_args();root=a.mission.parent
    mission=json.loads(a.mission.read_text());replay=json.loads((root/'replay_summary.json').read_text())
    graph=json.loads((root/'backend/graph_snapshot.json').read_text());evaluation=json.loads((root/'evaluation.json').read_text())
    assert all(replay['robots'][r['robot']]['frames'] + replay['robots'][r['robot']].get('initialization_skipped_frames', 0) == r['frames'] for r in mission['robots']), 'replay did not account for every requested image'
    assert len(graph['poses'])==sum(s['keyframes'] for s in replay['robots'].values()),'final graph has missing keyframes'
    assert len(graph['loops'])==sum(s['loops'] for s in replay['robots'].values()),'final graph has missing verified constraints'
    sensor_seconds=max((r['original_last_ns']-r['original_first_ns'])/1e9 for r in mission['robots'])
    report={'alignment':'SE(3), one rigid alignment per connected component, no scale fit','components':graph['components'],
            'images':sum(r['frames'] for r in mission['robots']),
            'graph_keyframes':len(graph['poses']),'verified_constraints':len(graph['loops']),'final_solve_ms':graph['solve_ms'],
            'robots':{},'joint_by_component':evaluation['joint_by_component'],'wall_seconds':replay['wall_seconds'],
            'nominal_replay_rate':mission['rate'],'effective_replay_rate':sensor_seconds/replay['wall_seconds'],
            'observed_topic_cdr_bytes':replay['topic_cdr_bytes'],
            'topic_payload_note':replay['topic_cdr_bytes_note']+' These counters cover sequence announcements, loop constraints, and graph snapshots; other topics are not included. Latest-only graph subscriptions can skip revisions.'}
    # Accuracy acceptance is separate from successful replay/transport. Keep
    # an explicit reviewed assessment when regenerating a mission summary.
    assessment=root/'accuracy_assessment.json'
    if assessment.exists():
        report['accuracy_assessment']=json.loads(assessment.read_text())
    aggregate=Counter();service_totals=Counter();node_traffic={};unavailable_traffic={}
    def read_traffic(name,path):
        try:
            node_traffic[name]=json.loads(path.read_text())
        except (OSError,ValueError) as error:
            if not a.allow_incomplete_traffic:
                raise
            unavailable_traffic[name]={'path':str(path),'error':str(error)}
    journal=json_lines(root/'backend/graph.jsonl')
    revisions=json_lines(root/'backend/revisions.jsonl')
    solved=[r for r in revisions if 'final_cost' in r]
    assert all(np.isfinite(r['initial_cost']) and np.isfinite(r['final_cost'])
               and r['final_cost']<=r['initial_cost']+1e-9 for r in solved), 'invalid published graph objective'
    report['optimization']={'accepted_updates':len(solved),
        'rejected_updates':sum(r.get('event')=='rejected_solution' for r in revisions),
        'all_published_objectives_non_increasing':True,'solve_timing':timing(solved,'solve_ms'),
        'final_revision':graph['revision'],'final_input_revision':graph['input_revision']}
    for robot in mission['robots']:
        name=robot['robot'];directory=root/'robots'/name
        events=json_lines(directory/'events.jsonl');retrieval=json_lines(directory/'retrieval.jsonl')
        times=[e['process_ms'] for e in events if e['event']=='vio']
        selected={pair(e['pair']):e for e in retrieval if e['event']=='refinement'}
        verified=[e for e in events if e['event']=='verification']
        counts=Counter(e['diagnostic']['reason'] for e in verified);aggregate.update(counts)
        assert len(verified)==len({pair(e['pair']) for e in verified}),f'{name}: duplicate matching attempts'
        assert set(selected)=={pair(e['pair']) for e in verified},f'{name}: loop worker did not drain'
        accuracy=evaluation['robots'][name]
        report['robots'][name]={**replay['robots'][name],
            'ate_raw_se3_rmse_m':accuracy['raw']['ate_translation_se3_m']['rmse'],
            'ate_corrected_se3_rmse_m':accuracy['corrected']['ate_translation_se3_m']['rmse'],
            'vio_ms_median':float(np.median(times)),'vio_ms_p95':float(np.percentile(times,95)),
            'matching_attempts':len(verified),'verification_outcomes':dict(counts),
            'sequence_encoding':timing(events,'encoding_ms'),
            'matching_and_geometry':timing(verified,'matching_geometry_ms'),
            'exchange_through_verification':timing(verified,'exchange_through_verification_ms'),
            'inlier_counts_accepted':[e['diagnostic']['pnp_inliers'] for e in verified if e['diagnostic']['reason']=='verified']}
        if a.backfill_archive:
            records=[e['keyframe'] for e in journal if 'keyframe' in e and e['keyframe']['key']['robot']==name]
            loops=[e['constraint'] for e in events if e['event']=='accepted_loop']
            attempts=[e['pair'] for e in verified]
            for filename,values in [('keyframes.jsonl',records),('loops.jsonl',loops),('attempts.jsonl',attempts)]:
                destination=directory/filename
                if not destination.exists():
                    with destination.open('x') as stream:
                        for value in values:stream.write(json.dumps(value,separators=(',',':'))+'\n')
        read_traffic(name,directory/'communication.json')
    read_traffic('backend',root/'backend/communication.json')
    for traffic in node_traffic.values():
        service_totals.update({k:v for k,v in traffic.items() if isinstance(v,int)})
    report['verification_outcomes']=dict(aggregate);report['service_totals']=dict(service_totals)
    report['service_totals_complete']=not unavailable_traffic
    report['service_unavailable_nodes']=unavailable_traffic
    report['matching_attempts']=sum(aggregate.values())
    report['verification_success_rate']=aggregate['verified']/max(1,report['matching_attempts'])
    report['service_payload_note']='Typed field bytes counted at requesting nodes, excluding CDR lengths/padding, DDS overhead and retransmission.'
    if unavailable_traffic:
        report['service_payload_note']+=' PARTIAL totals: unavailable nodes are explicitly listed and excluded; their final counters have not been reconstructed.'
    report['service_by_node']=node_traffic
    comparison=root/'refinement_comparison.json'
    if comparison.exists():
        v=json.loads(comparison.read_text());report['refinement_comparison']={k:v[k] for k in ('identical_retrieval_pairs','refined','fixed_last')}
        report['controlled_batch_evaluation']={}
        for mode in ('refined','fixed_last'):
            path=root/'refinement'/mode/'evaluation.json'
            if path.exists():
                e=json.loads(path.read_text());report['controlled_batch_evaluation'][mode]={
                    'components':e['components'],'joint_by_component':e['joint_by_component'],
                    'per_robot_ate_m':{n:v['corrected']['ate_translation_se3_m']['rmse'] for n,v in e['robots'].items()}}
    (root/'validation_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('components','verified_constraints','robots','service_totals')},indent=2))

if __name__=='__main__':main()
