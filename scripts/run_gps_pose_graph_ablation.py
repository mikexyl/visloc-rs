#!/usr/bin/env python3
"""Six controlled horizontal-GPS ablations; all use frozen raw VIO and loops."""
import argparse
import json
from pathlib import Path
import subprocess


def run(root,config_path,binary):
    mission=json.loads((root/'source_mission.json').read_text());robot=mission['robots'][0]
    source=Path(robot['config']).parent/'backend/graph.jsonl'
    output=root/'frozen';output.mkdir(exist_ok=False)
    config=json.loads(config_path.read_text());commands=[]
    for name in ['raw','loops','gps_huber','gps_switchable','loops_gps_huber','loops_gps_switchable']:
        effective=json.loads(json.dumps(config));effective['mode']='pose_graph';effective['gps']['enabled']='gps' in name
        effective['gps']['robust_mode']='huber' if 'huber' in name else 'switchable'
        path=output/f'{name}_config.json';path.write_text(json.dumps(effective,indent=2)+'\n')
        command=[str(binary),str(source),str(root/'gps_records.jsonl'),str(path),str(robot['offset_ns']),str('loops' in name).lower(),str(output/name)]
        completed=subprocess.run(command,text=True,capture_output=True)
        (output/f'{name}.log').write_text(completed.stdout+completed.stderr)
        completed.check_returncode()
        commands.append(command);print(name,completed.stdout.strip(),flush=True)
    (output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('result',type=Path);p.add_argument('--config',type=Path)
    p.add_argument('--binary',type=Path,default=Path('target/release/examples/gps_backend_replay'));a=p.parse_args()
    run(a.result.resolve(),(a.config or a.result/'gps_backend_config.json').resolve(),a.binary.resolve())
