#!/usr/bin/env python3
"""Resolve generated Rust messages from sourced ROS prefixes, without mutation
of the ROS installation. Older colcon-ros-cargo only searches rust_packages;
recent rosidl generators instead install Cargo package metadata.
"""
import os
from pathlib import Path
import json
import re

repo = Path(__file__).resolve().parents[1]
packages = {}
for prefix in reversed(os.environ.get('AMENT_PREFIX_PATH', '').split(':')):
    if not prefix:
        continue
    for manifest in Path(prefix).glob('share/*/rust/Cargo.toml'):
        packages[manifest.parent.parent.name] = manifest.parent.resolve()
required = {'visloc_msgs', 'sensor_msgs', 'nav_msgs', 'geometry_msgs', 'std_msgs', 'tf2_msgs', 'builtin_interfaces'}
missing = required - packages.keys()
if missing:
    raise SystemExit(f'Missing generated Rust message packages: {sorted(missing)}; source the ROS and message overlay first')
# Only patch the generated dependency closure, not unrelated installed ROS
# packages or the installed copy of visloc_ros itself.
selected = {}
pending = list(required)
while pending:
    name = pending.pop()
    if name in selected:
        continue
    if name not in packages:
        raise SystemExit(f'Missing generated Rust dependency: {name}')
    selected[name] = packages[name]
    manifest = (packages[name] / 'Cargo.toml').read_text()
    if '# ROS Dependencies' in manifest:
        section = manifest.split('# ROS Dependencies', 1)[1].split('[', 1)[0]
        pending.extend(re.findall(r'^([a-z][a-z0-9_]*)\s*=', section, re.MULTILINE))
target = repo / 'ros2/visloc_ros/.cargo/config.toml'
target.parent.mkdir(exist_ok=True)
target.write_text('[patch.crates-io]\n' + ''.join(
    f'{name} = {{ path = {json.dumps(str(path))} }}\n' for name, path in sorted(selected.items())))
print(f'Resolved {len(selected)} generated Rust message packages')
