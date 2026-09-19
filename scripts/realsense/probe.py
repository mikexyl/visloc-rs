import json
import pyrealsense2 as rs
out=[]
for d in rs.context().query_devices():
    record={'name':d.get_info(rs.camera_info.name),'serial':d.get_info(rs.camera_info.serial_number),'firmware':d.get_info(rs.camera_info.firmware_version),'sensors':[]}
    for s in d.query_sensors():
        profiles=[]
        for p in s.get_stream_profiles():
            if p.stream_type() in (rs.stream.gyro,rs.stream.accel) or (p.stream_type()==rs.stream.infrared and p.as_video_stream_profile().width()==640 and p.as_video_stream_profile().height()==480):
                profiles.append(str(p))
        record['sensors'].append({'name':s.get_info(rs.camera_info.name),'profiles':profiles})
    out.append(record)
print(json.dumps(out,indent=2))
