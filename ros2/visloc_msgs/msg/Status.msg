string robot
string session
bool ready
bool finished
uint64 frames
uint64 keyframes
uint64 sequences
uint64 loops
uint64 dropped_images
uint64 dropped_keyframes
uint64 request_failures
int64 timestamp_ns
string error
# Last input accepted by the VIO/startup worker, before an odometry result exists.
int64 input_timestamp_ns
bool initializing
# Accepted inputs and deliberately omitted pre-initialization frames are not drops.
uint64 frames_ingested
uint64 initialization_skipped_frames
