//! Counts typed service field payloads, without claiming DDS/CDR overhead.
use std::sync::atomic::{AtomicU64, Ordering};
use visloc_msgs::{msg as m, srv as s};
pub trait Payload {
    fn bytes(&self) -> usize;
}
macro_rules! payload { ($t:ty, $v:ident, $e:expr) => { impl Payload for $t { fn bytes(&self)->usize { let $v=self; $e } } }; }
fn keys(v: &[m::Key]) -> usize {
    v.iter().map(Payload::bytes).sum()
}
payload!(m::Key, v, v.robot.len() + v.session.len() + 8);
payload!(
    m::Keyframe,
    v,
    v.key.bytes() + 8 + 56 + 1 + v.previous.bytes()
);
payload!(
    m::SequenceAnnouncement,
    v,
    v.key.bytes()
        + keys(&v.members)
        + keys(&v.selected)
        + 16
        + v.model_id.len()
        + 2048
        + v.excluded_keyframes.len() * 8
);
payload!(
    m::LoopConstraint,
    v,
    v.sequence_from.bytes()
        + v.sequence_to.bytes()
        + v.from_key.bytes()
        + v.to_key.bytes()
        + 56
        + 288
        + 4
        + 12
        + 16
        + 1
        + v.verification.reason.len()
);
payload!(s::GetHistory_Request, v, {
    let _ = v;
    26
});
payload!(
    s::GetHistory_Response,
    v,
    v.session.len()
        + v.keyframes.iter().map(Payload::bytes).sum::<usize>()
        + v.sequences.iter().map(Payload::bytes).sum::<usize>()
        + v.loops.iter().map(Payload::bytes).sum::<usize>()
        + 25
);
payload!(s::GetSequence_Request, v, v.key.bytes());
payload!(
    s::GetSequence_Response,
    v,
    1 + v.descriptors.sequence.bytes() + keys(&v.descriptors.frames) + 10240
);
payload!(s::GetFeatures_Request, v, v.key.bytes());
payload!(
    s::GetFeatures_Response,
    v,
    1 + v.frame.key.bytes() + 8 + 96 + v.frame.features.len() * (8 + 8 + 256 + 1 + 24)
);
static CALLS: AtomicU64 = AtomicU64::new(0);
static SENT: AtomicU64 = AtomicU64::new(0);
static RECEIVED: AtomicU64 = AtomicU64::new(0);
static RESPONSES: AtomicU64 = AtomicU64::new(0);
pub enum Queue {
    Sensor,
    Communication,
    Graph,
}
static QUEUE_DROPS: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
pub fn dropped(queue: Queue) {
    QUEUE_DROPS[queue as usize].fetch_add(1, Ordering::Relaxed);
}
pub fn sent(v: &impl Payload) {
    CALLS.fetch_add(1, Ordering::Relaxed);
    SENT.fetch_add(v.bytes() as u64, Ordering::Relaxed);
}
pub fn received(v: &impl Payload) {
    RESPONSES.fetch_add(1, Ordering::Relaxed);
    RECEIVED.fetch_add(v.bytes() as u64, Ordering::Relaxed);
}
pub fn snapshot() -> serde_json::Value {
    serde_json::json!({"service_attempts":CALLS.load(Ordering::Relaxed),"service_responses":RESPONSES.load(Ordering::Relaxed),"service_request_field_bytes":SENT.load(Ordering::Relaxed),"service_response_field_bytes":RECEIVED.load(Ordering::Relaxed),"sensor_ingress_drops":QUEUE_DROPS[0].load(Ordering::Relaxed),"communication_queue_drops":QUEUE_DROPS[1].load(Ordering::Relaxed),"graph_queue_drops":QUEUE_DROPS[2].load(Ordering::Relaxed),"note":"Typed field payload bytes; excludes CDR lengths, padding, encapsulation, DDS headers and retransmission. Counted at requesting node."})
}
