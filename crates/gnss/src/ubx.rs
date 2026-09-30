//! Bounded streaming UBX parser. Payloads may cross arbitrary ROS messages.
use crate::{Diagnostics, Epoch, Observation, Satellite, System};
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subframe {
    pub satellite: Satellite,
    pub signal: u8,
    pub words: Vec<u32>,
    pub receipt_ns: i64,
}
#[derive(Debug, Clone)]
pub enum Message {
    Epoch(Epoch),
    Subframe(Subframe),
}
#[derive(Debug, Default)]
pub struct Decoder {
    bytes: Vec<u8>,
    pub diagnostics: Diagnostics,
}
fn u16le(p: &[u8]) -> u16 {
    u16::from_le_bytes(p[..2].try_into().unwrap())
}
fn f64le(p: &[u8]) -> f64 {
    f64::from_le_bytes(p[..8].try_into().unwrap())
}
fn f32le(p: &[u8]) -> f64 {
    f32::from_le_bytes(p[..4].try_into().unwrap()) as f64
}
impl Decoder {
    pub fn push(&mut self, bytes: &[u8], receipt_ns: i64) -> Vec<Message> {
        let mut out = Vec::new();
        // Feed in bounded chunks: an arbitrarily large publisher cannot grow storage.
        for chunk in bytes.chunks(1024) {
            self.bytes.extend_from_slice(chunk);
            loop {
                let Some(start) = self.bytes.windows(2).position(|w| w == [0xb5, 0x62]) else {
                    let last = self.bytes.last().copied();
                    self.bytes.clear();
                    if last == Some(0xb5) {
                        self.bytes.push(0xb5);
                    }
                    break;
                };
                self.bytes.drain(..start);
                if self.bytes.len() < 6 {
                    break;
                }
                let n = u16le(&self.bytes[4..]) as usize;
                if n > 4096 {
                    self.diagnostics.malformed_packets += 1;
                    self.bytes.remove(0);
                    continue;
                }
                if self.bytes.len() < n + 8 {
                    break;
                }
                let (mut a, mut b) = (0u8, 0u8);
                for v in &self.bytes[2..n + 6] {
                    a = a.wrapping_add(*v);
                    b = b.wrapping_add(a);
                }
                if [a, b] != self.bytes[n + 6..n + 8] {
                    self.diagnostics.checksum_errors += 1;
                    self.bytes.remove(0);
                    continue;
                }
                let payload = self.bytes[6..n + 6].to_vec();
                let result = match (self.bytes[2], self.bytes[3]) {
                    (2, 0x15) => self.rawx(&payload, receipt_ns),
                    (2, 0x13) => self.sfrbx(&payload, receipt_ns),
                    _ => Ok(None),
                };
                match result {
                    Ok(Some(m)) => out.push(m),
                    Err(()) => self.diagnostics.malformed_packets += 1,
                    _ => (),
                }
                self.bytes.drain(..n + 8);
            }
        }
        out
    }
    fn rawx(&mut self, p: &[u8], receipt_ns: i64) -> Result<Option<Message>, ()> {
        if p.len() < 16 || p[13] > 1 || p.len() != 16 + p[11] as usize * 32 {
            return Err(());
        }
        let tow = f64le(p);
        let week = u16le(&p[8..]);
        if !tow.is_finite() || !(0. ..crate::WEEK).contains(&tow) || week == 0 {
            return Err(());
        }
        let mut obs = std::collections::BTreeMap::<Satellite, Observation>::new();
        for m in p[16..].chunks_exact(32) {
            let system = match (m[20], m[22]) {
                (0, 0) => System::Gps,
                (2, 0 | 1) => System::Galileo,
                _ => {
                    self.diagnostics.unsupported_signals += 1;
                    *self
                        .diagnostics
                        .unsupported_signal_counts
                        .entry(format!("constellation_{}_signal_{}", m[20], m[22]))
                        .or_default() += 1;
                    continue;
                }
            };
            let pr = f64le(m);
            let dop = f32le(&m[16..]);
            let reason = if m[30] & 1 == 0 {
                Some("invalid_tracking_status")
            } else if !(1e6..1e8).contains(&pr) {
                Some("invalid_pseudorange")
            } else if !dop.is_finite() {
                Some("invalid_doppler")
            } else {
                None
            };
            if let Some(reason) = reason {
                *self
                    .diagnostics
                    .invalid_observation_counts
                    .entry(reason.into())
                    .or_default() += 1;
                continue;
            }
            let satellite = Satellite { system, prn: m[21] };
            if satellite.prn == 0 || satellite.prn > 64 {
                *self
                    .diagnostics
                    .invalid_observation_counts
                    .entry("invalid_satellite_id".into())
                    .or_default() += 1;
                continue;
            }
            let o = Observation {
                satellite,
                signal: m[22],
                pseudorange_m: pr,
                doppler_hz: dop,
                pseudorange_std_m: 0.01 * 2f64.powi((m[27] & 15) as i32),
                doppler_std_hz: 0.002 * 2f64.powi((m[29] & 15) as i32),
                cn0_dbhz: m[26],
                lock_ms: u16le(&m[24..]),
            };
            if obs
                .get(&satellite)
                .is_none_or(|old| o.cn0_dbhz > old.cn0_dbhz)
            {
                obs.insert(satellite, o);
            }
        }
        self.diagnostics.epochs += 1;
        Ok(Some(Message::Epoch(Epoch {
            week,
            tow_s: tow,
            leap_seconds: p[10] as i8,
            leap_seconds_valid: p[12] & 1 != 0,
            clock_reset: p[12] & 2 != 0,
            receipt_ns,
            observations: obs.into_values().collect(),
        })))
    }
    fn sfrbx(&mut self, p: &[u8], receipt_ns: i64) -> Result<Option<Message>, ()> {
        if p.len() < 8 || p.len() != 8 + 4 * p[4] as usize || p[6] != 2 {
            return Err(());
        }
        let system = match p[0] {
            0 => System::Gps,
            2 => System::Galileo,
            _ => return Ok(None),
        };
        if (system == System::Gps && p[2] != 0) || (system == System::Galileo && p[2] > 1) {
            return Ok(None);
        }
        Ok(Some(Message::Subframe(Subframe {
            satellite: Satellite { system, prn: p[1] },
            signal: p[2],
            words: p[8..]
                .chunks_exact(4)
                .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
                .collect(),
            receipt_ns,
        })))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn packet(p: &[u8]) -> Vec<u8> {
        let mut v = vec![0xb5, 0x62, 2, 0x15];
        v.extend((p.len() as u16).to_le_bytes());
        v.extend(p);
        let (mut a, mut b) = (0u8, 0u8);
        for x in &v[2..] {
            a = a.wrapping_add(*x);
            b = b.wrapping_add(a);
        }
        v.extend([a, b]);
        v
    }
    #[test]
    fn fragmented_corrupt_resynchronization_and_clock_reset() {
        let mut p = vec![0; 48];
        p[..8].copy_from_slice(&123.4f64.to_le_bytes());
        p[8..10].copy_from_slice(&2400u16.to_le_bytes());
        p[10] = 18;
        p[11] = 1;
        p[12] = 3;
        p[13] = 1;
        p[16..24].copy_from_slice(&22e6f64.to_le_bytes());
        p[32..36].copy_from_slice(&(-42f32).to_le_bytes());
        p[37] = 3;
        p[42] = 40;
        p[46] = 1;
        let good = packet(&p);
        let mut bad = good.clone();
        bad[9] ^= 1;
        let mut d = Decoder::default();
        assert!(d.push(&bad, 1).is_empty());
        let mut results = Vec::new();
        for c in good.chunks(3) {
            results.extend(d.push(c, 2));
        }
        assert_eq!(d.diagnostics.checksum_errors, 1);
        assert_eq!(results.len(), 1);
        let Message::Epoch(e) = &results[0] else {
            panic!()
        };
        assert!(e.clock_reset);
        assert_eq!(e.observations[0].satellite.prn, 3);
        assert_eq!(
            e.nominal_utc_ns(),
            Some((GPS_EPOCH_TEST + 2400 * 604800 - 18) * 1_000_000_000 + 123_400_000_000)
        );
    }
    const GPS_EPOCH_TEST: i64 = 315964800;
    #[test]
    fn rawx_time_is_continuous_across_week_rollover_and_preserves_fraction() {
        let mut e = Epoch {
            week: 2400,
            tow_s: 604799.875,
            leap_seconds: 18,
            leap_seconds_valid: true,
            clock_reset: false,
            receipt_ns: 0,
            observations: Vec::new(),
        };
        let before = e.nominal_utc_ns().unwrap();
        e.week += 1;
        e.tow_s = 0.125;
        assert_eq!(e.nominal_utc_ns().unwrap() - before, 250_000_000);
        e.leap_seconds_valid = false;
        assert_eq!(e.nominal_utc_ns(), None);
        e.leap_seconds_valid = true;
        e.tow_s = f64::NAN;
        assert_eq!(e.nominal_utc_ns(), None);
    }
    #[test]
    fn garbage_storage_bounded() {
        let mut d = Decoder::default();
        d.push(&vec![5; 1_000_000], 0);
        assert!(d.bytes.len() < 4096);
    }
}
