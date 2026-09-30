//! Typed raw GNSS ROS contracts; no opaque JSON transport.
use visloc_gnss::{navigation::Ephemeris, Epoch, Observation, Satellite, System};
use visloc_msgs::msg as m;
fn satellite(system: u8, prn: u8) -> Result<Satellite, String> {
    let system = match system {
        0 => System::Gps,
        2 => System::Galileo,
        _ => return Err("unsupported GNSS system".into()),
    };
    if prn == 0 || prn > 64 {
        return Err("invalid GNSS satellite".into());
    }
    Ok(Satellite { system, prn })
}
fn id(system: System) -> u8 {
    match system {
        System::Gps => 0,
        System::Galileo => 2,
    }
}
pub fn epoch_wire(e: &Epoch) -> m::GnssEpoch {
    m::GnssEpoch {
        week: e.week,
        tow_s: e.tow_s,
        leap_seconds: e.leap_seconds,
        leap_seconds_valid: e.leap_seconds_valid,
        clock_reset: e.clock_reset,
        receipt_ns: e.receipt_ns,
        observations: e
            .observations
            .iter()
            .map(|o| m::GnssObservation {
                gnss_id: id(o.satellite.system),
                sv_id: o.satellite.prn,
                signal_id: o.signal,
                pseudorange_m: o.pseudorange_m,
                doppler_hz: o.doppler_hz,
                pseudorange_std_m: o.pseudorange_std_m,
                doppler_std_hz: o.doppler_std_hz,
                cn0_dbhz: o.cn0_dbhz,
                lock_ms: o.lock_ms,
            })
            .collect(),
    }
}
pub fn epoch_from(e: m::GnssEpoch) -> Result<Epoch, String> {
    if e.week == 0
        || !e.tow_s.is_finite()
        || !(0. ..visloc_gnss::WEEK).contains(&e.tow_s)
        || e.observations.len() > 64
    {
        return Err("invalid GNSS epoch".into());
    }
    let observations = e
        .observations
        .into_iter()
        .map(|o| {
            if !matches!((o.gnss_id, o.signal_id), (0, 0) | (2, 0 | 1)) {
                return Err("unsupported GNSS signal".into());
            }
            if !(1e6..1e8).contains(&o.pseudorange_m)
                || !o.doppler_hz.is_finite()
                || !o.pseudorange_std_m.is_finite()
                || !o.doppler_std_hz.is_finite()
                || o.pseudorange_std_m <= 0.
                || o.doppler_std_hz <= 0.
            {
                return Err("invalid GNSS observation".into());
            }
            Ok(Observation {
                satellite: satellite(o.gnss_id, o.sv_id)?,
                signal: o.signal_id,
                pseudorange_m: o.pseudorange_m,
                doppler_hz: o.doppler_hz,
                pseudorange_std_m: o.pseudorange_std_m,
                doppler_std_hz: o.doppler_std_hz,
                cn0_dbhz: o.cn0_dbhz,
                lock_ms: o.lock_ms,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(Epoch {
        week: e.week,
        tow_s: e.tow_s,
        leap_seconds: e.leap_seconds,
        leap_seconds_valid: e.leap_seconds_valid,
        clock_reset: e.clock_reset,
        receipt_ns: e.receipt_ns,
        observations,
    })
}
pub fn ephemeris_wire(e: &Ephemeris) -> m::GnssEphemeris {
    m::GnssEphemeris {
        gnss_id: id(e.satellite.system),
        sv_id: e.satellite.prn,
        week: e.week,
        toe: e.toe,
        toc: e.toc,
        issue: e.issue,
        healthy: e.healthy,
        sqrt_a: e.sqrt_a,
        eccentricity: e.eccentricity,
        m0: e.m0,
        delta_n: e.delta_n,
        omega0: e.omega0,
        inclination: e.inclination,
        argument: e.argument,
        omega_dot: e.omega_dot,
        inclination_dot: e.inclination_dot,
        cuc: e.cuc,
        cus: e.cus,
        crc: e.crc,
        crs: e.crs,
        cic: e.cic,
        cis: e.cis,
        af0: e.af0,
        af1: e.af1,
        af2: e.af2,
        group_delay_s: e.group_delay_s,
    }
}
pub fn ephemeris_from(e: m::GnssEphemeris) -> Result<Ephemeris, String> {
    let satellite = satellite(e.gnss_id, e.sv_id)?;
    if [
        e.toe,
        e.toc,
        e.sqrt_a,
        e.eccentricity,
        e.m0,
        e.delta_n,
        e.omega0,
        e.inclination,
        e.argument,
        e.omega_dot,
        e.inclination_dot,
        e.cuc,
        e.cus,
        e.crc,
        e.crs,
        e.cic,
        e.cis,
        e.af0,
        e.af1,
        e.af2,
        e.group_delay_s,
    ]
    .iter()
    .any(|v| !v.is_finite())
        || !(4000. ..6000.).contains(&e.sqrt_a)
        || !(0. ..1.).contains(&e.eccentricity)
        || e.week == 0 || !(0. ..visloc_gnss::WEEK).contains(&e.toe)
        || !(0. ..visloc_gnss::WEEK).contains(&e.toc)
    {
        return Err("invalid GNSS ephemeris".into());
    }
    Ok(Ephemeris {
        satellite,
        week: e.week,
        toe: e.toe,
        toc: e.toc,
        issue: e.issue,
        healthy: e.healthy,
        sqrt_a: e.sqrt_a,
        eccentricity: e.eccentricity,
        m0: e.m0,
        delta_n: e.delta_n,
        omega0: e.omega0,
        inclination: e.inclination,
        argument: e.argument,
        omega_dot: e.omega_dot,
        inclination_dot: e.inclination_dot,
        cuc: e.cuc,
        cus: e.cus,
        crc: e.crc,
        crs: e.crs,
        cic: e.cic,
        cis: e.cis,
        af0: e.af0,
        af1: e.af1,
        af2: e.af2,
        group_delay_s: e.group_delay_s,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn epoch_round_trip_and_malformed_input() {
        let e = Epoch {
            week: 2400,
            tow_s: 100.2,
            leap_seconds: 18,
            leap_seconds_valid: true,
            clock_reset: true,
            receipt_ns: 123,
            observations: vec![Observation {
                satellite: Satellite {
                    system: System::Galileo,
                    prn: 25,
                },
                signal: 1,
                pseudorange_m: 2e7,
                doppler_hz: -500.,
                pseudorange_std_m: 3.,
                doppler_std_hz: 0.1,
                cn0_dbhz: 45,
                lock_ms: 1000,
            }],
        };
        let round = epoch_from(epoch_wire(&e)).unwrap();
        assert_eq!(
            serde_json::to_value(e).unwrap(),
            serde_json::to_value(round).unwrap()
        );
        let mut bad = m::GnssEpoch::default();
        bad.week = 2400;
        bad.tow_s = f64::NAN;
        assert!(epoch_from(bad).is_err());
    }
    #[test]
    fn broadcast_round_trip() {
        let e = Ephemeris {
            satellite: Satellite {
                system: System::Gps,
                prn: 7,
            },
            week: 2400,
            toe: 100.,
            toc: 100.,
            issue: 4,
            healthy: true,
            sqrt_a: 5153.6,
            eccentricity: 0.01,
            m0: 0.2,
            delta_n: 1e-9,
            omega0: 0.3,
            inclination: 1.,
            argument: 0.1,
            omega_dot: -1e-9,
            inclination_dot: 0.,
            cuc: 1e-6,
            cus: 2e-6,
            crc: 50.,
            crs: 10.,
            cic: 3e-6,
            cis: 4e-6,
            af0: 1e-4,
            af1: 1e-12,
            af2: 0.,
            group_delay_s: 1e-8,
        };
        assert_eq!(
            serde_json::to_value(&e).unwrap(),
            serde_json::to_value(ephemeris_from(ephemeris_wire(&e)).unwrap()).unwrap()
        );
    }
}
