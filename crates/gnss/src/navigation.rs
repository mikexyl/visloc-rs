//! Broadcast orbit equations and GPS LNAV / Galileo I/NAV bit layouts.
//! Bit-field formulas checked against RTKLIB (T. Takasu / T. Suzuki);
//! see THIRD_PARTY.md. No C/C++ code is linked.
use crate::ubx::Subframe;
use crate::{Satellite, System, C, OMEGA_E, WEEK};
use nalgebra::{Matrix3, Vector3};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::f64::consts::PI;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ephemeris {
    pub satellite: Satellite,
    pub week: u16,
    pub toe: f64,
    pub toc: f64,
    pub issue: u16,
    pub healthy: bool,
    pub sqrt_a: f64,
    pub eccentricity: f64,
    pub m0: f64,
    pub delta_n: f64,
    pub omega0: f64,
    pub inclination: f64,
    pub argument: f64,
    pub omega_dot: f64,
    pub inclination_dot: f64,
    pub cuc: f64,
    pub cus: f64,
    pub crc: f64,
    pub crs: f64,
    pub cic: f64,
    pub cis: f64,
    pub af0: f64,
    pub af1: f64,
    pub af2: f64,
    pub group_delay_s: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SatelliteState {
    pub position_m: Vector3<f64>,
    pub velocity_m_s: Vector3<f64>,
    pub clock_m: f64,
    pub clock_drift_m_s: f64,
}
fn half_week(t: f64) -> f64 {
    (t + WEEK / 2.).rem_euclid(WEEK) - WEEK / 2.
}
impl Ephemeris {
    fn point(&self, t: f64) -> (Vector3<f64>, f64) {
        let tk = half_week(t - self.week as f64 * WEEK - self.toe);
        let a = self.sqrt_a * self.sqrt_a;
        let mu = if self.satellite.system == System::Gps {
            3.986005e14
        } else {
            3.986004418e14
        };
        let n = (mu / (a * a * a)).sqrt() + self.delta_n;
        let m = self.m0 + n * tk;
        let mut e = m;
        for _ in 0..20 {
            let d = (e - self.eccentricity * e.sin() - m) / (1. - self.eccentricity * e.cos());
            e -= d;
            if d.abs() < 1e-13 {
                break;
            }
        }
        let phi = ((1. - self.eccentricity.powi(2)).sqrt() * e.sin())
            .atan2(e.cos() - self.eccentricity)
            + self.argument;
        let (s, c) = (2. * phi).sin_cos();
        let u = phi + self.cus * s + self.cuc * c;
        let radius = a * (1. - self.eccentricity * e.cos()) + self.crs * s + self.crc * c;
        let i = self.inclination + self.inclination_dot * tk + self.cis * s + self.cic * c;
        let omega = self.omega0 + (self.omega_dot - OMEGA_E) * tk - OMEGA_E * self.toe;
        let (x, y) = (radius * u.cos(), radius * u.sin());
        let (so, co) = omega.sin_cos();
        let pos = Vector3::new(
            x * co - y * i.cos() * so,
            x * so + y * i.cos() * co,
            y * i.sin(),
        );
        let dt = half_week(t - self.week as f64 * WEEK - self.toc);
        let clock = self.af0 + self.af1 * dt + self.af2 * dt * dt
            - 2. * (mu * a).sqrt() * self.eccentricity * e.sin() / (C * C)
            - self.group_delay_s;
        (pos, clock * C)
    }
    pub fn state(&self, t: f64) -> Option<SatelliteState> {
        if !self.healthy
            || !(4000. ..6000.).contains(&self.sqrt_a)
            || !(0. ..1.).contains(&self.eccentricity)
            || !t.is_finite()
            || (t - self.week as f64 * WEEK - self.toe).abs() > 7200.
        {
            return None;
        }
        let (p, b) = self.point(t);
        let (pp, bp) = self.point(t + 0.01);
        let (pm, bm) = self.point(t - 0.01);
        Some(SatelliteState {
            position_m: p,
            velocity_m_s: (pp - pm) / 0.02,
            clock_m: b,
            clock_drift_m_s: (bp - bm) / 0.02,
        })
    }
    pub fn at_transmission(&self, receive_s: f64, pseudorange_m: f64) -> Option<SatelliteState> {
        let tx = receive_s - pseudorange_m / C;
        let clock = self.state(tx)?.clock_m / C;
        let mut state = self.state(tx - clock)?;
        let a = OMEGA_E * pseudorange_m / C;
        let (s, c) = a.sin_cos();
        let rotation = Matrix3::new(c, s, 0., -s, c, 0., 0., 0., 1.);
        state.position_m = rotation * state.position_m;
        state.velocity_m_s = rotation * state.velocity_m_s;
        Some(state)
    }
}
pub fn ecef_to_geodetic(p: Vector3<f64>) -> Vector3<f64> {
    let a = 6378137.;
    let e2 = 6.69437999014e-3;
    let xy = p.x.hypot(p.y);
    let mut lat = p.z.atan2(xy * (1. - e2));
    let mut h = 0.;
    for _ in 0..10 {
        let n = a / (1. - e2 * lat.sin().powi(2)).sqrt();
        h = xy / lat.cos() - n;
        lat = p.z.atan2(xy * (1. - e2 * n / (n + h)));
    }
    Vector3::new(lat, p.y.atan2(p.x), h)
}
pub fn enu_to_ecef(origin: Vector3<f64>) -> Matrix3<f64> {
    let p = ecef_to_geodetic(origin);
    let (sl, cl) = p.x.sin_cos();
    let (so, co) = p.y.sin_cos();
    Matrix3::new(-so, -sl * co, cl * co, co, -sl * so, cl * so, 0., cl, sl)
}
pub fn azimuth_elevation(receiver: Vector3<f64>, satellite: Vector3<f64>) -> (f64, f64) {
    let v = enu_to_ecef(receiver).transpose() * (satellite - receiver).normalize();
    (
        v.x.atan2(v.y).rem_euclid(2. * PI),
        v.z.clamp(-1., 1.).asin(),
    )
}
/// Saastamoinen standard atmosphere + Klobuchar L1 broadcast correction.
/// Galileo E1 shares L1 frequency; GPS broadcast ionosphere is used as a
/// conservative common model, not represented as Galileo NeQuick.
pub fn atmospheric_delay(
    receiver: Vector3<f64>,
    satellite: Vector3<f64>,
    tow: f64,
    iono: Option<[f64; 8]>,
) -> f64 {
    let p = ecef_to_geodetic(receiver);
    let (az, el) = azimuth_elevation(receiver, satellite);
    if el <= 0. {
        return 0.;
    }
    let h = p.z.clamp(-100., 10000.);
    let temp = 15. - 0.0065 * h + 273.16;
    let pressure = 1013.25 * (1. - 2.2557e-5 * h).powf(5.2568);
    let vapour = 6.108 * 0.7 * ((17.15 * temp - 4684.) / (temp - 38.45)).exp();
    let trop = 0.002277 * (pressure + (1255. / temp + 0.05) * vapour)
        / (el.sin() * (1. - 0.00266 * (2. * p.x).cos() - 0.00028 * h / 1000.));
    let a = iono.unwrap_or([
        0.1118e-7, -0.7451e-8, -0.5961e-7, 0.1192e-6, 0.1167e6, -0.2294e6, -0.1311e6, 0.1049e7,
    ]);
    let psi = 0.0137 / (el / PI + 0.11) - 0.022;
    let phi = (p.x / PI + psi * az.cos()).clamp(-0.416, 0.416);
    let lam = p.y / PI + psi * az.sin() / (phi * PI).cos();
    let phi_m = phi + 0.064 * ((lam - 1.617) * PI).cos();
    let local = (43200. * lam + tow).rem_euclid(86400.);
    let amp = (a[0] + phi_m * (a[1] + phi_m * (a[2] + phi_m * a[3]))).max(0.);
    let per = (a[4] + phi_m * (a[5] + phi_m * (a[6] + phi_m * a[7]))).max(72000.);
    let x = 2. * PI * (local - 50400.) / per;
    let f = 1. + 16. * (0.53 - el / PI).powi(3);
    trop + C
        * f
        * (5e-9
            + if x.abs() < 1.57 {
                amp * (1. - x * x / 2. + x.powi(4) / 24.)
            } else {
                0.
            })
}
pub(crate) fn bits(p: &[u8], pos: usize, n: usize) -> u32 {
    (0..n).fold(0, |v, i| {
        (v << 1) | ((p[(pos + i) / 8] >> (7 - (pos + i) % 8)) & 1) as u32
    })
}
fn signed(p: &[u8], pos: usize, n: usize) -> i32 {
    let x = bits(p, pos, n);
    ((x << (32 - n)) as i32) >> (32 - n)
}
fn u(p: &[u8], i: usize, n: usize, pow: i32) -> f64 {
    bits(p, i, n) as f64 * 2f64.powi(pow)
}
fn s(p: &[u8], i: usize, n: usize, pow: i32) -> f64 {
    signed(p, i, n) as f64 * 2f64.powi(pow)
}
fn crc24q(bytes: &[u8]) -> u32 {
    let mut crc = 0u32;
    for b in bytes {
        crc ^= (*b as u32) << 16;
        for _ in 0..8 {
            crc <<= 1;
            if crc & 0x1000000 != 0 {
                crc ^= 0x1864cfb;
            }
        }
    }
    crc & 0xffffff
}
#[derive(Debug, Default)]
pub struct Navigation {
    gps: BTreeMap<u8, [Option<Vec<u8>>; 3]>,
    gal: BTreeMap<u8, [Option<Vec<u8>>; 6]>,
    pub ephemerides: BTreeMap<Satellite, Vec<Ephemeris>>,
    pub ionosphere: Option<[f64; 8]>,
    pub rejected_pages: u64,
}
impl Navigation {
    pub fn insert(&mut self, e: Ephemeris) -> bool {
        let v = self.ephemerides.entry(e.satellite).or_default();
        if !v
            .iter()
            .any(|x| x.week == e.week && x.toe == e.toe && x.issue == e.issue)
        {
            v.push(e);
            v.sort_by(|a, b| a.week.cmp(&b.week).then(a.toe.total_cmp(&b.toe)));
            if v.len() > 8 {
                v.remove(0);
            }
            true
        } else {
            false
        }
    }
    pub fn select(&self, sat: Satellite, t: f64) -> Option<&Ephemeris> {
        self.ephemerides
            .get(&sat)?
            .iter()
            .filter(|e| e.healthy && (t - e.week as f64 * WEEK - e.toe).abs() < 7200.)
            .min_by(|a, b| {
                (t - a.week as f64 * WEEK - a.toe)
                    .abs()
                    .total_cmp(&(t - b.week as f64 * WEEK - b.toe).abs())
            })
    }
    pub fn push(&mut self, f: &Subframe, current_week: u16) -> Option<Ephemeris> {
        let result = match f.satellite.system {
            System::Gps => self.gps_page(f, current_week),
            System::Galileo => self.gal_page(f),
        };
        result.filter(|e| self.insert(e.clone()))
    }
    fn gps_page(&mut self, f: &Subframe, week: u16) -> Option<Ephemeris> {
        if f.words.len() != 10 {
            return None;
        }
        let p: Vec<u8> = f
            .words
            .iter()
            .flat_map(|w| (w >> 6).to_be_bytes()[1..].to_vec())
            .collect();
        if bits(&p, 0, 8) != 0x8b {
            self.rejected_pages += 1;
            return None;
        }
        let id = bits(&p, 43, 3) as usize;
        if id == 4 && bits(&p, 50, 6) == 56 {
            let mut a = [0.; 8];
            for (i, v) in a.iter_mut().enumerate() {
                *v = s(&p, 56 + i * 8, 8, [-30, -27, -24, -24, 11, 14, 16, 16][i]);
            }
            self.ionosphere = Some(a);
        }
        if !(1..=3).contains(&id) {
            return None;
        }
        let pages = self
            .gps
            .entry(f.satellite.prn)
            .or_insert_with(|| std::array::from_fn(|_| None));
        pages[id - 1] = Some(p);
        let (p1, p2, p3) = (pages[0].as_ref()?, pages[1].as_ref()?, pages[2].as_ref()?);
        let issue = bits(p2, 48, 8);
        if bits(p3, 216, 8) != issue || bits(p1, 168, 8) != issue {
            return None;
        }
        let w = bits(p1, 48, 10) as i32;
        let mut full = w + ((week as i32 - w + 512) / 1024) * 1024;
        let toe = u(p2, 216, 16, 4);
        let tow = bits(p1, 24, 17) as f64 * 6.;
        if toe < tow - 302400. {
            full += 1;
        } else if toe > tow + 302400. {
            full -= 1;
        }
        Some(Ephemeris {
            satellite: f.satellite,
            week: full as u16,
            toe,
            toc: u(p1, 176, 16, 4),
            issue: issue as u16,
            healthy: bits(p1, 64, 6) == 0,
            sqrt_a: u(p2, 184, 32, -19),
            eccentricity: u(p2, 136, 32, -33),
            m0: s(p2, 88, 32, -31) * PI,
            delta_n: s(p2, 72, 16, -43) * PI,
            omega0: s(p3, 64, 32, -31) * PI,
            inclination: s(p3, 112, 32, -31) * PI,
            argument: s(p3, 160, 32, -31) * PI,
            omega_dot: s(p3, 192, 24, -43) * PI,
            inclination_dot: s(p3, 224, 14, -43) * PI,
            cuc: s(p2, 120, 16, -29),
            cus: s(p2, 168, 16, -29),
            crc: s(p3, 144, 16, -5),
            crs: s(p2, 56, 16, -5),
            cic: s(p3, 48, 16, -29),
            cis: s(p3, 96, 16, -29),
            af0: s(p1, 216, 22, -31),
            af1: s(p1, 200, 16, -43),
            af2: s(p1, 192, 8, -55),
            group_delay_s: s(p1, 160, 8, -31),
        })
    }
    fn gal_page(&mut self, f: &Subframe) -> Option<Ephemeris> {
        if f.words.len() < 8 {
            return None;
        }
        let p: Vec<u8> = f.words[..8].iter().flat_map(|w| w.to_be_bytes()).collect();
        if bits(&p, 0, 2) != 0 || bits(&p, 128, 2) != 2 {
            return None;
        }
        // Exactly 114 even and 82 odd bits, with four leading zero bits.
        let mut crc_data = vec![0u8; 25];
        for i in 0..196 {
            let v = if i < 114 {
                bits(&p, i, 1)
            } else {
                bits(&p, 128 + i - 114, 1)
            };
            crc_data[(i + 4) / 8] |= (v as u8) << (7 - (i + 4) % 8);
        }
        if crc24q(&crc_data) != bits(&p, 210, 24) {
            self.rejected_pages += 1;
            return None;
        }
        let kind = bits(&p, 2, 6) as usize;
        if kind > 5 {
            return None;
        }
        let mut word = vec![0u8; 16];
        for i in 0..128 {
            let v = if i < 112 {
                bits(&p, 2 + i, 1)
            } else {
                bits(&p, 130 + i - 112, 1)
            };
            word[i / 8] |= (v as u8) << (7 - i % 8);
        }
        let pages = self
            .gal
            .entry(f.satellite.prn)
            .or_insert_with(|| std::array::from_fn(|_| None));
        pages[kind] = Some(word);
        let (p0, p1, p2, p3, p4, p5) = (
            pages[0].as_ref()?,
            pages[1].as_ref()?,
            pages[2].as_ref()?,
            pages[3].as_ref()?,
            pages[4].as_ref()?,
            pages[5].as_ref()?,
        );
        let issue = bits(p1, 6, 10);
        if bits(p0, 6, 2) != 2
            || [p2, p3, p4].iter().any(|p| bits(p, 6, 10) != issue)
            || bits(p4, 16, 6) != f.satellite.prn as u32
        {
            return None;
        }
        let mut week = bits(p0, 96, 12) as i32 + 1024;
        let toe = u(p1, 16, 14, 0) * 60.;
        let tow = bits(p0, 108, 20) as f64;
        if toe - tow > 302400. {
            week -= 1;
        } else if toe - tow < -302400. {
            week += 1;
        }
        Some(Ephemeris {
            satellite: f.satellite,
            week: week as u16,
            toe,
            toc: u(p4, 54, 14, 0) * 60.,
            issue: issue as u16,
            healthy: bits(p5, 69, 2) == 0 && bits(p5, 72, 1) == 0,
            sqrt_a: u(p1, 94, 32, -19),
            eccentricity: u(p1, 62, 32, -33),
            m0: s(p1, 30, 32, -31) * PI,
            delta_n: s(p3, 40, 16, -43) * PI,
            omega0: s(p2, 16, 32, -31) * PI,
            inclination: s(p2, 48, 32, -31) * PI,
            argument: s(p2, 80, 32, -31) * PI,
            omega_dot: s(p3, 16, 24, -43) * PI,
            inclination_dot: s(p2, 112, 14, -43) * PI,
            cuc: s(p3, 56, 16, -29),
            cus: s(p3, 72, 16, -29),
            crc: s(p3, 88, 16, -5),
            crs: s(p3, 104, 16, -5),
            cic: s(p4, 22, 16, -29),
            cis: s(p4, 38, 16, -29),
            af0: s(p4, 68, 31, -34),
            af1: s(p4, 99, 21, -46),
            af2: s(p4, 120, 6, -59),
            group_delay_s: s(p5, 57, 10, -32),
        })
    }
}
