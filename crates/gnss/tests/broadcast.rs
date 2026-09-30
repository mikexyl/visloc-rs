use visloc_gnss::{navigation::Navigation, ubx::Subframe, Satellite, System};
#[test]
fn recorded_gps_and_nine_word_galileo_match_independent_broadcast_rinex() {
    let frames: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/south_ece_sfrbx.json")).unwrap();
    let mut nav = Navigation::default();
    for f in frames.as_array().unwrap() {
        let sat = Satellite {
            system: if f["gnss_id"] == 0 {
                System::Gps
            } else {
                System::Galileo
            },
            prn: f["sv_id"].as_u64().unwrap() as u8,
        };
        nav.push(
            &Subframe {
                satellite: sat,
                signal: f["signal"].as_u64().unwrap() as u8,
                words: f["words"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_u64().unwrap() as u32)
                    .collect(),
                receipt_ns: 0,
            },
            2438,
        );
    }
    assert_eq!(nav.rejected_pages, 0);
    assert_eq!(nav.ephemerides.len(), 2);
    let lines: Vec<_> = include_str!("fixtures/south_ece_broadcast_reference.nav")
        .lines()
        .collect();
    for record in lines.chunks_exact(8) {
        let sat = Satellite {
            system: if record[0].starts_with('G') {
                System::Gps
            } else {
                System::Galileo
            },
            prn: record[0][1..3].parse().unwrap(),
        };
        let e = &nav.ephemerides[&sat][0];
        let mut values = Vec::new();
        for line in &record[1..] {
            for i in 0..4 {
                let start = 4 + i * 19;
                if let Some(s) = line.get(start..start + 19) {
                    values.push(s.replace('D', "E").trim().parse::<f64>().unwrap());
                }
            }
        }
        let actual = [
            e.issue as f64,
            e.crs,
            e.delta_n,
            e.m0,
            e.cuc,
            e.eccentricity,
            e.cus,
            e.sqrt_a,
            e.toe,
            e.cic,
            e.omega0,
            e.cis,
            e.inclination,
            e.crc,
            e.argument,
            e.omega_dot,
            e.inclination_dot,
        ];
        for (i, (&a, &b)) in actual.iter().zip(&values).enumerate() {
            assert!(
                (a - b).abs() < b.abs().max(1.) * 1e-11,
                "{sat:?} field {i}: {a} vs {b}"
            );
        }
        for (i, a) in [e.af0, e.af1, e.af2].into_iter().enumerate() {
            let b = record[0][23 + i * 19..42 + i * 19]
                .replace('D', "E")
                .trim()
                .parse::<f64>()
                .unwrap();
            assert!((a - b).abs() < 1e-12);
        }
        assert_eq!(e.week, values[18] as u16);
        let state = e
            .state(e.week as f64 * visloc_gnss::WEEK + e.toe + 100.)
            .unwrap();
        assert!((2e7..3.2e7).contains(&state.position_m.norm()));
        assert!((1000. ..5000.).contains(&state.velocity_m_s.norm()));
    }
}
#[test]
fn galileo_crc_corruption_does_not_create_ephemeris() {
    let frames: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/south_ece_sfrbx.json")).unwrap();
    let mut nav = Navigation::default();
    for f in frames
        .as_array()
        .unwrap()
        .iter()
        .filter(|f| f["gnss_id"] == 2)
    {
        let mut words: Vec<_> = f["words"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect();
        words[1] ^= 4;
        nav.push(
            &Subframe {
                satellite: Satellite {
                    system: System::Galileo,
                    prn: 25,
                },
                signal: 1,
                words,
                receipt_ns: 0,
            },
            2438,
        );
    }
    assert!(nav.ephemerides.is_empty());
    assert!(nav.rejected_pages > 0);
}
