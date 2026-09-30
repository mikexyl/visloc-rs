//! User-supplied nominal mechanical datums. No online extrinsic calibration.
use nalgebra::{Matrix3, UnitQuaternion, Vector3};
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MechanicalDatums {
    pub camera_cad_from_gps_rotation: [[f64; 3]; 3],
    pub camera_cad_from_gps_translation_m: [f64; 3],
    pub rear_mount_to_left_ir_camera_cad_m: [f64; 3],
    pub receiver_bottom_to_antenna_receiver_cad_m: [f64; 3],
    pub camera_axes: String,
    pub receiver_axes: String,
    pub nominal: bool,
}
pub fn mechanical_datums(gap_mm: f64) -> MechanicalDatums {
    MechanicalDatums {
        camera_cad_from_gps_rotation: [[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]],
        camera_cad_from_gps_translation_m: [-0.095, 0.087, 0.048],
        rear_mount_to_left_ir_camera_cad_m: [0.0475, -0.02135, 0.],
        receiver_bottom_to_antenna_receiver_cad_m: [0., 0.01045, 0.0525 + gap_mm * 1e-3],
        camera_axes: "X right looking at the front; Y rearward; Z upward".into(),
        receiver_axes: "Y USB end toward antenna; Z upward; X completes right-handed frame".into(),
        nominal: true,
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeverArm {
    pub gasket_gap_mm: f64,
    pub optical_to_antenna_m: Vector3<f64>,
    pub imu_to_antenna_m: Vector3<f64>,
    pub std_m: f64,
    pub provenance: String,
}
pub fn lever_arm(
    gap_mm: f64,
    std_m: f64,
    r_imu_cam: &UnitQuaternion<f64>,
    t_imu_cam: Vector3<f64>,
) -> LeverArm {
    let optical = Vector3::new(0.1425, -(100.5 + gap_mm) * 1e-3, -0.0979);
    LeverArm { gasket_gap_mm:gap_mm, optical_to_antenna_m:optical,
        imu_to_antenna_m:r_imu_cam*optical+t_imu_cam, std_m,
        provenance:"Nominal D455f rear-mount / CH7604A casing geometry; measured T_imu_cam; fixed extrinsics".into() }
}
pub fn camera_cad_from_gps() -> (Matrix3<f64>, Vector3<f64>) {
    (
        Matrix3::from_diagonal(&Vector3::new(-1., -1., 1.)),
        Vector3::new(-0.095, 0.087, 0.048),
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Quaternion;
    #[test]
    fn supplied_datums_and_measured_calibration() {
        let r = UnitQuaternion::new_normalize(Quaternion::new(
            0.9999977654675232,
            -0.001375972032525672,
            0.0005955286880858952,
            -0.001490337716126886,
        ));
        let t = Vector3::new(
            -0.030887849037458916,
            0.006890377543619629,
            0.01904559767519483,
        );
        let arm = lever_arm(0.6, 0.02, &r, t);
        assert!(
            (arm.imu_to_antenna_m
                - Vector3::new(
                    0.11119323117036664,
                    -0.09490301030319159,
                    -0.07874470264976632
                ))
            .norm()
                < 1e-12
        );
        let (r, t) = camera_cad_from_gps();
        let p = r * Vector3::new(0., 0.01045, 0.0531) + t;
        let b = Matrix3::new(-1., 0., 0., 0., 0., -1., 0., -1., 0.);
        assert!(
            (b.transpose() * (p - Vector3::new(0.0475, -0.02135, 0.)) - arm.optical_to_antenna_m)
                .norm()
                < 1e-12
        );
        assert!(
            (lever_arm(1.6, 0.02, &UnitQuaternion::identity(), Vector3::zeros())
                .optical_to_antenna_m
                .y
                + 0.1021)
                .abs()
                < 1e-12
        );
    }
}
