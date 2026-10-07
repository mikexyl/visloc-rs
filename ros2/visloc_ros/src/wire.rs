use visloc_msgs::msg as m;
use visloc_multi_robot as c;

pub trait Wire: Sized {
    type Msg;
    fn wire(&self) -> Self::Msg;
    fn from_wire(m: Self::Msg) -> Self;
}
impl Wire for c::Key {
    type Msg = m::Key;
    fn wire(&self) -> m::Key {
        m::Key {
            robot: self.robot.clone(),
            session: self.session.clone(),
            id: self.id,
        }
    }
    fn from_wire(m: m::Key) -> Self {
        Self {
            robot: m.robot,
            session: m.session,
            id: m.id,
        }
    }
}
impl Wire for c::Transform {
    type Msg = m::Transform;
    fn wire(&self) -> m::Transform {
        m::Transform {
            translation: self.translation,
            rotation_xyzw: self.rotation_xyzw,
        }
    }
    fn from_wire(m: m::Transform) -> Self {
        Self {
            translation: m.translation,
            rotation_xyzw: m.rotation_xyzw,
        }
    }
}
impl Wire for c::CameraModel {
    type Msg = m::Camera;
    fn wire(&self) -> m::Camera {
        m::Camera {
            width: self.width,
            height: self.height,
            intrinsics: self.intrinsics,
            camera_to_body: self.camera_to_body.wire(),
        }
    }
    fn from_wire(m: m::Camera) -> Self {
        Self {
            width: m.width,
            height: m.height,
            intrinsics: m.intrinsics,
            camera_to_body: c::Transform::from_wire(m.camera_to_body),
        }
    }
}
impl Wire for c::KeyframeRecord {
    type Msg = m::Keyframe;
    fn wire(&self) -> m::Keyframe {
        m::Keyframe {
            key: self.key.wire(),
            timestamp_ns: self.timestamp_ns,
            body_to_odom: self.body_to_odom.wire(),
            has_previous: self.previous.is_some(),
            previous: self.previous.as_ref().map(Wire::wire).unwrap_or_default(),
        }
    }
    fn from_wire(m: m::Keyframe) -> Self {
        Self {
            key: c::Key::from_wire(m.key),
            timestamp_ns: m.timestamp_ns,
            body_to_odom: c::Transform::from_wire(m.body_to_odom),
            previous: m.has_previous.then(|| c::Key::from_wire(m.previous)),
        }
    }
}
impl Wire for c::Sequence {
    type Msg = m::SequenceAnnouncement;
    fn wire(&self) -> m::SequenceAnnouncement {
        m::SequenceAnnouncement {
            key: self.key.wire(),
            members: std::array::from_fn(|i| self.members[i].wire()),
            selected: std::array::from_fn(|i| self.selected[i].wire()),
            start_ns: self.start_ns,
            end_ns: self.end_ns,
            model_id: self.model_id.clone(),
            descriptor: self.descriptor.clone().try_into().unwrap(),
            excluded_keyframes: self.excluded_keyframes.clone(),
        }
    }
    fn from_wire(m: m::SequenceAnnouncement) -> Self {
        Self {
            key: c::Key::from_wire(m.key),
            members: m.members.into_iter().map(c::Key::from_wire).collect(),
            selected: m.selected.into_iter().map(c::Key::from_wire).collect(),
            start_ns: m.start_ns,
            end_ns: m.end_ns,
            model_id: m.model_id,
            descriptor: m.descriptor.to_vec(),
            excluded_keyframes: m.excluded_keyframes,
        }
    }
}
impl Wire for c::FrameDescriptors {
    type Msg = m::FrameDescriptors;
    fn wire(&self) -> m::FrameDescriptors {
        m::FrameDescriptors {
            sequence: self.sequence.wire(),
            frames: std::array::from_fn(|i| self.frames[i].wire()),
            descriptors: self.descriptors.clone().try_into().unwrap(),
        }
    }
    fn from_wire(m: m::FrameDescriptors) -> Self {
        Self {
            sequence: c::Key::from_wire(m.sequence),
            frames: m.frames.into_iter().map(c::Key::from_wire).collect(),
            descriptors: m.descriptors.to_vec(),
        }
    }
}
impl Wire for c::FeatureFrame {
    type Msg = m::FeatureFrame;
    fn wire(&self) -> m::FeatureFrame {
        m::FeatureFrame {
            key: self.key.wire(),
            timestamp_ns: self.timestamp_ns,
            camera: self.camera.wire(),
            features: (0..self.pixels.len())
                .map(|i| m::Feature {
                    track_id: self.track_ids[i],
                    pixel: self.pixels[i],
                    descriptor: self.descriptors[i * 64..(i + 1) * 64].try_into().unwrap(),
                    has_landmark: self.points_camera[i].is_some(),
                    point_camera: self.points_camera[i].unwrap_or_default(),
                })
                .collect(),
        }
    }
    fn from_wire(m: m::FeatureFrame) -> Self {
        Self {
            key: c::Key::from_wire(m.key),
            timestamp_ns: m.timestamp_ns,
            camera: c::CameraModel::from_wire(m.camera),
            pixels: m.features.iter().map(|f| f.pixel).collect(),
            descriptors: m.features.iter().flat_map(|f| f.descriptor).collect(),
            track_ids: m.features.iter().map(|f| f.track_id).collect(),
            points_camera: m
                .features
                .iter()
                .map(|f| f.has_landmark.then_some(f.point_camera))
                .collect(),
        }
    }
}
impl Wire for c::Verification {
    type Msg = m::Verification;
    fn wire(&self) -> m::Verification {
        m::Verification {
            matches: self.matches as u32,
            two_d_inliers: self.two_d_inliers as u32,
            pnp_inliers: self.pnp_inliers as u32,
            reprojection_px: self.reprojection_px,
            coverage: self.coverage,
            reverse: self.reverse,
            reason: self.reason.clone(),
        }
    }
    fn from_wire(m: m::Verification) -> Self {
        Self {
            matches: m.matches as usize,
            two_d_inliers: m.two_d_inliers as usize,
            pnp_inliers: m.pnp_inliers as usize,
            reprojection_px: m.reprojection_px,
            coverage: m.coverage,
            reverse: m.reverse,
            reason: m.reason,
        }
    }
}
impl Wire for c::LoopConstraint {
    type Msg = m::LoopConstraint;
    fn wire(&self) -> m::LoopConstraint {
        m::LoopConstraint {
            sequence_from: self.pair.0.wire(),
            sequence_to: self.pair.1.wire(),
            from_key: self.from.wire(),
            to_key: self.to.wire(),
            to_from: self.to_from.wire(),
            information: self.information.clone().try_into().unwrap(),
            similarity: self.similarity,
            verification: self.verification.wire(),
        }
    }
    fn from_wire(m: m::LoopConstraint) -> Self {
        Self {
            pair: c::Pair(
                c::Key::from_wire(m.sequence_from),
                c::Key::from_wire(m.sequence_to),
            ),
            from: c::Key::from_wire(m.from_key),
            to: c::Key::from_wire(m.to_key),
            to_from: c::Transform::from_wire(m.to_from),
            information: m.information.to_vec(),
            similarity: m.similarity,
            verification: c::Verification::from_wire(m.verification),
        }
    }
}
impl Wire for c::OptimizedPose {
    type Msg = m::OptimizedPose;
    fn wire(&self) -> m::OptimizedPose {
        m::OptimizedPose {
            key: self.key.wire(),
            timestamp_ns: self.timestamp_ns,
            component: self.component.wire(),
            body_to_map: self.body_to_map.wire(),
            map_from_odom: self.map_from_odom.wire(),
        }
    }
    fn from_wire(m: m::OptimizedPose) -> Self {
        Self {
            key: c::Key::from_wire(m.key),
            timestamp_ns: m.timestamp_ns,
            component: c::Key::from_wire(m.component),
            body_to_map: c::Transform::from_wire(m.body_to_map),
            map_from_odom: c::Transform::from_wire(m.map_from_odom),
        }
    }
}
impl Wire for c::GraphSnapshot {
    type Msg = m::GraphSnapshot;
    fn wire(&self) -> m::GraphSnapshot {
        m::GraphSnapshot {
            revision: self.revision,
            input_revision: self.input_revision,
            poses: self.poses.iter().map(Wire::wire).collect(),
            loops: self.loops.iter().map(Wire::wire).collect(),
            components: self.components as u32,
            initial_cost: self.initial_cost,
            final_cost: self.final_cost,
            solve_ms: self.solve_ms,
            backend_mode: match self.backend_mode {
                c::BackendMode::PoseGraph => "pose_graph",
                c::BackendMode::GlobalBundleAdjustment => "global_bundle_adjustment",
            }
            .into(),
            landmarks: self.landmarks.iter().map(Wire::wire).collect(),
        }
    }
    fn from_wire(m: m::GraphSnapshot) -> Self {
        Self {
            revision: m.revision,
            input_revision: m.input_revision,
            poses: m
                .poses
                .into_iter()
                .map(c::OptimizedPose::from_wire)
                .collect(),
            loops: m
                .loops
                .into_iter()
                .map(c::LoopConstraint::from_wire)
                .collect(),
            components: m.components as usize,
            initial_cost: m.initial_cost,
            final_cost: m.final_cost,
            solve_ms: m.solve_ms,
            backend_mode: if m.backend_mode == "global_bundle_adjustment" {
                c::BackendMode::GlobalBundleAdjustment
            } else {
                c::BackendMode::PoseGraph
            },
            landmarks: m
                .landmarks
                .into_iter()
                .map(c::OptimizedLandmark::from_wire)
                .collect(),
            bundle_diagnostics: Default::default(),
            gps: Default::default(),
            // Detailed optimizer diagnostics are persisted in backend journals.
            optimizer_reports: Default::default(),
        }
    }
}
pub fn stamp(ns: i64) -> builtin_interfaces::msg::Time {
    builtin_interfaces::msg::Time {
        sec: ns.div_euclid(1_000_000_000) as i32,
        nanosec: ns.rem_euclid(1_000_000_000) as u32,
    }
}
pub fn pose(t: &c::Transform) -> geometry_msgs::msg::Pose {
    let [x, y, z] = t.translation;
    let [qx, qy, qz, qw] = t.rotation_xyzw;
    geometry_msgs::msg::Pose {
        position: geometry_msgs::msg::Point { x, y, z },
        orientation: geometry_msgs::msg::Quaternion {
            x: qx,
            y: qy,
            z: qz,
            w: qw,
        },
    }
}
pub fn tf(
    t: &c::Transform,
    parent: &str,
    child: &str,
    ns: i64,
) -> geometry_msgs::msg::TransformStamped {
    let [x, y, z] = t.translation;
    let [qx, qy, qz, qw] = t.rotation_xyzw;
    geometry_msgs::msg::TransformStamped {
        header: std_msgs::msg::Header {
            stamp: stamp(ns),
            frame_id: parent.into(),
        },
        child_frame_id: child.into(),
        transform: geometry_msgs::msg::Transform {
            translation: geometry_msgs::msg::Vector3 { x, y, z },
            rotation: geometry_msgs::msg::Quaternion {
                x: qx,
                y: qy,
                z: qz,
                w: qw,
            },
        },
    }
}

impl Wire for c::GpsRecord {
    type Msg = m::GpsFix;
    fn wire(&self) -> Self::Msg {
        m::GpsFix {
            key: self.key.wire(),
            timestamp_ns: self.timestamp_ns,
            receipt_timestamp_ns: self.receipt_timestamp_ns,
            time_source: self.time_source.clone(),
            has_position: self.lla.is_some(),
            lla: self.lla.unwrap_or([0.; 3]),
            status: self.status,
            has_quality: self.quality.is_some(),
            quality: self.quality.unwrap_or(0),
            has_hdop: self.hdop.is_some(),
            hdop: self.hdop.unwrap_or(0.),
            has_covariance: self.covariance_enu.is_some(),
            covariance_enu: self.covariance_enu.unwrap_or([0.; 9]),
        }
    }
    fn from_wire(m: Self::Msg) -> Self {
        Self {
            key: c::Key::from_wire(m.key),
            timestamp_ns: m.timestamp_ns,
            receipt_timestamp_ns: m.receipt_timestamp_ns,
            time_source: m.time_source,
            lla: m.has_position.then_some(m.lla),
            status: m.status,
            quality: m.has_quality.then_some(m.quality),
            hdop: m.has_hdop.then_some(m.hdop),
            covariance_enu: m.has_covariance.then_some(m.covariance_enu),
        }
    }
}

impl Wire for c::BundleFrame {
    type Msg = m::BundleFrame;
    fn wire(&self) -> Self::Msg {
        m::BundleFrame {
            key: self.key.wire(),
            timestamp_ns: self.timestamp_ns,
            views: self
                .views
                .iter()
                .map(|v| m::CameraObservations {
                    camera: v.camera.wire(),
                    observations: v
                        .observations
                        .iter()
                        .map(|o| m::LandmarkObservation {
                            track_id: o.track_id,
                            pixel: o.pixel,
                            has_landmark: o.point_camera.is_some(),
                            point_camera: o.point_camera.unwrap_or([0.; 3]),
                        })
                        .collect(),
                })
                .collect(),
        }
    }
    fn from_wire(m: Self::Msg) -> Self {
        Self {
            key: c::Key::from_wire(m.key),
            timestamp_ns: m.timestamp_ns,
            views: m
                .views
                .into_iter()
                .map(|v| c::CameraObservations {
                    camera: c::CameraModel::from_wire(v.camera),
                    observations: v
                        .observations
                        .into_iter()
                        .map(|o| c::LandmarkObservation {
                            track_id: o.track_id,
                            pixel: o.pixel,
                            point_camera: o.has_landmark.then_some(o.point_camera),
                        })
                        .collect(),
                })
                .collect(),
        }
    }
}
impl Wire for c::OptimizedLandmark {
    type Msg = m::OptimizedLandmark;
    fn wire(&self) -> Self::Msg {
        m::OptimizedLandmark {
            key: self.key.wire(),
            component: self.component.wire(),
            position: self.position,
            observations: self.observations as u32,
        }
    }
    fn from_wire(m: Self::Msg) -> Self {
        Self {
            key: c::Key::from_wire(m.key),
            component: c::Key::from_wire(m.component),
            position: m.position,
            observations: m.observations as usize,
        }
    }
}

#[cfg(test)]
mod bundle_tests {
    use super::*;
    #[test]
    fn complete_stereo_observation_and_graph_roundtrip() {
        let frame = c::BundleFrame {
            key: c::Key::new("a", "s", 7),
            timestamp_ns: 1_790_502_544_606_403_113,
            views: (0..2)
                .map(|i| c::CameraObservations {
                    camera: c::CameraModel {
                        width: 640,
                        height: 480,
                        intrinsics: [400., 410., 320., 240.],
                        camera_to_body: c::Transform {
                            translation: [i as f64 * 0.095095, 0., 0.],
                            ..Default::default()
                        },
                    },
                    observations: vec![
                        c::LandmarkObservation {
                            track_id: 12,
                            pixel: [120.25, 140.75],
                            point_camera: Some([1., 2., 4.]),
                        },
                        c::LandmarkObservation {
                            track_id: 13,
                            pixel: [123., 134.],
                            point_camera: None,
                        },
                    ],
                })
                .collect(),
        };
        assert_eq!(
            serde_json::to_value(&frame).unwrap(),
            serde_json::to_value(c::BundleFrame::from_wire(frame.wire())).unwrap()
        );
        let graph = c::GraphSnapshot {
            backend_mode: c::BackendMode::GlobalBundleAdjustment,
            landmarks: vec![c::OptimizedLandmark {
                key: c::Key::new("a", "s", 12),
                component: c::Key::new("a", "s", 0),
                position: [1., 2., 3.],
                observations: 4,
            }],
            ..Default::default()
        };
        let restored = c::GraphSnapshot::from_wire(graph.wire());
        assert_eq!(restored.backend_mode, graph.backend_mode);
        assert_eq!(
            serde_json::to_value(restored.landmarks).unwrap(),
            serde_json::to_value(graph.landmarks).unwrap()
        );
    }
}
