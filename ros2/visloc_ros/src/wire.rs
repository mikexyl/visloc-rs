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
