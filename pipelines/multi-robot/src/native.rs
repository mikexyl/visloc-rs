//! GPU ownership stays on the caller's loop worker. ROS only transports the
//! resulting typed records; model execution never runs inside a ROS callback.
use crate::{sequence::*, *};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use visloc_core::geometry::{reproject, Pose};
use visloc_online_loop::{
    models::{jist_image_tensor, Features, Models},
    select_observations, Config, Frame,
};

pub struct EncodedSequence {
    pub sequence: Sequence,
    pub descriptors: FrameDescriptors,
    pub features: Vec<FeatureFrame>,
}
pub struct Encoder {
    models: Models,
    camera: CameraModel,
    config: Config,
    builder: SequenceBuilder,
    frames: BTreeMap<u64, Frame>,
    owner: Key,
    next_sequence: u64,
    model_id: String,
    pub resets: u64,
    pub skipped: u64,
}
impl Encoder {
    pub fn new(config: Config, camera: CameraModel, owner: Key) -> Result<Self> {
        config.validate().map_err(|e| Error(e.to_string()))?;
        if config.matcher_keypoints != 128 {
            return Err(Error(
                "SB-SLAM mode requires exactly 128 real matcher features".into(),
            ));
        }
        camera.camera()?;
        owner.validate()?;
        let mut hash = Sha256::new();
        hash.update(b"visloc-jist-imagenet-rgb-five-xfeat-lighterglue-v1");
        // ONNX identities are portable across GPUs, unlike engine bytes.
        let manifest = config.jist_engine.parent().unwrap().join("manifest.json");
        let manifest: Option<serde_json::Value> = std::fs::read(&manifest)
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok());
        for (name, path) in [
            ("jist", &config.jist_engine),
            ("xfeat", &config.xfeat_engine),
            ("lighterglue", &config.lighterglue_engine),
        ] {
            if let Some(sha) = manifest
                .as_ref()
                .and_then(|m| m["models"][name]["onnx_sha256"].as_str())
            {
                hash.update(sha.as_bytes());
            } else {
                hash.update(std::fs::read(path).map_err(|e| Error(e.to_string()))?);
            }
        }
        hash.update(config.matcher_keypoints.to_le_bytes());
        let model_id = format!("{:x}", hash.finalize());
        Ok(Self {
            models: Models::new(&config).map_err(|e| Error(e.to_string()))?,
            camera,
            config,
            builder: SequenceBuilder::default(),
            frames: BTreeMap::new(),
            owner,
            next_sequence: 0,
            model_id,
            resets: 0,
            skipped: 0,
        })
    }
    pub fn push(
        &mut self,
        mut frame: Frame,
        record: KeyframeRecord,
    ) -> Result<Option<EncodedSequence>> {
        frame.validate().map_err(|e| Error(e.to_string()))?;
        frame.body_to_world.rotation.renormalize();
        if !record.key.same_session(&self.owner) || frame.keyframe_index != record.key.id {
            return Err(Error("frame/session identity mismatch".into()));
        }
        let center = frame
            .body_to_world
            .compose(&self.camera.camera_to_body.se3()?)
            .translation;
        let meta = KeyframeMeta {
            record,
            camera_position: center.into(),
            landmarks: frame
                .observations
                .iter()
                .filter(|o| o.point_world.is_some())
                .map(|o| o.track_id)
                .collect(),
        };
        let part = self.builder.push(meta)?;
        if part.reset {
            self.frames.clear();
            self.resets += 1;
        }
        if !part.retained {
            self.skipped += 1;
            return Ok(None);
        }
        self.frames.insert(frame.keyframe_index, frame);
        let Some(block) = part.complete else {
            return Ok(None);
        };
        let positions: Vec<_> = block.iter().map(|m| m.camera_position).collect();
        let selection = select_five(&positions)?;
        let members: Vec<_> = block.iter().map(|m| m.record.key.clone()).collect();
        let selected: Vec<_> = selection.iter().map(|&i| members[i].clone()).collect();
        let images: Vec<_> = selected
            .iter()
            .map(|k| jist_image_tensor(&self.frames[&k.id]))
            .collect();
        let (descriptor, rows) = self
            .models
            .sequence(&images)
            .map_err(|e| Error(e.to_string()))?;
        let key = Key {
            id: self.next_sequence,
            ..self.owner.clone()
        };
        self.next_sequence += 1;
        let sequence = Sequence {
            key: key.clone(),
            start_ns: block[0].record.timestamp_ns,
            end_ns: block[9].record.timestamp_ns,
            excluded_keyframes: self.builder.neighborhood(&members),
            members,
            selected: selected.clone(),
            descriptor,
            model_id: self.model_id.clone(),
        };
        let descriptors = FrameDescriptors {
            sequence: key,
            frames: selected.clone(),
            descriptors: rows.into_iter().flatten().collect(),
        };
        let mut features = Vec::new();
        for key in selected {
            let mut frame = self.frames.remove(&key.id).unwrap();
            frame.observations = select_observations(
                frame.observations,
                frame.width,
                frame.height,
                self.config.matcher_keypoints,
            );
            let extracted = self
                .models
                .features(&frame)
                .map_err(|e| Error(e.to_string()))?;
            let camera_from_world = frame
                .body_to_world
                .compose(&self.camera.camera_to_body.se3()?)
                .inverse();
            let camera = self.camera.camera()?;
            let points = frame
                .observations
                .iter()
                .take(extracted.pixels.len())
                .map(|o| {
                    o.point_world.and_then(|p| {
                        let p = camera_from_world.transform_point(&p);
                        let pixel = reproject(&camera, &Pose::identity(), &p)?;
                        (p.z > 0.1 && p.z < 150. && (pixel - o.pixel).norm() < 3.)
                            .then_some(p.coords.into())
                    })
                })
                .collect();
            features.push(FeatureFrame {
                key,
                timestamp_ns: frame.timestamp_ns,
                camera: self.camera.clone(),
                track_ids: frame
                    .observations
                    .iter()
                    .take(extracted.pixels.len())
                    .map(|o| o.track_id)
                    .collect(),
                pixels: extracted.pixels,
                descriptors: extracted.descriptors,
                points_camera: points,
            });
        }
        self.frames.clear();
        Ok(Some(EncodedSequence {
            sequence,
            descriptors,
            features,
        }))
    }
    pub fn verify(
        &mut self,
        a: &FeatureFrame,
        b: &FeatureFrame,
        pair: Pair,
        similarity: f32,
    ) -> Result<(Option<LoopConstraint>, Verification)> {
        a.validate()?;
        b.validate()?;
        let to_features = |f: &FeatureFrame| Features {
            pixels: f.pixels.clone(),
            descriptors: f.descriptors.clone(),
            image_size: [f.camera.width as f32, f.camera.height as f32],
        };
        let matches = self
            .models
            .matches(&to_features(a), &to_features(b))
            .map_err(|e| Error(e.to_string()))?;
        geometry::verify(a, b, &matches, pair, similarity)
    }
}
