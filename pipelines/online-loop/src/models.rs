//! TensorRT-only model adapters. XFeat descriptors are sampled at VIO's
//! measured feature locations, preserving exact metric landmark associations.
use crate::{Config, Error, Frame, Result};
use visloc_tensorrt::{DataType, Input, Output, Session};

pub const SEQUENCE: usize = 5;
const DIM: usize = 512;
const WIDTH: usize = 320;
const HEIGHT: usize = 224;

pub struct Models {
    jist: Session,
    xfeat: Session,
    matcher: Session,
    keypoints: usize,
}

#[derive(Clone)]
pub struct Features {
    pub pixels: Vec<[f32; 2]>,
    pub descriptors: Vec<f32>,
    pub image_size: [f32; 2],
}

fn values(outputs: &[Output], name: &str, shape: &[i64]) -> Result<Vec<f32>> {
    let out = outputs
        .iter()
        .find(|o| o.info.name == name)
        .ok_or_else(|| Error(format!("Missing model output {name}")))?;
    if out.info.shape != shape {
        return Err(Error(format!(
            "{name}: unexpected shape {:?}",
            out.info.shape
        )));
    }
    let values = out.to_f32().map_err(|e| Error(e.to_string()))?;
    if values.iter().any(|v| !v.is_finite()) {
        return Err(Error(format!("{name}: nonfinite inference output")));
    }
    Ok(values)
}

fn normalize(v: &mut [f32]) -> Result<()> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if !norm.is_finite() || norm < 1e-10 {
        return Err(Error("Degenerate model descriptor".into()));
    }
    for x in v {
        *x /= norm;
    }
    Ok(())
}

/// Half-pixel bilinear resize, grayscale replicated into RGB; NCHW float [0,1].
pub fn image_tensor(frame: &Frame, width: usize, height: usize) -> Vec<f32> {
    let mut plane = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        let sy = ((y as f64 + 0.5) * frame.height as f64 / height as f64 - 0.5)
            .clamp(0.0, (frame.height - 1) as f64);
        let y0 = sy.floor() as usize;
        let y1 = (y0 + 1).min(frame.height - 1);
        let dy = (sy - y0 as f64) as f32;
        for x in 0..width {
            let sx = ((x as f64 + 0.5) * frame.width as f64 / width as f64 - 0.5)
                .clamp(0.0, (frame.width - 1) as f64);
            let x0 = sx.floor() as usize;
            let x1 = (x0 + 1).min(frame.width - 1);
            let dx = (sx - x0 as f64) as f32;
            let at = |x, y| frame.gray[y * frame.width + x] as f32;
            plane.push(
                ((1.0 - dy) * ((1.0 - dx) * at(x0, y0) + dx * at(x1, y0))
                    + dy * ((1.0 - dx) * at(x0, y1) + dx * at(x1, y1)))
                    / 255.0,
            );
        }
    }
    plane.extend_from_within(..width * height);
    plane.extend_from_within(..width * height);
    plane
}

/// The JIST export starts at the ResNet convolution; ImageNet normalization
/// is external, as in the official JIST evaluation transform.
pub fn jist_image_tensor(frame: &Frame) -> Vec<f32> {
    let mut values = image_tensor(frame, 512, 288);
    for (c, plane) in values.chunks_mut(512 * 288).enumerate() {
        for value in plane {
            *value = (*value - [0.485, 0.456, 0.406][c]) / [0.229, 0.224, 0.225][c];
        }
    }
    values
}

fn cubic(x: f32) -> f32 {
    let x = x.abs();
    if x <= 1.0 {
        1.25 * x * x * x - 2.25 * x * x + 1.0
    } else if x < 2.0 {
        -0.75 * x * x * x + 3.75 * x * x - 6.0 * x + 3.0
    } else {
        0.0
    }
}

/// PyTorch grid_sample(bicubic, padding_mode=zeros, align_corners=false),
/// with XFeat's pixel normalization 2*xy/(W-1,H-1)-1.
fn sample_descriptor(dense: &[f32], xy: [f32; 2]) -> Result<Vec<f32>> {
    let (w, h) = (WIDTH / 8, HEIGHT / 8);
    let sx = xy[0] * w as f32 / (WIDTH - 1) as f32 - 0.5;
    let sy = xy[1] * h as f32 / (HEIGHT - 1) as f32 - 0.5;
    let mut descriptor = vec![0.0; 64];
    for y in sy.floor() as i32 - 1..=sy.floor() as i32 + 2 {
        for x in sx.floor() as i32 - 1..=sx.floor() as i32 + 2 {
            if x < 0 || y < 0 || x >= w as i32 || y >= h as i32 {
                continue;
            }
            let weight = cubic(sx - x as f32) * cubic(sy - y as f32);
            for c in 0..64 {
                descriptor[c] += weight * dense[(c * h + y as usize) * w + x as usize];
            }
        }
    }
    normalize(&mut descriptor)?;
    Ok(descriptor)
}

impl Models {
    pub fn new(config: &Config) -> Result<Self> {
        let open = |p| Session::from_file(p, config.device).map_err(|e| Error(e.to_string()));
        let models = Self {
            jist: open(&config.jist_engine)?,
            xfeat: open(&config.xfeat_engine)?,
            matcher: open(&config.lighterglue_engine)?,
            keypoints: config.matcher_keypoints,
        };
        for (session, expected) in [
            (&models.jist, vec![("input", vec![1, 5, 3, 288, 512])]),
            (&models.xfeat, vec![("input", vec![1, 3, 224, 320])]),
            (
                &models.matcher,
                vec![
                    ("mkpts0", vec![1, config.matcher_keypoints as i64, 2]),
                    ("feats0", vec![1, config.matcher_keypoints as i64, 64]),
                    ("image0_size", vec![2]),
                    ("mkpts1", vec![1, config.matcher_keypoints as i64, 2]),
                    ("feats1", vec![1, config.matcher_keypoints as i64, 64]),
                    ("image1_size", vec![2]),
                ],
            ),
        ] {
            if session.tensors().iter().filter(|t| t.is_input).count() != expected.len() {
                return Err(Error("Unexpected model input count".into()));
            }
            for (name, shape) in expected {
                if !session.tensors().iter().any(|t| {
                    t.is_input && t.name == name && t.shape == shape && t.dtype == DataType::F32
                }) {
                    return Err(Error(format!(
                        "Incompatible model input {name}; expected FP32 {shape:?}"
                    )));
                }
            }
        }
        Ok(models)
    }

    pub fn sequence(&mut self, images: &[Vec<f32>]) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
        if images.len() != SEQUENCE || images.iter().any(|x| x.len() != 3 * 288 * 512) {
            return Err(Error("JIST requires five distinct resized frames".into()));
        }
        let data: Vec<f32> = images.iter().flatten().copied().collect();
        let outputs = self
            .jist
            .run(&[Input::f32("input", &[1, 5, 3, 288, 512], &data)], 0)
            .map_err(|e| Error(e.to_string()))?;
        let mut sequence = values(&outputs, "output", &[1, DIM as i64])?;
        normalize(&mut sequence)?;
        let mut frames = values(&outputs, "frame_descriptors", &[5, DIM as i64])?;
        for row in frames.chunks_mut(DIM) {
            normalize(row)?;
        }
        Ok((sequence, frames.chunks(DIM).map(|x| x.to_vec()).collect()))
    }

    pub fn features(&mut self, frame: &Frame) -> Result<Features> {
        if frame.observations.len() < self.keypoints {
            return Ok(Features {
                pixels: Vec::new(),
                descriptors: Vec::new(),
                image_size: [frame.width as f32, frame.height as f32],
            });
        }
        let input = image_tensor(frame, WIDTH, HEIGHT);
        let outputs = self
            .xfeat
            .run(&[Input::f32("input", &[1, 3, 224, 320], &input)], 0)
            .map_err(|e| Error(e.to_string()))?;
        let mut dense = values(&outputs, "output", &[1, 64, 28, 40])?;
        for i in 0..28 * 40 {
            let norm = (0..64)
                .map(|c| dense[c * 28 * 40 + i].powi(2))
                .sum::<f32>()
                .sqrt()
                .max(1e-12);
            for c in 0..64 {
                dense[c * 28 * 40 + i] /= norm;
            }
        }
        let mut features = Features {
            pixels: Vec::new(),
            descriptors: Vec::new(),
            image_size: [frame.width as f32, frame.height as f32],
        };
        for observation in frame.observations.iter().take(self.keypoints) {
            let pixel = [observation.pixel.x as f32, observation.pixel.y as f32];
            let xy = [
                pixel[0] * WIDTH as f32 / frame.width as f32,
                pixel[1] * HEIGHT as f32 / frame.height as f32,
            ];
            features.descriptors.extend(sample_descriptor(&dense, xy)?);
            features.pixels.push(pixel);
        }
        Ok(features)
    }

    pub fn matches(&mut self, a: &Features, b: &Features) -> Result<Vec<(usize, usize, f32)>> {
        let (n, m) = (a.pixels.len(), b.pixels.len());
        if n == 0 || m == 0 {
            return Ok(Vec::new());
        }
        if n != self.keypoints || m != self.keypoints {
            return Err(Error(
                "LighterGlue feature count does not match its fixed TensorRT profile".into(),
            ));
        }
        let ak: Vec<f32> = a.pixels.iter().flatten().copied().collect();
        let bk: Vec<f32> = b.pixels.iter().flatten().copied().collect();
        let outputs = self
            .matcher
            .run(
                &[
                    Input::f32("mkpts0", &[1, n as i64, 2], &ak),
                    Input::f32("feats0", &[1, n as i64, 64], &a.descriptors),
                    Input::f32("image0_size", &[2], &a.image_size),
                    Input::f32("mkpts1", &[1, m as i64, 2], &bk),
                    Input::f32("feats1", &[1, m as i64, 64], &b.descriptors),
                    Input::f32("image1_size", &[2], &b.image_size),
                ],
                0,
            )
            .map_err(|e| Error(e.to_string()))?;
        let matched = outputs
            .iter()
            .find(|o| o.info.name == "matches")
            .ok_or_else(|| Error("Missing LighterGlue matches".into()))?;
        let indices = matched.to_i64().map_err(|e| Error(e.to_string()))?;
        let count = indices.len() / 2;
        if matched.info.shape != [count as i64, 2] {
            return Err(Error("Invalid match shape".into()));
        }
        let scores = values(&outputs, "scores", &[count as i64])?;
        let mut result = Vec::with_capacity(count);
        for (pair, score) in indices.chunks_exact(2).zip(scores) {
            if pair[0] < 0 || pair[0] >= n as i64 || pair[1] < 0 || pair[1] >= m as i64 {
                return Err(Error("LighterGlue returned invalid match indices".into()));
            }
            result.push((pair[0] as usize, pair[1] as usize, score));
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bicubic_descriptors_match_pytorch_at_interior_and_borders() {
        // Reference: normalize(sin(arange(64*28*40)*.013), dim=1), then
        // grid_sample(mode="bicubic", align_corners=False, padding_mode="zeros")
        // at 2*xy/[319,223]-1, followed by channel normalization.
        let mut dense: Vec<f32> = (0..64 * 28 * 40)
            .map(|i| (i as f32 * 0.013).sin())
            .collect();
        for i in 0..28 * 40 {
            let norm = (0..64)
                .map(|c| dense[c * 28 * 40 + i].powi(2))
                .sum::<f32>()
                .sqrt();
            for c in 0..64 {
                dense[c * 28 * 40 + i] /= norm;
            }
        }
        let channels = [0, 1, 7, 11, 23, 31, 47, 63];
        let expected = [
            [
                -0.016498443,
                0.16854812,
                0.1714975,
                0.02732173,
                0.17433962,
                -0.1605278,
                -0.10639066,
                -0.02799169,
            ],
            [
                0.035791326,
                0.144442,
                0.17809203,
                -0.02504788,
                0.15609299,
                -0.13108873,
                -0.060223754,
                0.024376823,
            ],
            [
                0.15417607,
                -0.14057502,
                -0.055510625,
                -0.1590743,
                -0.12659806,
                0.15210989,
                0.17577302,
                0.15935925,
            ],
        ];
        for (xy, expected) in [[0.0, 0.0], [157.25, 99.5], [319.0, 223.0]]
            .into_iter()
            .zip(expected)
        {
            let actual = sample_descriptor(&dense, xy).unwrap();
            for (channel, expected) in channels.into_iter().zip(expected) {
                assert!(
                    (actual[channel] - expected).abs() < 2e-5,
                    "{xy:?} channel {channel}: {} != {expected}",
                    actual[channel]
                );
            }
        }
    }
}
