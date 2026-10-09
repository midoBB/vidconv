//! ffprobe JSON → `VideoInfo`. Parsing is pure so it can be unit-tested.
use serde::Deserialize;
use std::path::Path;
use std::process::{Command, Stdio};

#[derive(Debug, thiserror::Error)]
pub enum ProbeError {
    #[error("ffprobe failed: {0}")]
    Ffprobe(String),
    #[error("invalid ffprobe output: {0}")]
    Json(#[from] serde_json::Error),
    #[error("no video stream")]
    NoVideo,
    #[error("invalid frame size")]
    BadSize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VideoInfo {
    pub codec: String,
    pub width: u32,
    pub height: u32,
    /// Sample aspect ratio (w/h); 1.0 when unknown.
    pub sar: f64,
    /// Clockwise degrees (0/90/180/270) the player rotates the frame.
    pub rotation: u32,
    pub fps: f64,
    pub bitrate_kbps: u64,
    pub duration: f64,
    pub pix_fmt: String,
    pub color_transfer: String,
    pub has_audio: bool,
    pub audio_channels: u32,
    /// Container stream index of the chosen video stream (not always 0: cover art, etc.).
    pub video_index: u32,
    pub creation_time: Option<String>,
    pub tagged: bool,
}

impl VideoInfo {
    pub fn is_hdr(&self) -> bool {
        matches!(self.color_transfer.as_str(), "smpte2084" | "arib-std-b67")
    }

    pub fn is_high_bit_depth(&self) -> bool {
        self.pix_fmt.contains("10") || self.pix_fmt.contains("12") || self.pix_fmt.contains("16")
    }
}

#[derive(Deserialize)]
struct Raw {
    #[serde(default)]
    streams: Vec<RawStream>,
    format: Option<RawFormat>,
}

#[derive(Deserialize, Default)]
struct RawStream {
    index: Option<u32>,
    channels: Option<u32>,
    codec_type: Option<String>,
    codec_name: Option<String>,
    width: Option<u32>,
    height: Option<u32>,
    sample_aspect_ratio: Option<String>,
    r_frame_rate: Option<String>,
    avg_frame_rate: Option<String>,
    bit_rate: Option<String>,
    duration: Option<String>,
    pix_fmt: Option<String>,
    color_transfer: Option<String>,
    #[serde(default)]
    disposition: Disposition,
    #[serde(default)]
    side_data_list: Vec<SideData>,
    #[serde(default)]
    tags: Tags,
}

#[derive(Deserialize, Default)]
struct Disposition {
    #[serde(default)]
    attached_pic: u8,
}

#[derive(Deserialize, Default)]
struct SideData {
    rotation: Option<f64>,
}

#[derive(Deserialize, Default)]
struct Tags {
    rotate: Option<String>,
}

#[derive(Deserialize, Default)]
struct RawFormat {
    duration: Option<String>,
    bit_rate: Option<String>,
    size: Option<String>,
    #[serde(default)]
    tags: std::collections::HashMap<String, String>,
}

/// "30000/1001" or "25" → fps. Returns None for 0/0 and garbage.
pub fn parse_rate(s: &str) -> Option<f64> {
    let v = match s.split_once('/') {
        Some((n, d)) => {
            let (n, d) = (n.trim().parse::<f64>().ok()?, d.trim().parse::<f64>().ok()?);
            if d == 0.0 {
                return None;
            }
            n / d
        }
        None => s.trim().parse::<f64>().ok()?,
    };
    (v.is_finite() && v > 0.0).then_some(v)
}

fn parse_sar(s: Option<&str>) -> f64 {
    s.and_then(|s| s.split_once(':'))
        .and_then(|(n, d)| Some((n.parse::<f64>().ok()?, d.parse::<f64>().ok()?)))
        .filter(|(n, d)| *n > 0.0 && *d > 0.0)
        .map(|(n, d)| n / d)
        .unwrap_or(1.0)
}

fn normalize_rotation(deg: f64) -> u32 {
    // ffmpeg reports display-matrix rotation counter-clockwise (e.g. -90 for a
    // clockwise-90 phone video); normalise to clockwise 0/90/180/270.
    let cw = (-deg).rem_euclid(360.0).round() as i64;
    ((cw + 45) / 90 % 4 * 90) as u32
}

pub fn parse(json: &[u8], file_size: Option<u64>) -> Result<VideoInfo, ProbeError> {
    let raw: Raw = serde_json::from_slice(json)?;
    let fmt = raw.format.unwrap_or_default();
    let v = raw
        .streams
        .iter()
        .find(|s| s.codec_type.as_deref() == Some("video") && s.disposition.attached_pic == 0)
        .ok_or(ProbeError::NoVideo)?;
    let (width, height) = match (v.width, v.height) {
        (Some(w), Some(h)) if w > 0 && h > 0 => (w, h),
        _ => return Err(ProbeError::BadSize),
    };
    let num = |s: &Option<String>| s.as_deref().and_then(|s| s.parse::<f64>().ok());
    let duration = num(&v.duration)
        .or_else(|| num(&fmt.duration))
        .filter(|d| *d > 0.0)
        .unwrap_or(0.0);
    let fps = v
        .avg_frame_rate
        .as_deref()
        .and_then(parse_rate)
        .or_else(|| v.r_frame_rate.as_deref().and_then(parse_rate))
        .unwrap_or(24.0);
    // Stream bitrate -> container bitrate -> size/duration (MKV/WebM often lack both).
    let bps = num(&v.bit_rate)
        .or_else(|| num(&fmt.bit_rate))
        .or_else(|| {
            let size = num(&fmt.size).or(file_size.map(|s| s as f64))?;
            (duration > 0.0).then(|| size * 8.0 / duration)
        })
        .unwrap_or(f64::MAX);
    let rotation = v
        .side_data_list
        .iter()
        .find_map(|s| s.rotation)
        .or_else(|| {
            v.tags
                .rotate
                .as_deref()
                .and_then(|r| r.parse().ok())
                .map(|r: f64| -r)
        })
        .map(normalize_rotation)
        .unwrap_or(0);
    let tagged = fmt
        .tags
        .iter()
        .any(|(k, val)| k.eq_ignore_ascii_case("vidconv") && val == "1");
    let creation_time = fmt
        .tags
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("creation_time"))
        .map(|(_, v)| v.clone());
    Ok(VideoInfo {
        codec: v.codec_name.clone().unwrap_or_default(),
        width,
        height,
        sar: parse_sar(v.sample_aspect_ratio.as_deref()),
        rotation,
        fps,
        bitrate_kbps: if bps.is_finite() && bps < u64::MAX as f64 {
            (bps / 1000.0) as u64
        } else {
            u64::MAX
        },
        duration,
        pix_fmt: v.pix_fmt.clone().unwrap_or_default(),
        color_transfer: v.color_transfer.clone().unwrap_or_default(),
        has_audio: raw
            .streams
            .iter()
            .any(|s| s.codec_type.as_deref() == Some("audio")),
        audio_channels: raw
            .streams
            .iter()
            .find(|s| s.codec_type.as_deref() == Some("audio"))
            .and_then(|s| s.channels)
            .unwrap_or(0),
        video_index: v.index.unwrap_or(0),
        creation_time,
        tagged,
    })
}

pub fn probe(ffprobe: &Path, file: &Path) -> Result<VideoInfo, ProbeError> {
    let out = Command::new(ffprobe)
        .args([
            "-v",
            "error",
            "-of",
            "json",
            "-show_format",
            "-show_streams",
            "-i",
        ])
        .arg(crate::tools::arg_path(file))
        .stdin(Stdio::null())
        .output()
        .map_err(|e| ProbeError::Ffprobe(e.to_string()))?;
    if !out.status.success() {
        // ffprobe prints several diagnostic lines; the last one names the actual problem.
        let stderr = String::from_utf8_lossy(&out.stderr);
        let reason = stderr
            .lines()
            .rev()
            .find(|l| !l.trim().is_empty())
            .unwrap_or("unknown error");
        return Err(ProbeError::Ffprobe(reason.trim().to_string()));
    }
    parse(&out.stdout, std::fs::metadata(file).ok().map(|m| m.len()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rates() {
        assert_eq!(parse_rate("25"), Some(25.0));
        assert!((parse_rate("30000/1001").unwrap() - 29.97).abs() < 0.01);
        assert_eq!(parse_rate("0/0"), None);
        assert_eq!(parse_rate("N/A"), None);
    }

    #[test]
    fn rotation() {
        assert_eq!(normalize_rotation(-90.0), 90);
        assert_eq!(normalize_rotation(90.0), 270);
        assert_eq!(normalize_rotation(0.0), 0);
        assert_eq!(normalize_rotation(180.0), 180);
    }

    #[test]
    fn missing_everything_is_handled() {
        let j = br#"{"streams":[{"codec_type":"video","codec_name":"vp9","width":640,"height":360,
            "r_frame_rate":"0/0","duration":"N/A"}],"format":{"duration":"10.0","size":"1250000"}}"#;
        let i = parse(j, None).unwrap();
        assert_eq!(i.duration, 10.0);
        assert_eq!(i.bitrate_kbps, 1000); // estimated from size/duration
        assert_eq!(i.fps, 24.0);
    }

    #[test]
    fn no_video_and_attached_pic() {
        let j = br#"{"streams":[{"codec_type":"video","width":10,"height":10,"disposition":{"attached_pic":1}}],"format":{}}"#;
        assert!(matches!(parse(j, None), Err(ProbeError::NoVideo)));
    }

    #[test]
    fn tag_and_rotation_side_data() {
        let j = br#"{"streams":[{"codec_type":"video","codec_name":"h264","width":1920,"height":1080,
            "side_data_list":[{"rotation":-90}]},{"codec_type":"audio"}],"format":{"tags":{"VIDCONV":"1"}}}"#;
        let i = parse(j, Some(1000)).unwrap();
        assert!(i.tagged && i.has_audio);
        assert_eq!(i.rotation, 90);
    }
}
