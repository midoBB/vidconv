//! External tool discovery and capability probing.
use std::collections::{HashMap, HashSet};
use std::ffi::OsString;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};

pub struct Tools {
    pub ffmpeg: PathBuf,
    pub ffprobe: PathBuf,
}

pub struct Caps {
    pub tonemap: bool,
    pub vaapi_device: Option<PathBuf>,
    /// Decode profiles `vainfo` lists for the device; `None` if vainfo is missing or unusable.
    listed: Option<HashSet<String>>,
    /// Per-run cache of "can VAAPI decode this source codec", filled lazily.
    decode: Mutex<HashMap<String, bool>>,
}

impl Caps {
    /// Whether the VAAPI device can decode `codec`, probed once per run on `sample`
    /// (first file seen with that codec) with a 1-second hardware-only decode.
    pub fn vaapi_decodes(&self, ffmpeg: &Path, codec: &str, sample: &Path) -> bool {
        let Some(dev) = &self.vaapi_device else {
            return false;
        };
        if let (Some(listed), Some(prefix)) = (&self.listed, profile_prefix(codec))
            && !listed.iter().any(|p| p.starts_with(prefix))
        {
            return false;
        }
        let mut cache = self.decode.lock().unwrap();
        *cache
            .entry(codec.to_string())
            .or_insert_with(|| probe_decode(ffmpeg, dev, sample))
    }
}

fn find(name: &str) -> Option<PathBuf> {
    std::env::split_paths(&std::env::var_os("PATH")?)
        .map(|d| d.join(name))
        .find(|p| {
            p.metadata()
                .map(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
                .unwrap_or(false)
        })
}

/// Returns the missing tools on failure.
pub fn locate() -> Result<Tools, Vec<&'static str>> {
    match (find("ffmpeg"), find("ffprobe")) {
        (Some(ffmpeg), Some(ffprobe)) => Ok(Tools { ffmpeg, ffprobe }),
        (a, b) => Err([("ffmpeg", a.is_none()), ("ffprobe", b.is_none())]
            .into_iter()
            .filter_map(|(n, missing)| missing.then_some(n))
            .collect()),
    }
}

/// Makes a path safe to pass as a positional/`-i` argument (no leading '-').
pub fn arg_path(p: &Path) -> OsString {
    if p.is_relative() && p.to_string_lossy().starts_with('-') {
        Path::new(".").join(p).into_os_string()
    } else {
        p.as_os_str().to_owned()
    }
}

fn has_filters(ffmpeg: &Path, names: &[&str]) -> bool {
    let Ok(out) = Command::new(ffmpeg)
        .args(["-hide_banner", "-filters"])
        .stdin(Stdio::null())
        .output()
    else {
        return false;
    };
    let text = String::from_utf8_lossy(&out.stdout);
    names
        .iter()
        .all(|n| text.lines().any(|l| l.split_whitespace().nth(1) == Some(n)))
}

fn render_nodes() -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir("/dev/dri")
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .is_some_and(|n| n.to_string_lossy().starts_with("renderD"))
        })
        .collect();
    v.sort();
    v
}

/// 1-frame VAAPI test encode; returns the first working device.
fn detect_vaapi(ffmpeg: &Path, wanted: Option<&Path>) -> Option<PathBuf> {
    let candidates = match wanted {
        Some(p) => vec![p.to_path_buf()],
        None => render_nodes(),
    };
    candidates.into_iter().find(|dev| {
        let mut child = match Command::new(ffmpeg)
            .args(["-hide_banner", "-v", "error", "-nostdin", "-init_hw_device"])
            .arg(format!("vaapi=card:{}", dev.display()))
            .args([
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=128x128:d=0.2",
                "-vf",
                "format=nv12,hwupload",
                "-filter_hw_device",
                "card",
                "-c:v",
                "h264_vaapi",
                "-frames:v",
                "1",
                "-f",
                "null",
                "-",
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
        {
            Ok(c) => c,
            Err(_) => return false,
        };
        let start = Instant::now();
        loop {
            match child.try_wait() {
                Ok(Some(st)) => return st.success(),
                Ok(None) if start.elapsed() < Duration::from_secs(10) => {
                    std::thread::sleep(Duration::from_millis(50))
                }
                _ => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return false;
                }
            }
        }
    })
}

/// `vainfo` profile-name prefix for an ffmpeg codec name; `None` if unmapped.
fn profile_prefix(codec: &str) -> Option<&'static str> {
    Some(match codec {
        "h264" => "VAProfileH264",
        "hevc" => "VAProfileHEVC",
        "av1" => "VAProfileAV1",
        "vp9" => "VAProfileVP9",
        "vp8" => "VAProfileVP8",
        "mpeg2video" => "VAProfileMPEG2",
        "vc1" => "VAProfileVC1",
        "mjpeg" => "VAProfileJPEG",
        _ => return None,
    })
}

/// Profiles with a decode (VLD) entrypoint in `vainfo` output.
fn parse_vainfo(text: &str) -> HashSet<String> {
    text.lines()
        .filter_map(|l| l.split_once(':'))
        .filter(|(_, e)| e.trim() == "VAEntrypointVLD")
        .map(|(p, _)| p.trim().to_string())
        .filter(|p| p.starts_with("VAProfile"))
        .collect()
}

/// Fast pre-filter: ask `vainfo` which profiles the device can decode.
fn list_decode_profiles(dev: &Path) -> Option<HashSet<String>> {
    let vainfo = find("vainfo")?;
    let out = Command::new(vainfo)
        .args(["--display", "drm", "--device"])
        .arg(dev)
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    let set = parse_vainfo(&String::from_utf8_lossy(&out.stdout));
    (out.status.success() && !set.is_empty()).then_some(set)
}

fn succeeds_within(mut child: std::process::Child, limit: Duration) -> bool {
    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(st)) => return st.success(),
            Ok(None) if start.elapsed() < limit => std::thread::sleep(Duration::from_millis(20)),
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return false;
            }
        }
    }
}

/// Decodes 1s of `sample` with output forced to VAAPI surfaces, so ffmpeg errors
/// instead of silently falling back to software decoding.
fn probe_decode(ffmpeg: &Path, dev: &Path, sample: &Path) -> bool {
    Command::new(ffmpeg)
        .args(["-hide_banner", "-v", "error", "-nostdin", "-init_hw_device"])
        .arg(format!("vaapi=card:{}", dev.display()))
        .args([
            "-hwaccel",
            "vaapi",
            "-hwaccel_device",
            "card",
            "-hwaccel_output_format",
            "vaapi",
            "-t",
            "1",
            "-i",
        ])
        .arg(arg_path(sample))
        .args(["-map", "0:v:0", "-frames:v", "5", "-f", "null", "-"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .is_ok_and(|c| succeeds_within(c, Duration::from_secs(10)))
}

pub fn detect(tools: &Tools, want_hw: bool, device: Option<&Path>) -> Caps {
    let vaapi_device = if want_hw {
        detect_vaapi(&tools.ffmpeg, device)
    } else {
        None
    };
    Caps {
        listed: vaapi_device.as_deref().and_then(list_decode_profiles),
        vaapi_device,
        tonemap: has_filters(&tools.ffmpeg, &["zscale", "tonemap"]),
        decode: Mutex::new(HashMap::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vainfo_output_is_parsed() {
        let text = "vainfo: VA-API version: 1.20\n\
            \x20     VAProfileH264Main               : VAEntrypointVLD\n\
            \x20     VAProfileH264Main               : VAEntrypointEncSlice\n\
            \x20     VAProfileHEVCMain10             : VAEntrypointEncSlice\n\
            \x20     VAProfileVP9Profile0            : VAEntrypointVLD\n";
        let set = parse_vainfo(text);
        assert_eq!(set.len(), 2);
        assert!(set.contains("VAProfileH264Main") && set.contains("VAProfileVP9Profile0"));
    }

    #[test]
    fn leading_dash_is_protected() {
        assert_eq!(arg_path(Path::new("-x.mkv")), OsString::from("./-x.mkv"));
        assert_eq!(
            arg_path(Path::new("/a/-x.mkv")),
            OsString::from("/a/-x.mkv")
        );
        assert_eq!(arg_path(Path::new("a.mkv")), OsString::from("a.mkv"));
    }
}
