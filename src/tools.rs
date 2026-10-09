//! External tool discovery and capability probing.
use std::ffi::OsString;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

pub struct Tools {
    pub ffmpeg: PathBuf,
    pub ffprobe: PathBuf,
}

pub struct Caps {
    pub tonemap: bool,
    pub vaapi_device: Option<PathBuf>,
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

pub fn detect(tools: &Tools, want_hw: bool, device: Option<&Path>) -> Caps {
    Caps {
        tonemap: has_filters(&tools.ffmpeg, &["zscale", "tonemap"]),
        vaapi_device: if want_hw {
            detect_vaapi(&tools.ffmpeg, device)
        } else {
            None
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
