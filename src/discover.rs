//! One discovery path for files, directories and `--all`.
use crate::cli::SortBy;
use crate::probe::{self, VideoInfo};
use rayon::prelude::*;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

const VIDEO_EXTS: &[&str] = &[
    "mp4", "mkv", "avi", "mov", "webm", "m4v", "wmv", "flv", "mpg", "mpeg", "ts", "m2ts", "mts",
    "3gp", "ogv", "vob", "divx", "asf", "f4v", "mxf",
];

pub struct Candidate {
    pub path: PathBuf,
    pub info: VideoInfo,
    pub size: u64,
    pub mtime: SystemTime,
}

pub struct Failure {
    pub path: PathBuf,
    pub reason: String,
    /// The user named this file explicitly (as opposed to finding it in a directory).
    pub explicit: bool,
}

#[derive(Default)]
pub struct Found {
    pub files: Vec<Candidate>,
    pub failures: Vec<Failure>,
}

pub struct Opts<'a> {
    pub ffprobe: &'a Path,
    pub recursive: bool,
    /// Several files / directories: apply cutoff and skip already converted files up front.
    pub multi: bool,
    pub cutoff: u32,
    pub sort: SortBy,
}

fn is_video_name(p: &Path) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| VIDEO_EXTS.contains(&e.to_ascii_lowercase().as_str()))
}

fn walk(dir: &Path, recursive: bool, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for e in rd.flatten() {
        if e.file_name().to_string_lossy().starts_with('.') {
            continue; // hidden files, including our own temp outputs
        }
        let Ok(ft) = e.file_type() else { continue };
        let p = e.path();
        if ft.is_dir() {
            if recursive {
                walk(&p, recursive, out);
            }
        } else if (ft.is_file() || ft.is_symlink()) && is_video_name(&p) {
            out.push(p);
        }
    }
}

/// Expand inputs to `(path, explicit)`, de-duplicated by canonical path.
fn expand(inputs: &[PathBuf], recursive: bool) -> Vec<(PathBuf, bool)> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for input in inputs {
        let mut found = Vec::new();
        let explicit = !input.is_dir();
        if explicit {
            found.push(input.clone());
        } else {
            walk(input, recursive, &mut found);
            found.sort();
        }
        for p in found {
            let key = std::fs::canonicalize(&p).unwrap_or_else(|_| p.clone());
            if seen.insert(key) {
                out.push((p, explicit));
            }
        }
    }
    out
}

pub fn discover(inputs: &[PathBuf], o: &Opts) -> Found {
    let items = expand(inputs, o.recursive);
    let results: Vec<Result<Candidate, Failure>> = items
        .par_iter()
        .map(|(path, explicit)| {
            let fail = |reason: String| Failure {
                path: path.clone(),
                reason,
                explicit: *explicit,
            };
            let meta = std::fs::metadata(path).map_err(|e| fail(format!("cannot access: {e}")))?;
            if !meta.is_file() {
                return Err(fail("not a regular file".into()));
            }
            let info = probe::probe(o.ffprobe, path).map_err(|e| fail(e.to_string()))?;
            Ok(Candidate {
                path: path.clone(),
                info,
                size: meta.len(),
                mtime: meta.modified().unwrap_or(SystemTime::UNIX_EPOCH),
            })
        })
        .collect();

    let mut found = Found::default();
    for r in results {
        match r {
            Ok(c) => found.files.push(c),
            Err(f) => found.failures.push(f),
        }
    }
    if o.multi {
        found
            .files
            .retain(|c| !c.info.tagged && !below_cutoff(&c.info, o.cutoff));
    }
    match o.sort {
        SortBy::Size => found
            .files
            .sort_by(|a, b| b.size.cmp(&a.size).then_with(|| a.path.cmp(&b.path))),
        SortBy::Date => found
            .files
            .sort_by(|a, b| b.mtime.cmp(&a.mtime).then_with(|| a.path.cmp(&b.path))),
    }
    found
}

/// Low-bitrate files are left alone, except HEVC/AV1 which are always worth re-encoding.
pub fn below_cutoff(info: &VideoInfo, cutoff: u32) -> bool {
    info.bitrate_kbps < cutoff as u64 && !matches!(info.codec.as_str(), "hevc" | "av1")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn expand_filters_hidden_nonvideo_and_dedups() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path();
        for n in [
            "a.mkv",
            "b.MP4",
            ".hidden.mkv",
            ".vidconv-1-0.tmp.mp4",
            "notes.txt",
            "noext",
        ] {
            fs::write(p.join(n), b"x").unwrap();
        }
        fs::create_dir(p.join("sub")).unwrap();
        fs::write(p.join("sub/c.avi"), b"x").unwrap();
        std::os::unix::fs::symlink(p.join("a.mkv"), p.join("link.mkv")).unwrap();
        let names = |r: bool| -> Vec<String> {
            let mut v: Vec<_> = expand(&[p.to_path_buf(), p.join("a.mkv")], r)
                .into_iter()
                .map(|(x, _)| x.strip_prefix(p).unwrap().to_string_lossy().into_owned())
                .collect();
            v.sort();
            v
        };
        // link.mkv and a.mkv are the same file → one entry
        assert_eq!(names(false).len(), 2);
        assert_eq!(names(true).len(), 3);
        assert!(
            !names(true)
                .iter()
                .any(|n| n.contains("hidden") || n.contains("tmp") || n.contains("txt"))
        );
    }

    #[test]
    fn cutoff_rule() {
        let mut i = VideoInfo {
            codec: "h264".into(),
            width: 1,
            height: 1,
            sar: 1.0,
            rotation: 0,
            fps: 1.0,
            bitrate_kbps: 1000,
            duration: 1.0,
            pix_fmt: String::new(),
            color_transfer: String::new(),
            has_audio: false,
            audio_channels: 0,
            video_index: 0,
            creation_time: None,
            tagged: false,
        };
        assert!(below_cutoff(&i, 3000));
        i.codec = "hevc".into();
        assert!(!below_cutoff(&i, 3000));
    }
}
