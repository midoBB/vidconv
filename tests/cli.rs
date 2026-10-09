//! End-to-end tests: run the real binary against small ffmpeg-generated clips.
use serde_json::Value;
use std::fs;
use std::path::Path;
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

fn have_ffmpeg() -> bool {
    ["ffmpeg", "ffprobe"]
        .iter()
        .all(|t| Command::new(t).arg("-version").output().is_ok())
}

macro_rules! require_ffmpeg {
    () => {
        if !have_ffmpeg() {
            eprintln!("ffmpeg not installed; skipping");
            return;
        }
    };
}

fn ffmpeg(args: &[&str]) {
    let st = Command::new("ffmpeg")
        .args(["-v", "error", "-y", "-nostdin"])
        .args(args)
        .status()
        .unwrap();
    assert!(st.success(), "ffmpeg {args:?} failed");
}

/// Lossless-ish H.264 clip: high bitrate, so conversion always shrinks it.
fn make_clip(path: &Path, size: &str, secs: u32) {
    ffmpeg(&[
        "-f",
        "lavfi",
        "-i",
        &format!("testsrc2=s={size}:d={secs}:r=30"),
        "-f",
        "lavfi",
        "-i",
        &format!("sine=d={secs}"),
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "5",
        "-c:a",
        "aac",
        "-shortest",
        path.to_str().unwrap(),
    ]);
}

fn probe(path: &Path) -> Value {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-of",
            "json",
            "-show_streams",
            "-show_format",
            "-i",
        ])
        .arg(path)
        .output()
        .unwrap();
    serde_json::from_slice(&out.stdout).unwrap()
}

fn video(v: &Value) -> &Value {
    v["streams"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["codec_type"] == "video")
        .unwrap()
}

fn vidconv(dir: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_vidconv"))
        .current_dir(dir)
        .args(["--no-hw", "--preset", "ultrafast", "--quiet"])
        .args(args)
        .output()
        .unwrap()
}

fn text(b: &[u8]) -> String {
    String::from_utf8_lossy(b).into_owned()
}

fn leftovers(dir: &Path) -> Vec<String> {
    fs::read_dir(dir)
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with(".vidconv-"))
        .collect()
}

#[test]
fn replace_converts_mkv_to_mp4_tags_it_and_skips_on_rerun() {
    require_ffmpeg!();
    let d = tempfile::tempdir().unwrap();
    make_clip(&d.path().join("in.mkv"), "640x360", 2);
    let o = vidconv(d.path(), &["in.mkv"]);
    assert!(o.status.success(), "{}", text(&o.stderr));
    assert!(
        !d.path().join("in.mkv").exists(),
        "original must be replaced"
    );
    let p = probe(&d.path().join("in.mp4"));
    assert_eq!(p["format"]["tags"]["vidconv"], "1");
    assert_eq!(video(&p)["codec_name"], "h264");
    assert_eq!(video(&p)["pix_fmt"], "yuv420p");
    assert!(leftovers(d.path()).is_empty());
    assert!(!d.path().join("vidconv_errors.log").exists());

    let o = Command::new(env!("CARGO_BIN_EXE_vidconv"))
        .current_dir(d.path())
        .args(["--no-hw", "in.mp4"])
        .output()
        .unwrap();
    assert!(o.status.success());
    assert!(
        text(&o.stdout).contains("already converted"),
        "{}",
        text(&o.stdout)
    );
}

#[test]
fn in_place_mp4_replace_keeps_the_converted_file() {
    // Regression: the original path was unlinked after the atomic rename, deleting the result.
    require_ffmpeg!();
    let d = tempfile::tempdir().unwrap();
    for n in ["a.mp4", "b.mp4"] {
        make_clip(&d.path().join(n), "640x360", 2);
    }
    let before = fs::metadata(d.path().join("a.mp4")).unwrap().len();
    let o = vidconv(d.path(), &["--all"]);
    assert!(o.status.success(), "{}", text(&o.stderr));
    for n in ["a.mp4", "b.mp4"] {
        let path = d.path().join(n);
        assert!(path.exists(), "{n} must survive in-place conversion");
        assert_eq!(probe(&path)["format"]["tags"]["vidconv"], "1");
    }
    assert!(fs::metadata(d.path().join("a.mp4")).unwrap().len() < before);
    assert!(leftovers(d.path()).is_empty());
}

#[test]
fn keep_never_clobbers_existing_files() {
    require_ffmpeg!();
    let d = tempfile::tempdir().unwrap();
    make_clip(&d.path().join("a.mkv"), "320x240", 1);
    fs::write(d.path().join("a-720.mp4"), b"mine").unwrap();
    let o = vidconv(d.path(), &["-k", "a.mkv"]);
    assert!(o.status.success(), "{}", text(&o.stderr));
    assert!(d.path().join("a.mkv").exists());
    assert_eq!(fs::read(d.path().join("a-720.mp4")).unwrap(), b"mine");
    assert!(d.path().join("a-720-1.mp4").exists());
}

#[test]
fn existing_mp4_with_same_stem_is_not_overwritten_by_mkv_replace() {
    require_ffmpeg!();
    let d = tempfile::tempdir().unwrap();
    make_clip(&d.path().join("b.mkv"), "320x240", 1);
    fs::write(d.path().join("b.mp4"), b"other").unwrap();
    let o = vidconv(d.path(), &["b.mkv"]);
    assert!(o.status.success(), "{}", text(&o.stderr));
    assert_eq!(fs::read(d.path().join("b.mp4")).unwrap(), b"other");
    assert!(d.path().join("b-720.mp4").exists());
}

#[test]
fn odd_dimensions_become_even_without_padding() {
    require_ffmpeg!();
    let d = tempfile::tempdir().unwrap();
    // ffv1 allows odd sizes (libx264 yuv420p does not).
    ffmpeg(&[
        "-f",
        "lavfi",
        "-i",
        "testsrc2=s=641x361:d=1:r=30",
        "-pix_fmt",
        "yuv444p",
        "-c:v",
        "ffv1",
        d.path().join("odd.mkv").to_str().unwrap(),
    ]);
    let o = vidconv(d.path(), &["odd.mkv"]);
    assert!(o.status.success(), "{}", text(&o.stderr));
    let p = probe(&d.path().join("odd.mp4"));
    assert_eq!(
        (video(&p)["width"].as_u64(), video(&p)["height"].as_u64()),
        (Some(640), Some(360))
    );
}

#[test]
fn rotated_source_becomes_portrait() {
    require_ffmpeg!();
    let d = tempfile::tempdir().unwrap();
    make_clip(&d.path().join("src.mp4"), "640x360", 1);
    ffmpeg(&[
        "-display_rotation",
        "90",
        "-i",
        d.path().join("src.mp4").to_str().unwrap(),
        "-c",
        "copy",
        d.path().join("rot.mkv").to_str().unwrap(),
    ]);
    fs::remove_file(d.path().join("src.mp4")).unwrap();
    let o = vidconv(d.path(), &["rot.mkv"]);
    assert!(o.status.success(), "{}", text(&o.stderr));
    let p = probe(&d.path().join("rot.mp4"));
    let (w, h) = (
        video(&p)["width"].as_u64().unwrap(),
        video(&p)["height"].as_u64().unwrap(),
    );
    assert!(h > w, "expected portrait output, got {w}x{h}");
}

fn ffmpeg_has_filters(names: &[&str]) -> bool {
    let out = Command::new("ffmpeg")
        .args(["-hide_banner", "-filters"])
        .output()
        .unwrap();
    let t = text(&out.stdout);
    names
        .iter()
        .all(|n| t.lines().any(|l| l.split_whitespace().nth(1) == Some(n)))
}

#[test]
fn hdr_source_is_tone_mapped_to_8bit_bt709() {
    require_ffmpeg!();
    let d = tempfile::tempdir().unwrap();
    // HEVC keeps the PQ/BT.2020 tags (ffv1 would silently drop them, hiding the HDR path).
    ffmpeg(&[
        "-f",
        "lavfi",
        "-i",
        "testsrc2=s=320x240:d=1:r=30",
        "-vf",
        "format=yuv420p10le,setparams=color_primaries=bt2020:color_trc=smpte2084:colorspace=bt2020nc",
        "-c:v",
        "libx265",
        "-preset",
        "ultrafast",
        "-x265-params",
        "log-level=error",
        d.path().join("hdr.mkv").to_str().unwrap(),
    ]);
    let src = probe(&d.path().join("hdr.mkv"));
    assert_eq!(
        video(&src)["color_transfer"],
        "smpte2084",
        "fixture must really be HDR"
    );
    let o = vidconv(d.path(), &["--force", "hdr.mkv"]);
    assert!(o.status.success(), "{}", text(&o.stderr));
    let p = probe(&d.path().join("hdr.mp4"));
    assert_eq!(video(&p)["pix_fmt"], "yuv420p");
    if ffmpeg_has_filters(&["zscale", "tonemap"]) {
        // Tone mapping really ran: the output is tagged SDR bt709 instead of inheriting PQ.
        assert_eq!(video(&p)["color_transfer"], "bt709");
        assert_eq!(video(&p)["color_primaries"], "bt709");
    } else {
        eprintln!("ffmpeg lacks zscale/tonemap: tone-mapping assertion skipped");
    }
}

#[test]
fn only_first_audio_and_no_subs_or_chapters() {
    require_ffmpeg!();
    let d = tempfile::tempdir().unwrap();
    ffmpeg(&[
        "-f",
        "lavfi",
        "-i",
        "testsrc2=s=320x240:d=1:r=30",
        "-f",
        "lavfi",
        "-i",
        "sine=d=1",
        "-f",
        "lavfi",
        "-i",
        "sine=f=800:d=1",
        "-map",
        "0",
        "-map",
        "1",
        "-map",
        "2",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "5",
        "-c:a",
        "aac",
        d.path().join("multi.mkv").to_str().unwrap(),
    ]);
    let o = vidconv(d.path(), &["multi.mkv"]);
    assert!(o.status.success(), "{}", text(&o.stderr));
    let p = probe(&d.path().join("multi.mp4"));
    let kinds: Vec<_> = p["streams"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["codec_type"].as_str().unwrap())
        .collect();
    assert_eq!(kinds, ["video", "audio"]);
}

#[test]
fn awkward_file_names_work() {
    require_ffmpeg!();
    let d = tempfile::tempdir().unwrap();
    for n in ["-dash [x].mkv", "ünï [b].mkv"] {
        make_clip(&d.path().join(n), "320x240", 1);
    }
    let o = vidconv(
        d.path(),
        &[
            "-b",
            "1000",
            "-c",
            "1000",
            "--",
            "-dash [x].mkv",
            "ünï [b].mkv",
        ],
    );
    assert!(o.status.success(), "{}", text(&o.stderr));
    assert!(d.path().join("-dash [x].mp4").exists() && d.path().join("ünï [b].mp4").exists());
}

#[test]
fn symlink_is_left_alone() {
    require_ffmpeg!();
    let d = tempfile::tempdir().unwrap();
    make_clip(&d.path().join("real.mkv"), "320x240", 1);
    std::os::unix::fs::symlink("real.mkv", d.path().join("link.mkv")).unwrap();
    let o = vidconv(d.path(), &["link.mkv"]);
    assert!(o.status.success(), "{}", text(&o.stderr));
    assert!(
        fs::symlink_metadata(d.path().join("link.mkv"))
            .unwrap()
            .file_type()
            .is_symlink()
    );
    assert!(d.path().join("real.mkv").exists());
    assert!(d.path().join("link-720.mp4").exists());
}

#[test]
fn corrupt_input_fails_with_log_and_exit_code_1() {
    require_ffmpeg!();
    let d = tempfile::tempdir().unwrap();
    fs::write(d.path().join("bad.mkv"), b"this is not a video").unwrap();
    let o = vidconv(d.path(), &["bad.mkv"]);
    assert_eq!(o.status.code(), Some(1));
    assert!(text(&o.stderr).contains("bad.mkv"));
    assert!(d.path().join("vidconv_errors.log").exists());
    assert!(leftovers(d.path()).is_empty());
}

#[test]
fn directory_scan_ignores_non_video_and_reports_empty() {
    require_ffmpeg!();
    let d = tempfile::tempdir().unwrap();
    fs::write(d.path().join("notes.txt"), b"hi").unwrap();
    let o = Command::new(env!("CARGO_BIN_EXE_vidconv"))
        .current_dir(d.path())
        .args(["--no-hw", "--all"])
        .output()
        .unwrap();
    assert!(o.status.success());
    assert!(text(&o.stdout).contains("No video files"));
    // explicit directory argument behaves the same (used to be silently ignored)
    let o = Command::new(env!("CARGO_BIN_EXE_vidconv"))
        .current_dir(d.path())
        .args(["--no-hw", "."])
        .output()
        .unwrap();
    assert!(text(&o.stdout).contains("No video files"));
}

#[test]
fn output_dir_keeps_originals() {
    require_ffmpeg!();
    let d = tempfile::tempdir().unwrap();
    make_clip(&d.path().join("c.mkv"), "320x240", 1);
    let o = vidconv(d.path(), &["--output-dir", "out", "c.mkv"]);
    assert!(o.status.success(), "{}", text(&o.stderr));
    assert!(d.path().join("c.mkv").exists() && d.path().join("out/c.mp4").exists());
}

#[test]
fn sigint_cleans_up_and_exits_130() {
    require_ffmpeg!();
    let d = tempfile::tempdir().unwrap();
    make_clip(&d.path().join("long.mkv"), "640x360", 20);
    let mut child = Command::new(env!("CARGO_BIN_EXE_vidconv"))
        .current_dir(d.path())
        .args(["--no-hw", "--preset", "veryslow", "--quiet", "long.mkv"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let start = Instant::now();
    while leftovers(d.path()).is_empty() {
        assert!(
            start.elapsed() < Duration::from_secs(10),
            "temp output never appeared"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
    std::thread::sleep(Duration::from_millis(700));
    unsafe { libc::kill(child.id() as i32, libc::SIGINT) };
    let status = child.wait().unwrap();
    assert_eq!(status.code(), Some(130));
    assert!(d.path().join("long.mkv").exists(), "original must survive");
    assert!(
        leftovers(d.path()).is_empty(),
        "temp output must be removed"
    );
}

#[test]
fn usage_errors_and_completions() {
    let d = tempfile::tempdir().unwrap();
    let o = Command::new(env!("CARGO_BIN_EXE_vidconv"))
        .current_dir(d.path())
        .output()
        .unwrap();
    assert_eq!(o.status.code(), Some(2));
    let o = Command::new(env!("CARGO_BIN_EXE_vidconv"))
        .args(["--completions", "bash"])
        .output()
        .unwrap();
    assert!(o.status.success() && text(&o.stdout).contains("vidconv"));
    let o = Command::new(env!("CARGO_BIN_EXE_vidconv"))
        .args(["-b", "50", "x"])
        .output()
        .unwrap();
    assert_eq!(o.status.code(), Some(2));
}

fn json_lines(o: &Output) -> Vec<Value> {
    text(&o.stdout)
        .lines()
        .map(|l| serde_json::from_str(l).expect("every stdout line is JSON"))
        .collect()
}

fn make_lossless(path: &Path, size: &str, secs: u32) {
    ffmpeg(&[
        "-f",
        "lavfi",
        "-i",
        &format!("testsrc2=s={size}:d={secs}:r=30"),
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "0",
        path.to_str().unwrap(),
    ]);
}

fn make_tiny_bitrate(path: &Path) {
    ffmpeg(&[
        "-f",
        "lavfi",
        "-i",
        "color=c=gray:s=320x240:d=2:r=10",
        "-c:v",
        "libx264",
        "-preset",
        "veryslow",
        "-crf",
        "51",
        path.to_str().unwrap(),
    ]);
}

#[test]
fn source_without_audio() {
    require_ffmpeg!();
    let d = tempfile::tempdir().unwrap();
    make_lossless(&d.path().join("mute.mkv"), "320x240", 1);
    let o = vidconv(d.path(), &["mute.mkv"]);
    assert!(o.status.success(), "{}", text(&o.stderr));
    let p = probe(&d.path().join("mute.mp4"));
    assert!(
        p["streams"]
            .as_array()
            .unwrap()
            .iter()
            .all(|s| s["codec_type"] == "video")
    );
}

#[test]
fn text_subtitles_are_dropped() {
    require_ffmpeg!();
    let d = tempfile::tempdir().unwrap();
    fs::write(
        d.path().join("s.srt"),
        "1\n00:00:00,000 --> 00:00:01,000\nhello\n",
    )
    .unwrap();
    ffmpeg(&[
        "-f",
        "lavfi",
        "-i",
        "testsrc2=s=320x240:d=2:r=30",
        "-i",
        d.path().join("s.srt").to_str().unwrap(),
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "5",
        "-c:s",
        "srt",
        d.path().join("subs.mkv").to_str().unwrap(),
    ]);
    let before = probe(&d.path().join("subs.mkv"));
    assert!(
        before["streams"]
            .as_array()
            .unwrap()
            .iter()
            .any(|s| s["codec_type"] == "subtitle")
    );
    let o = vidconv(d.path(), &["subs.mkv"]);
    assert!(o.status.success(), "{}", text(&o.stderr));
    let p = probe(&d.path().join("subs.mp4"));
    assert!(
        p["streams"]
            .as_array()
            .unwrap()
            .iter()
            .all(|s| s["codec_type"] == "video")
    );
}

#[test]
fn single_frame_tiny_file_with_unknown_frame_rate() {
    require_ffmpeg!();
    let d = tempfile::tempdir().unwrap();
    ffmpeg(&[
        "-f",
        "lavfi",
        "-i",
        "testsrc2=s=32x32:r=30",
        "-frames:v",
        "1",
        "-c:v",
        "libx264",
        d.path().join("one.mkv").to_str().unwrap(),
    ]);
    // a one-frame mkv has no average frame rate (0/0) -- must not crash or vanish
    let o = vidconv(d.path(), &["--force", "one.mkv"]);
    assert!(o.status.success(), "{}", text(&o.stderr));
    let p = probe(&d.path().join("one.mp4"));
    assert_eq!(
        (video(&p)["width"].as_u64(), video(&p)["height"].as_u64()),
        (Some(32), Some(32))
    );
}

#[test]
fn result_not_smaller_keeps_original_unless_forced() {
    require_ffmpeg!();
    let d = tempfile::tempdir().unwrap();
    make_tiny_bitrate(&d.path().join("lo.mkv"));
    let o = vidconv(d.path(), &["--json", "lo.mkv"]);
    assert!(o.status.success(), "{}", text(&o.stderr));
    let ev = json_lines(&o);
    assert_eq!(ev[0]["status"], "skipped");
    assert!(ev[0]["reason"].as_str().unwrap().contains("no size gain"));
    assert!(d.path().join("lo.mkv").exists() && !d.path().join("lo.mp4").exists());
    assert!(leftovers(d.path()).is_empty());

    let o = vidconv(d.path(), &["--force", "lo.mkv"]);
    assert!(o.status.success(), "{}", text(&o.stderr));
    assert!(!d.path().join("lo.mkv").exists() && d.path().join("lo.mp4").exists());
}

#[test]
fn multi_mode_skips_low_bitrate_and_already_converted_files() {
    require_ffmpeg!();
    let d = tempfile::tempdir().unwrap();
    make_lossless(&d.path().join("hi.mkv"), "640x360", 1);
    make_tiny_bitrate(&d.path().join("lo.mkv"));
    let o = vidconv(d.path(), &["--all"]);
    assert!(o.status.success(), "{}", text(&o.stderr));
    assert!(d.path().join("hi.mp4").exists() && !d.path().join("hi.mkv").exists());
    assert!(
        d.path().join("lo.mkv").exists(),
        "below the cutoff: untouched"
    );
    // second run: hi.mp4 is tagged, lo.mkv still below cutoff -> nothing to do
    let o = Command::new(env!("CARGO_BIN_EXE_vidconv"))
        .current_dir(d.path())
        .args(["--no-hw", "--all"])
        .output()
        .unwrap();
    assert!(o.status.success());
    assert!(
        text(&o.stdout).contains("No video files"),
        "{}",
        text(&o.stdout)
    );
}

#[test]
fn recursion_is_opt_in() {
    require_ffmpeg!();
    let d = tempfile::tempdir().unwrap();
    fs::create_dir(d.path().join("sub")).unwrap();
    make_lossless(&d.path().join("sub/x.mkv"), "640x360", 1);
    let o = vidconv(d.path(), &["."]);
    assert!(o.status.success());
    assert!(
        d.path().join("sub/x.mkv").exists(),
        "not recursive by default"
    );
    let o = vidconv(d.path(), &["-r", "."]);
    assert!(o.status.success(), "{}", text(&o.stderr));
    assert!(d.path().join("sub/x.mp4").exists());
}

#[test]
fn json_output_is_machine_readable() {
    require_ffmpeg!();
    let d = tempfile::tempdir().unwrap();
    make_lossless(&d.path().join("good.mkv"), "640x360", 1);
    fs::write(d.path().join("bad.mkv"), b"nope").unwrap();
    let o = vidconv(d.path(), &["--json", "good.mkv", "bad.mkv"]);
    assert_eq!(o.status.code(), Some(1));
    let ev = json_lines(&o);
    let summary = ev.last().unwrap();
    assert_eq!(summary["event"], "summary");
    assert_eq!(
        (summary["converted"].as_u64(), summary["failed"].as_u64()),
        (Some(1), Some(1))
    );
    assert!(summary["saved_bytes"].as_i64().unwrap() > 0);
    assert!(
        ev.iter().any(
            |e| e["status"] == "success" && e["output"].as_str().unwrap().ends_with("good.mp4")
        )
    );
    assert!(
        ev.iter()
            .any(|e| e["status"] == "failed" && e["path"].as_str().unwrap().ends_with("bad.mkv"))
    );
}

#[test]
fn read_only_directory_fails_cleanly() {
    require_ffmpeg!();
    use std::os::unix::fs::PermissionsExt;
    let d = tempfile::tempdir().unwrap();
    make_lossless(&d.path().join("ro.mkv"), "320x240", 1);
    fs::set_permissions(d.path(), fs::Permissions::from_mode(0o555)).unwrap();
    let writable = fs::File::create(d.path().join("probe")).is_ok();
    if writable {
        // running as root: permissions are not enforced, nothing to test
        fs::set_permissions(d.path(), fs::Permissions::from_mode(0o755)).unwrap();
        return;
    }
    let log_dir = tempfile::tempdir().unwrap();
    let o = Command::new(env!("CARGO_BIN_EXE_vidconv"))
        .current_dir(log_dir.path())
        .args(["--no-hw", "--quiet"])
        .arg(d.path().join("ro.mkv"))
        .env("XDG_STATE_HOME", log_dir.path())
        .output()
        .unwrap();
    fs::set_permissions(d.path(), fs::Permissions::from_mode(0o755)).unwrap();
    assert_eq!(o.status.code(), Some(1));
    assert!(
        text(&o.stderr).contains("cannot write"),
        "{}",
        text(&o.stderr)
    );
    assert!(d.path().join("ro.mkv").exists());
    assert!(leftovers(d.path()).is_empty());
}
