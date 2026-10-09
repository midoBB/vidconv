//! ffmpeg argument construction (pure) and a single process runner for sw and hw encodes.
use crate::plan::Geometry;
use crate::probe::VideoInfo;
use crate::tools::arg_path;
use std::ffi::OsString;
use std::io::{BufRead, BufReader};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::{Duration, Instant};

pub const STALL_TIMEOUT: Duration = Duration::from_secs(180);

#[derive(Debug, Clone)]
pub struct EncodeConfig {
    pub bitrate: u32,
    pub crf: u8,
    pub preset: String,
    pub audio_bitrate: u32,
    pub fragmented: bool,
}

#[derive(Debug, Clone)]
pub enum Kind {
    Software,
    Vaapi(PathBuf),
}

pub struct Job<'a> {
    pub input: &'a Path,
    pub output: &'a Path,
    pub info: &'a VideoInfo,
    pub geom: &'a Geometry,
    /// Complete `-vf` chain for the CPU part (scale/sharpen/tonemap).
    pub filters: &'a str,
    pub tonemap: bool,
}

fn push<S: Into<OsString>>(v: &mut Vec<OsString>, items: impl IntoIterator<Item = S>) {
    v.extend(items.into_iter().map(Into::into));
}

pub fn build_args(kind: &Kind, cfg: &EncodeConfig, job: &Job) -> Vec<OsString> {
    let mut a: Vec<OsString> = Vec::new();
    push(
        &mut a,
        [
            "-hide_banner",
            "-v",
            "error",
            "-nostdin",
            "-y",
            "-progress",
            "pipe:1",
            "-nostats",
        ],
    );
    if let Kind::Vaapi(dev) = kind {
        push(
            &mut a,
            [
                "-init_hw_device".to_string(),
                format!("vaapi=card:{}", dev.display()),
            ],
        );
        push(
            &mut a,
            [
                "-hwaccel",
                "vaapi",
                "-hwaccel_output_format",
                "vaapi",
                "-hwaccel_device",
                "card",
            ],
        );
    }
    a.push("-i".into());
    a.push(arg_path(job.input));
    if matches!(kind, Kind::Vaapi(_)) {
        push(&mut a, ["-filter_hw_device", "card"]);
    }
    push(
        &mut a,
        [
            "-map".to_string(),
            format!("0:{}", job.info.video_index),
            "-map".into(),
            "0:a:0?".into(),
        ],
    );
    push(&mut a, ["-sn", "-dn", "-map_chapters", "-1"]);
    let vf = match kind {
        Kind::Software => job.filters.to_string(),
        Kind::Vaapi(_) => format!(
            "hwdownload,format=nv12,{},hwupload,scale_vaapi=format=nv12",
            job.filters
        ),
    };
    push(
        &mut a,
        ["-vf".to_string(), vf, "-r".into(), job.geom.fps.to_string()],
    );
    let movflags = if cfg.fragmented {
        "frag_keyframe+empty_moov+use_metadata_tags"
    } else {
        "faststart+use_metadata_tags"
    };
    push(&mut a, ["-movflags", movflags]);
    let (b, maxrate, bufsize) = (cfg.bitrate, cfg.bitrate * 2, cfg.bitrate * 4);
    match kind {
        Kind::Vaapi(_) => push(
            &mut a,
            [
                "-c:v".to_string(),
                "h264_vaapi".into(),
                "-b:v".into(),
                format!("{b}K"),
                "-maxrate".into(),
                format!("{maxrate}K"),
                "-bufsize".into(),
                format!("{bufsize}K"),
                "-quality".into(),
                "4".into(),
                "-compression_level".into(),
                "0".into(),
                "-g".into(),
                "24".into(),
                "-keyint_min".into(),
                "1".into(),
            ],
        ),
        Kind::Software => {
            push(
                &mut a,
                [
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-profile:v",
                    "high",
                ],
            );
            if job.geom.height <= 1080 {
                push(&mut a, ["-level", "4.1"]);
            }
            push(
                &mut a,
                [
                    "-preset".to_string(),
                    cfg.preset.clone(),
                    "-crf".into(),
                    cfg.crf.to_string(),
                    "-maxrate".into(),
                    format!("{maxrate}K"),
                    "-bufsize".into(),
                    format!("{bufsize}K"),
                    "-g".into(),
                    "24".into(),
                    "-keyint_min".into(),
                    "1".into(),
                    "-sc_threshold".into(),
                    "40".into(),
                    "-x264-params".into(),
                    "aq-mode=2:aq-strength=0.8:rc-lookahead=60:me=umh:subme=10:trellis=2".into(),
                ],
            );
            if job.tonemap {
                push(
                    &mut a,
                    [
                        "-colorspace",
                        "bt709",
                        "-color_primaries",
                        "bt709",
                        "-color_trc",
                        "bt709",
                    ],
                );
            }
        }
    }
    push(&mut a, ["-map_metadata".to_string(), "-1".into()]);
    let title = job
        .input
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    push(
        &mut a,
        [
            "-metadata".to_string(),
            format!("title={title}"),
            "-metadata".into(),
            "vidconv=1".into(),
        ],
    );
    if let Some(ct) = &job.info.creation_time {
        push(
            &mut a,
            ["-metadata".to_string(), format!("creation_time={ct}")],
        );
    }
    if job.info.has_audio {
        push(
            &mut a,
            [
                "-c:a".to_string(),
                "aac".into(),
                "-b:a".into(),
                format!("{}k", cfg.audio_bitrate),
            ],
        );
        if job.info.audio_channels > 2 {
            push(&mut a, ["-ac", "2"]);
        }
    }
    a.push(job.output.as_os_str().to_owned());
    a
}

#[derive(Debug, PartialEq)]
pub enum RunOutcome {
    Ok,
    Failed { stderr: String },
    Stalled,
    Interrupted,
}

fn terminate(child: &mut std::process::Child) {
    // SAFETY: plain signal send to our own child.
    unsafe { libc::kill(child.id() as i32, libc::SIGTERM) };
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(3) {
        if matches!(child.try_wait(), Ok(Some(_))) {
            return;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    let _ = child.kill();
    let _ = child.wait();
}

/// One progress sample from ffmpeg's `-progress` stream.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Progress {
    /// Encoded media time in seconds.
    pub secs: f64,
    /// Encode speed relative to realtime (`2.3` = 2.3x).
    pub speed: Option<f64>,
    pub fps: Option<f64>,
    /// Output bytes written so far.
    pub bytes: Option<u64>,
}

/// Run ffmpeg, reporting progress samples through `on_progress`.
pub fn run(
    ffmpeg: &Path,
    args: &[OsString],
    stop: &AtomicBool,
    on_progress: &mut dyn FnMut(Progress),
) -> RunOutcome {
    let spawned = Command::new(ffmpeg)
        .args(args)
        .process_group(0) // terminal Ctrl-C must not hit ffmpeg directly; we shut it down ourselves
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn();
    let mut child = match spawned {
        Ok(c) => c,
        Err(e) => {
            return RunOutcome::Failed {
                stderr: format!("cannot start ffmpeg: {e}"),
            };
        }
    };
    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();
    let (tx, rx) = mpsc::channel::<Progress>();
    let out_thread = std::thread::spawn(move || {
        let mut cur = Progress::default();
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            let Some((key, val)) = line.split_once('=') else {
                continue;
            };
            let val = val.trim();
            match key {
                "fps" => cur.fps = val.parse().ok(),
                "total_size" => cur.bytes = val.parse().ok(),
                "speed" => cur.speed = val.trim_end_matches('x').trim().parse().ok(),
                "out_time_us" => {
                    if let Ok(us) = val.parse::<i64>() {
                        cur.secs = us.max(0) as f64 / 1e6;
                        if tx.send(cur).is_err() {
                            break;
                        }
                    }
                }
                _ => {}
            }
        }
    });
    let err_thread = std::thread::spawn(move || {
        let mut lines: Vec<String> = Vec::new();
        for l in BufReader::new(stderr).lines().map_while(Result::ok) {
            lines.push(l);
            if lines.len() > 200 {
                lines.remove(0);
            }
        }
        lines.join("\n")
    });

    let mut last = Instant::now();
    let early = loop {
        if stop.load(Ordering::Relaxed) {
            terminate(&mut child);
            break Some(RunOutcome::Interrupted);
        }
        match rx.recv_timeout(Duration::from_millis(200)) {
            Ok(t) => {
                last = Instant::now();
                on_progress(t);
            }
            Err(RecvTimeoutError::Timeout) if last.elapsed() > STALL_TIMEOUT => {
                terminate(&mut child);
                break Some(RunOutcome::Stalled);
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => break None,
        }
    };
    let status = child.wait();
    let _ = out_thread.join();
    let stderr = err_thread.join().unwrap_or_default();
    match (early, status) {
        (Some(o), _) => o,
        (None, Ok(s)) if s.success() => RunOutcome::Ok,
        (None, Ok(s)) => RunOutcome::Failed {
            stderr: format!("ffmpeg exited with {s}\n{stderr}"),
        },
        (None, Err(e)) => RunOutcome::Failed {
            stderr: format!("wait failed: {e}\n{stderr}"),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> EncodeConfig {
        EncodeConfig {
            bitrate: 2500,
            crf: 20,
            preset: "slow".into(),
            audio_bitrate: 64,
            fragmented: false,
        }
    }

    fn info() -> VideoInfo {
        VideoInfo {
            codec: "hevc".into(),
            width: 1920,
            height: 1080,
            sar: 1.0,
            rotation: 0,
            fps: 30.0,
            bitrate_kbps: 5000,
            duration: 10.0,
            pix_fmt: "yuv420p".into(),
            color_transfer: String::new(),
            has_audio: true,
            audio_channels: 6,
            video_index: 1,
            creation_time: Some("2020-01-02T03:04:05Z".into()),
            tagged: false,
        }
    }

    fn strs(a: Vec<OsString>) -> Vec<String> {
        a.into_iter()
            .map(|s| s.to_string_lossy().into_owned())
            .collect()
    }

    fn args(kind: Kind, cfg: EncodeConfig, info: VideoInfo) -> Vec<String> {
        let g = Geometry {
            width: 1280,
            height: 720,
            fps: 24.0,
        };
        let job = Job {
            input: Path::new("in put.mkv"),
            output: Path::new("out.mp4"),
            info: &info,
            geom: &g,
            filters: "scale=w=1280:h=720:flags=lanczos",
            tonemap: false,
        };
        strs(build_args(&kind, &cfg, &job))
    }

    fn has_seq(a: &[String], seq: &[&str]) -> bool {
        a.windows(seq.len())
            .any(|w| w.iter().zip(seq).all(|(x, y)| x == y))
    }

    #[test]
    fn software_args() {
        let a = args(Kind::Software, cfg(), info());
        assert!(has_seq(
            &a,
            &[
                "-i",
                "in put.mkv",
                "-map",
                "0:1",
                "-map",
                "0:a:0?",
                "-sn",
                "-dn",
                "-map_chapters",
                "-1"
            ]
        ));
        assert!(has_seq(
            &a,
            &[
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-profile:v",
                "high",
                "-level",
                "4.1"
            ]
        ));
        assert!(has_seq(
            &a,
            &["-crf", "20", "-maxrate", "5000K", "-bufsize", "10000K"]
        ));
        assert!(has_seq(&a, &["-movflags", "faststart+use_metadata_tags"]));
        assert!(has_seq(
            &a,
            &["-metadata", "title=in put", "-metadata", "vidconv=1"]
        ));
        assert!(has_seq(
            &a,
            &["-metadata", "creation_time=2020-01-02T03:04:05Z"]
        ));
        assert!(has_seq(&a, &["-c:a", "aac", "-b:a", "64k", "-ac", "2"]));
        assert_eq!(a.last().unwrap(), "out.mp4");
        assert!(!a.iter().any(|x| x == "h264_vaapi"));
    }

    #[test]
    fn vaapi_args() {
        let a = args(Kind::Vaapi("/dev/dri/renderD128".into()), cfg(), info());
        assert!(has_seq(
            &a,
            &[
                "-init_hw_device",
                "vaapi=card:/dev/dri/renderD128",
                "-hwaccel",
                "vaapi"
            ]
        ));
        assert!(has_seq(
            &a,
            &[
                "-c:v",
                "h264_vaapi",
                "-b:v",
                "2500K",
                "-maxrate",
                "5000K",
                "-bufsize",
                "10000K"
            ]
        ));
        let vf = &a[a.iter().position(|x| x == "-vf").unwrap() + 1];
        assert!(
            vf.starts_with("hwdownload,format=nv12,scale=")
                && vf.ends_with(",hwupload,scale_vaapi=format=nv12")
        );
    }

    #[test]
    fn fragmented_and_no_audio() {
        let mut i = info();
        i.has_audio = false;
        let mut c = cfg();
        c.fragmented = true;
        let a = args(Kind::Software, c, i);
        assert!(has_seq(
            &a,
            &["-movflags", "frag_keyframe+empty_moov+use_metadata_tags"]
        ));
        assert!(!a.iter().any(|x| x == "-c:a"));
    }

    #[test]
    fn runner_reports_failure_and_interrupt() {
        let stop = AtomicBool::new(false);
        let r = run(
            Path::new("/bin/sh"),
            &["-c".into(), "echo boom >&2; exit 3".into()],
            &stop,
            &mut |_| {},
        );
        assert!(matches!(r, RunOutcome::Failed { ref stderr } if stderr.contains("boom")));
        let r = run(
            Path::new("/bin/sh"),
            &["-c".into(), "echo out_time_us=2000000; exit 0".into()],
            &stop,
            &mut |p| assert_eq!(p.secs, 2.0),
        );
        assert_eq!(r, RunOutcome::Ok);
        let mut got = None;
        let r = run(
            Path::new("/bin/sh"),
            &[
                "-c".into(),
                "echo fps=60.0; echo total_size=1234; echo speed=2.50x; echo out_time_us=3000000"
                    .into(),
            ],
            &stop,
            &mut |p| got = Some(p),
        );
        assert_eq!(r, RunOutcome::Ok);
        assert_eq!(
            got,
            Some(Progress {
                secs: 3.0,
                speed: Some(2.5),
                fps: Some(60.0),
                bytes: Some(1234)
            })
        );
        // `speed=N/A` must not poison later samples.
        let mut got = None;
        run(
            Path::new("/bin/sh"),
            &[
                "-c".into(),
                "echo speed=N/A; echo out_time_us=1000000".into(),
            ],
            &stop,
            &mut |p| got = Some(p),
        );
        assert_eq!(got.map(|p| p.speed), Some(None));
        stop.store(true, Ordering::Relaxed);
        let r = run(
            Path::new("/bin/sh"),
            &["-c".into(), "sleep 30".into()],
            &stop,
            &mut |_| {},
        );
        assert_eq!(r, RunOutcome::Interrupted);
    }
}

/// Golden tests: argv captured from the original Python tool (via a logging ffmpeg shim) must equal
/// the Rust argv once the *documented* differences are normalised away.
#[cfg(test)]
mod golden {
    use super::*;
    use crate::plan::{self, PlanOpts};

    /// Strip `scale` dimensions and the Python-only pad/force_original_aspect_ratio parts.
    fn norm_vf(vf: &str) -> String {
        let vf = vf
            .replace(":force_original_aspect_ratio=decrease", "")
            .replace(",pad=ceil(iw/2)*2:ceil(ih/2)*2", "");
        let (mut out, mut rest) = (String::new(), vf.as_str());
        while let Some(i) = rest.find("scale=w=") {
            let (head, tail) = rest.split_at(i + "scale=w=".len());
            out.push_str(head);
            out.push('#');
            let tail = tail.trim_start_matches(|c: char| c.is_ascii_digit());
            let tail = tail.strip_prefix(":h=").expect("scale=w=N:h=M");
            out.push_str(":h=#");
            rest = tail.trim_start_matches(|c: char| c.is_ascii_digit());
        }
        out.push_str(rest);
        out
    }

    /// Drop `n` args starting at the first occurrence of `seq`.
    fn drop_seq(a: &mut Vec<String>, seq: &[&str]) {
        if let Some(i) = a
            .windows(seq.len())
            .position(|w| w.iter().zip(seq).all(|(x, y)| x == y))
        {
            a.drain(i..i + seq.len());
        }
    }

    fn common(a: &mut Vec<String>) {
        if let Some(i) = a.iter().position(|x| x == "-vf") {
            a[i + 1] = norm_vf(&a[i + 1]);
        }
        // Position and value of -movflags differ on purpose (faststart); the value is unit-tested above.
        if let Some(i) = a.iter().position(|x| x == "-movflags") {
            a.drain(i..i + 2);
        }
        *a.last_mut().unwrap() = "OUT".into();
    }

    fn python(raw: &str, has_audio: bool) -> Vec<String> {
        let mut a: Vec<String> = raw.lines().map(String::from).collect();
        drop_seq(&mut a, &["-stats"]);
        if !has_audio {
            drop_seq(&mut a, &["-c:a", "aac", "-b:a", "64k"]); // Python adds audio args even with no audio
        }
        common(&mut a);
        a
    }

    fn rust(kind: Kind, info: &VideoInfo, input: &str) -> Vec<String> {
        let o = PlanOpts::default();
        let g = plan::geometry(info, &o);
        let filters = plan::video_filters(info, &g, &o, false);
        let cfg = EncodeConfig {
            bitrate: 2500,
            crf: 20,
            preset: "slow".into(),
            audio_bitrate: 64,
            fragmented: false,
        };
        let job = Job {
            input: Path::new(input),
            output: Path::new("x.tmp"),
            info,
            geom: &g,
            filters: &filters,
            tonemap: false,
        };
        let mut a: Vec<String> = build_args(&kind, &cfg, &job)
            .into_iter()
            .map(|s| s.to_string_lossy().into_owned())
            .collect();
        for seq in [
            &["-nostdin"][..],
            &["-y"],
            &["-progress", "pipe:1"],
            &["-nostats"],
            &["-sn"],
            &["-dn"],
            &["-map_chapters", "-1"],
            &["-map", "0:0"],
            &["-map", "0:a:0?"],
        ] {
            drop_seq(&mut a, seq);
        }
        common(&mut a);
        a
    }

    fn info(w: u32, h: u32, fps: f64, codec: &str, audio: bool) -> VideoInfo {
        VideoInfo {
            codec: codec.into(),
            width: w,
            height: h,
            sar: 1.0,
            rotation: 0,
            fps,
            bitrate_kbps: 20000,
            duration: 2.0,
            pix_fmt: "yuv420p".into(),
            color_transfer: String::new(),
            has_audio: audio,
            audio_channels: if audio { 1 } else { 0 },
            video_index: 0,
            creation_time: None,
            tagged: false,
        }
    }

    fn check(name: &str, raw: &str, kind: Kind, i: VideoInfo) {
        let (py, rs) = (
            python(raw, i.has_audio),
            rust(
                kind,
                &i,
                &format!("{}.mkv", name.split('-').next().unwrap()),
            ),
        );
        assert_eq!(
            py, rs,
            "{name}: Rust args diverge from the Python tool beyond the documented differences"
        );
    }

    macro_rules! case {
        ($fn:ident, $name:literal, $kind:expr, $info:expr) => {
            #[test]
            fn $fn() {
                check(
                    $name,
                    include_str!(concat!("../tests/golden/", $name, ".argv")),
                    $kind,
                    $info,
                );
            }
        };
    }

    case!(
        land1080_sw,
        "land1080-sw",
        Kind::Software,
        info(1920, 1080, 30.0, "h264", true)
    );
    case!(
        port_sw,
        "port-sw",
        Kind::Software,
        info(1080, 1920, 30.0, "h264", true)
    );
    case!(
        small25_sw,
        "small25-sw",
        Kind::Software,
        info(640, 360, 25.0, "h264", true)
    );
    case!(
        fps60_sw,
        "fps60-sw",
        Kind::Software,
        info(1280, 720, 60.0, "h264", true)
    );
    case!(
        noaudio_sw,
        "noaudio-sw",
        Kind::Software,
        info(1920, 1080, 30.0, "h264", false)
    );
    case!(
        hevc_hw,
        "hevc-hw",
        Kind::Vaapi("/dev/dri/renderD128".into()),
        info(1920, 1080, 30.0, "hevc", true)
    );

    /// The intentional geometry differences, pinned so they can't drift silently.
    #[test]
    fn documented_geometry_differences() {
        let o = PlanOpts::default();
        let dims = |i: &VideoInfo| {
            let g = plan::geometry(i, &o);
            (g.width, g.height)
        };
        assert_eq!(dims(&info(1080, 1920, 30.0, "h264", true)), (404, 720)); // Python: 406x720 (rounded up)
        assert_eq!(dims(&info(640, 360, 25.0, "h264", true)), (640, 360)); // Python: upscaled to 1280x720
    }
}
