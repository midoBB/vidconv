//! Per-file pipeline: checks → plan → encode (hw, then sw) → verify → finalize.
use crate::cli::{Cli, HwMode};
use crate::discover::{Candidate, below_cutoff};
use crate::encode::{self, EncodeConfig, Job, Kind, Progress, RunOutcome, build_args};
use crate::errlog::ErrorLog;
use crate::output::{TempOutput, free_space, stem_with, unique_path};
use crate::plan::{self, PlanOpts};
use crate::probe;
use crate::tools::{Caps, Tools};
use crate::ui::{Ui, display_name, format_size};
use std::fs;
use std::io::ErrorKind;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

#[derive(Debug)]
pub enum Outcome {
    Success(PathBuf),
    Skipped(String),
    Failed(String),
    Interrupted,
}

pub struct Done {
    pub outcome: Outcome,
    /// Original size minus new size (negative when the result grew).
    pub saved: i64,
}

pub struct Ctx<'a> {
    pub cli: &'a Cli,
    pub tools: &'a Tools,
    pub caps: &'a Caps,
    pub stop: &'a AtomicBool,
    pub cutoff: u32,
    pub multi: bool,
    pub cfg: EncodeConfig,
    pub plan: PlanOpts,
}

fn done(outcome: Outcome) -> Done {
    Done { outcome, saved: 0 }
}

/// Failures that a software retry cannot fix.
fn input_related(stderr: &str) -> bool {
    [
        "Invalid data found",
        "No such file",
        "moov atom not found",
        "does not contain any stream",
    ]
    .iter()
    .any(|m| stderr.contains(m))
}

/// Expected output size with 20% headroom; 0 when the duration is unknown.
pub fn estimate_output_bytes(video_kbps: u32, audio_kbps: u32, duration: f64) -> u64 {
    if duration <= 0.0 {
        return 0;
    }
    ((video_kbps + audio_kbps) as f64 * 1000.0 / 8.0 * duration * 1.2) as u64
}

pub fn has_room(free: u64, need: u64) -> bool {
    free >= need
}

fn same_file(a: &fs::Metadata, b: &fs::Metadata) -> bool {
    a.dev() == b.dev() && a.ino() == b.ino()
}

pub fn process_file(ctx: &Ctx, c: &Candidate, ui: &Ui, log: &mut ErrorLog) -> Done {
    let path = &c.path;
    let cli = ctx.cli;
    let mut fail = |msg: String, detail: Option<&str>| {
        log.log(path, detail.unwrap_or(&msg));
        done(Outcome::Failed(msg))
    };

    let orig_meta = match fs::metadata(path) {
        Ok(m) => m,
        Err(e) if e.kind() == ErrorKind::NotFound => {
            return done(Outcome::Skipped("file no longer exists".into()));
        }
        Err(e) => return done(Outcome::Skipped(format!("cannot access file ({e})"))),
    };
    if c.info.tagged {
        return done(Outcome::Skipped("already converted by vidconv".into()));
    }
    if ctx.multi && below_cutoff(&c.info, ctx.cutoff) {
        return done(Outcome::Skipped("bitrate below cutoff".into()));
    }

    // Symlinks are never replaced: the result is written beside them and the link is left alone.
    let is_symlink = fs::symlink_metadata(path)
        .map(|m| m.file_type().is_symlink())
        .unwrap_or(false);
    let keep = cli.keep || cli.output_dir.is_some() || is_symlink;
    let out_dir = match &cli.output_dir {
        Some(d) => {
            if let Err(e) = fs::create_dir_all(d) {
                return fail(
                    format!("cannot create output directory {}: {e}", d.display()),
                    None,
                );
            }
            d.clone()
        }
        None => path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."))
            .to_path_buf(),
    };

    let need = estimate_output_bytes(cli.bitrate, cli.audio_bitrate, c.info.duration);
    if let Some(free) = free_space(&out_dir).filter(|&free| !has_room(free, need)) {
        return fail(
            format!(
                "not enough disk space in {} (need ~{}, have {})",
                out_dir.display(),
                format_size(need as i64),
                format_size(free as i64)
            ),
            None,
        );
    }
    let mut tmp = match TempOutput::create(&out_dir) {
        Ok(t) => t,
        Err(e) => return fail(format!("cannot write to {}: {e}", out_dir.display()), None),
    };

    let geom = plan::geometry(&c.info, &ctx.plan);
    let hdr = c.info.is_hdr();
    let tonemap = hdr && ctx.caps.tonemap;
    if hdr && !tonemap {
        ui.say(format!(
            "Warning: {} is HDR but ffmpeg lacks zscale/tonemap; colors may look washed out",
            display_name(path)
        ));
    }
    let sw_filters = plan::video_filters(&c.info, &geom, &ctx.plan, tonemap);
    let hw_filters = plan::video_filters(&c.info, &geom, &ctx.plan, false);

    let hw_device = match (&ctx.caps.vaapi_device, cli.hw) {
        (Some(dev), mode) if mode != HwMode::Off => {
            let eligible = !hdr
                && !c.info.is_high_bit_depth()
                && c.info.rotation == 0
                && (mode == HwMode::Force
                    || ctx
                        .caps
                        .vaapi_decodes(&ctx.tools.ffmpeg, &c.info.codec, path));
            eligible.then(|| dev.clone())
        }
        _ => None,
    };

    let duration = c.info.duration;
    let mut progress = |p: Progress| ui.progress(&p, duration);
    let mut attempt = |kind: Kind, filters: &str, tmp: &TempOutput| {
        ui.stage(match kind {
            Kind::Vaapi(_) => "Encoding (VAAPI)",
            Kind::Software => "Encoding (software)",
        });
        let job = Job {
            input: path,
            output: tmp.path(),
            info: &c.info,
            geom: &geom,
            filters,
            tonemap,
        };
        let args = build_args(&kind, &ctx.cfg, &job);
        encode::run(&ctx.tools.ffmpeg, &args, ctx.stop, &mut progress)
    };

    let result = match hw_device {
        Some(dev) => match attempt(Kind::Vaapi(dev), &hw_filters, &tmp) {
            RunOutcome::Failed { stderr } if input_related(&stderr) => {
                RunOutcome::Failed { stderr }
            }
            RunOutcome::Failed { .. } | RunOutcome::Stalled => {
                ui.say("Hardware encoding failed, trying software...");
                attempt(Kind::Software, &sw_filters, &tmp)
            }
            other => other,
        },
        None => attempt(Kind::Software, &sw_filters, &tmp),
    };
    match result {
        RunOutcome::Ok => {}
        RunOutcome::Interrupted => return done(Outcome::Interrupted),
        RunOutcome::Stalled => {
            return fail(
                format!(
                    "ffmpeg made no progress for {}s",
                    encode::STALL_TIMEOUT.as_secs()
                ),
                None,
            );
        }
        RunOutcome::Failed { stderr } => return fail("ffmpeg failed".into(), Some(&stderr)),
    }

    // Verify the result before touching the original.
    ui.stage("Verifying");
    let out_info = match probe::probe(&ctx.tools.ffprobe, tmp.path()) {
        Ok(i) => i,
        Err(e) => return fail(format!("output verification failed: {e}"), None),
    };
    if c.info.duration > 0.0 && out_info.duration > 0.0 {
        let tol = (c.info.duration * 0.05).max(2.0);
        if (out_info.duration - c.info.duration).abs() > tol {
            return fail(
                format!(
                    "output duration {:.1}s does not match source {:.1}s",
                    out_info.duration, c.info.duration
                ),
                None,
            );
        }
    }
    let new_size = match tmp.size() {
        Ok(s) => s,
        Err(e) => return fail(format!("cannot stat output: {e}"), None),
    };
    if !keep && new_size >= c.size && !cli.force {
        return done(Outcome::Skipped(format!(
            "no size gain ({} -> {}); original kept (use --force to replace anyway)",
            format_size(c.size as i64),
            format_size(new_size as i64)
        )));
    }
    let saved = c.size as i64 - new_size as i64;

    ui.stage("Finalizing");
    tmp.copy_attrs_from(&orig_meta);
    let stem = path.file_stem().unwrap_or(path.as_os_str());
    let mut finalize = || -> std::io::Result<PathBuf> {
        let persist_free =
            |tmp: &mut TempOutput, base: &std::ffi::OsStr| -> std::io::Result<PathBuf> {
                loop {
                    let dest = unique_path(&out_dir, base, "mp4");
                    match tmp.persist_noclobber(&dest) {
                        Ok(()) => return Ok(dest),
                        Err(e) if e.kind() == ErrorKind::AlreadyExists => continue,
                        Err(e) => return Err(e),
                    }
                }
            };
        if keep {
            let base = if cli.output_dir.is_some() {
                stem.to_owned()
            } else {
                stem_with(stem, "-720")
            };
            return persist_free(&mut tmp, &base);
        }
        let want = out_dir.join(stem_with(stem, ".mp4"));
        // `replaced_original`: the rename already replaced the source, so it must not be removed.
        let (dest, replaced_original) = match fs::symlink_metadata(&want) {
            Err(_) => {
                tmp.persist_noclobber(&want)?;
                (want, false)
            }
            Ok(m) if same_file(&m, &orig_meta) => {
                tmp.persist_overwrite(&want)?; // atomic in-place replace (foo.mp4 -> foo.mp4)
                (want, true)
            }
            Ok(_) => (persist_free(&mut tmp, &stem_with(stem, "-720"))?, false), // someone else's foo.mp4 exists
        };
        if !replaced_original
            && let Err(e) = fs::remove_file(path)
        {
            ui.error(format!(
                "Warning: converted {} but could not remove the original: {e}",
                display_name(path)
            ));
        }
        Ok(dest)
    };
    match finalize() {
        Ok(dest) => Done {
            outcome: Outcome::Success(
                dest.strip_prefix("./")
                    .map(Path::to_path_buf)
                    .unwrap_or(dest),
            ),
            saved,
        },
        Err(e) => fail(format!("could not finalize: {e}"), None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_and_room() {
        // 2564 kbps for 100 s ≈ 32 MB, plus 20% headroom
        let need = estimate_output_bytes(2500, 64, 100.0);
        assert_eq!(need, 38_460_000);
        assert_eq!(estimate_output_bytes(2500, 64, 0.0), 0);
        assert!(has_room(need, need));
        assert!(!has_room(need - 1, need));
        assert!(has_room(0, 0));
    }

    #[test]
    fn stderr_classification() {
        assert!(input_related("x: Invalid data found when processing input"));
        assert!(!input_related("Impossible to convert between the formats"));
    }
}
