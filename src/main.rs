mod cli;
mod discover;
mod encode;
mod errlog;
mod output;
mod plan;
mod probe;
mod process;
mod tools;
mod ui;

use clap::error::ErrorKind;
use clap::{CommandFactory, Parser};
use cli::{Cli, HwMode};
use console::style;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use ui::{Status, Ui, display_name, format_size};

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(e) => {
            eprintln!("Error: {e:#}");
            ExitCode::from(1)
        }
    }
}

fn run() -> anyhow::Result<ExitCode> {
    let cli = Cli::parse();
    if let Some(shell) = cli.completions {
        clap_complete::generate(
            shell,
            &mut Cli::command(),
            "vidconv",
            &mut std::io::stdout(),
        );
        return Ok(ExitCode::SUCCESS);
    }
    if !cli.all && cli.inputs.is_empty() {
        Cli::command()
            .error(
                ErrorKind::MissingRequiredArgument,
                "Must specify INPUT_PATH or use --all",
            )
            .exit();
    }
    if !(cli.max_fps.is_finite() && cli.max_fps > 0.0 && cli.max_fps <= 240.0) {
        Cli::command()
            .error(
                ErrorKind::ValueValidation,
                "--max-fps must be between 0 and 240",
            )
            .exit();
    }
    let tools = match tools::locate() {
        Ok(t) => t,
        Err(missing) => {
            eprintln!("Error: the following required tools are not installed:");
            for m in missing {
                eprintln!("  - {m}");
            }
            eprintln!("\nPlease install the missing tools and try again.");
            return Ok(ExitCode::from(1));
        }
    };

    // First Ctrl-C/SIGTERM asks for a clean stop; a second one exits immediately.
    let stop = Arc::new(AtomicBool::new(false));
    for sig in [signal_hook::consts::SIGINT, signal_hook::consts::SIGTERM] {
        signal_hook::flag::register_conditional_shutdown(sig, 130, Arc::clone(&stop))?;
        signal_hook::flag::register(sig, Arc::clone(&stop))?;
    }

    let mut log = errlog::ErrorLog::new();
    let cutoff = cli.effective_cutoff();
    let inputs: Vec<PathBuf> = if cli.inputs.is_empty() {
        vec![PathBuf::from(".")]
    } else {
        cli.inputs.clone()
    };
    let multi = cli.all || inputs.len() > 1 || inputs.iter().any(|p| p.is_dir());
    let want_hw = !cli.no_hw && cli.hw != HwMode::Off;
    let caps = tools::detect(&tools, want_hw, cli.vaapi_device.as_deref());

    if !cli.silent() {
        println!("{}", style("Configuration:").bold());
        let kv = |k: &str, v: String| println!("  {k}: {}", style(v).cyan());
        kv("Bitrate", format!("{} kbps", cli.bitrate));
        kv("Cutoff", format!("{cutoff} kbps"));
        kv(
            "Hardware Acceleration",
            match &caps.vaapi_device {
                Some(d) => format!("VAAPI ({})", d.display()),
                None if want_hw => "unavailable, using software".into(),
                None => "off".into(),
            },
        );
        kv(
            "Keep Originals",
            (cli.keep || cli.output_dir.is_some()).to_string(),
        );
        kv("Process All", multi.to_string());
        kv("Sort By", format!("{:?}", cli.sort_by).to_lowercase());
    }

    let found = discover::discover(
        &inputs,
        &discover::Opts {
            ffprobe: &tools.ffprobe,
            recursive: cli.recursive,
            multi,
            cutoff,
            sort: cli.sort_by,
        },
    );
    let mut errors = 0usize;
    for f in &found.failures {
        let msg = format!("{}: {}", f.path.display(), f.reason);
        log.log(&f.path, &f.reason);
        if cli.json {
            emit(
                serde_json::json!({"event": "file", "path": f.path.to_string_lossy(),
                "status": if f.explicit { "failed" } else { "ignored" }, "reason": f.reason}),
            );
        }
        if f.explicit {
            errors += 1;
            eprintln!("Cannot process {msg}");
        } else if !cli.silent() {
            eprintln!("Skipping {msg}");
        }
    }
    let files = found.files;
    if files.is_empty() {
        if !cli.silent() {
            println!("No video files to process.");
        }
        if cli.json {
            emit(summary_json(0, 0, errors, 0, false));
        }
        log.finish(errors > 0);
        return Ok(ExitCode::from(u8::from(errors > 0)));
    }
    if !cli.silent() {
        kv_files(files.len());
    }

    let ctx = process::Ctx {
        cli: &cli,
        tools: &tools,
        caps: &caps,
        stop: &stop,
        cutoff,
        multi,
        cfg: encode::EncodeConfig {
            bitrate: cli.bitrate,
            crf: cli.crf,
            preset: cli.preset.clone(),
            audio_bitrate: cli.audio_bitrate,
            fragmented: cli.fragmented,
        },
        plan: plan::PlanOpts {
            height: cli.height,
            max_fps: cli.max_fps,
            sharpen: !cli.no_sharpen,
        },
    };

    let names: Vec<String> = files.iter().map(|c| display_name(&c.path)).collect();
    let mut statuses: Vec<Option<Status>> = vec![None; files.len()];
    let (mut ok, mut skipped, mut saved_total, mut interrupted) = (0usize, 0usize, 0i64, false);
    let ui = Ui::new(files.len(), cli.silent());

    for (i, cand) in files.iter().enumerate() {
        if stop.load(Ordering::Relaxed) {
            interrupted = true;
            break;
        }
        ui.start_file(
            &names[i],
            ui::queue_lines(i, &names, &statuses),
            saved_total,
        );
        let done = catch_unwind(AssertUnwindSafe(|| {
            process::process_file(&ctx, cand, &ui, &mut log)
        }))
        .unwrap_or_else(|_| {
            log.log(
                &cand.path,
                "internal error (panic) while processing this file",
            );
            process::Done {
                outcome: process::Outcome::Failed("internal error".into()),
                saved: 0,
            }
        });
        saved_total += done.saved;
        match done.outcome {
            process::Outcome::Success(dest) => {
                ok += 1;
                statuses[i] = Some(Status::Success);
                ui.say(format!(
                    "Successfully processed: {} -> {}",
                    names[i],
                    display_name(&dest)
                ));
                if cli.json {
                    emit(
                        serde_json::json!({"event": "file", "path": cand.path.to_string_lossy(), "status": "success",
                        "output": dest.to_string_lossy(), "saved_bytes": done.saved}),
                    );
                }
            }
            process::Outcome::Skipped(reason) => {
                skipped += 1;
                statuses[i] = Some(Status::Skipped);
                ui.say(format!("Skipping {}: {reason}", names[i]));
                if cli.json {
                    emit(
                        serde_json::json!({"event": "file", "path": cand.path.to_string_lossy(), "status": "skipped",
                        "reason": reason}),
                    );
                }
            }
            process::Outcome::Failed(msg) => {
                errors += 1;
                statuses[i] = Some(Status::Failed);
                ui.error(format!("Failed to process: {}: {msg}", names[i]));
                if cli.json {
                    emit(
                        serde_json::json!({"event": "file", "path": cand.path.to_string_lossy(), "status": "failed",
                        "reason": msg}),
                    );
                }
            }
            process::Outcome::Interrupted => {
                interrupted = true;
                break;
            }
        }
        ui.finish_file();
    }
    ui.finish();

    if interrupted {
        eprintln!("\nInterrupted; temporary files removed.");
    }
    if !cli.silent() {
        println!("\nProcessed {ok} files successfully");
        if skipped > 0 {
            println!("Skipped {skipped} files");
        }
        println!("Total space saved: {}", format_size(saved_total));
    }
    if cli.json {
        emit(summary_json(ok, skipped, errors, saved_total, interrupted));
    }
    if errors > 0 {
        eprintln!(
            "Encountered {errors} errors - see {} for details",
            log.path().display()
        );
    }
    log.finish(errors > 0);
    Ok(if interrupted {
        ExitCode::from(130)
    } else {
        ExitCode::from(u8::from(errors > 0))
    })
}

fn summary_json(
    ok: usize,
    skipped: usize,
    failed: usize,
    saved: i64,
    interrupted: bool,
) -> serde_json::Value {
    serde_json::json!({"event": "summary", "converted": ok, "skipped": skipped, "failed": failed,
        "saved_bytes": saved, "interrupted": interrupted})
}

fn emit(v: serde_json::Value) {
    println!("{v}");
}

fn kv_files(n: usize) {
    println!("  Files to process: {}", style(n).cyan());
}
