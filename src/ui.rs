//! Terminal UI: total/file progress bars plus a small conversion queue panel.
use console::style;
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use std::io::IsTerminal;
use std::path::Path;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Status {
    Success,
    Failed,
    Skipped,
}

/// Printable file name: lossy UTF-8 with control characters neutralised (no escape-sequence injection).
pub fn display_name(p: &Path) -> String {
    let name = p.file_name().unwrap_or(p.as_os_str());
    name.to_string_lossy()
        .chars()
        .map(|c| if c.is_control() { '?' } else { c })
        .collect()
}

pub fn format_size(bytes: i64) -> String {
    let mut v = bytes.unsigned_abs() as f64;
    let mut unit = "B";
    for u in ["KB", "MB", "GB", "TB"] {
        if v < 1024.0 {
            break;
        }
        v /= 1024.0;
        unit = u;
    }
    let s = if unit == "B" {
        format!("{v:.0} B")
    } else {
        format!("{v:.2} {unit}")
    };
    if bytes < 0 {
        format!("-{s} (increase)")
    } else {
        s
    }
}

/// Queue window: two before, current, two after.
pub fn queue_lines(current: usize, names: &[String], statuses: &[Option<Status>]) -> Vec<String> {
    let (start, end) = (current.saturating_sub(2), (current + 3).min(names.len()));
    (start..end)
        .map(|i| {
            let prefix = if i < current {
                match statuses.get(i).copied().flatten() {
                    Some(Status::Success) => style("✓").green().to_string(),
                    Some(Status::Failed) => style("✗").red().to_string(),
                    Some(Status::Skipped) => style("S").cyan().to_string(),
                    None => "?".to_string(),
                }
            } else if i == current {
                style("→").yellow().to_string()
            } else {
                " ".to_string()
            };
            format!("{prefix} {}", names[i])
        })
        .collect()
}

pub struct Ui {
    mp: MultiProgress,
    total: ProgressBar,
    file: ProgressBar,
    queue: ProgressBar,
    quiet: bool,
    interactive: bool,
}

impl Ui {
    pub fn new(total_files: usize, quiet: bool) -> Self {
        let interactive = !quiet && std::io::stderr().is_terminal();
        let mp = MultiProgress::with_draw_target(if interactive {
            ProgressDrawTarget::stderr_with_hz(8)
        } else {
            ProgressDrawTarget::hidden()
        });
        let total = mp.add(ProgressBar::new(total_files as u64));
        total.set_style(
            ProgressStyle::with_template(
                "{prefix:.cyan} [{bar:30.cyan/blue}] {pos}/{len} {elapsed_precise}",
            )
            .unwrap()
            .progress_chars("█░ "),
        );
        total.set_prefix("Processing videos...");
        let file = mp.add(ProgressBar::new(100));
        file.set_style(
            ProgressStyle::with_template(
                "{prefix:.bold.blue} [{bar:30}] {percent:>3}% {elapsed_precise} ETA {eta}",
            )
            .unwrap()
            .progress_chars("█░ "),
        );
        let queue = mp.add(ProgressBar::new(0));
        queue.set_style(ProgressStyle::with_template("{msg}").unwrap());
        Self {
            mp,
            total,
            file,
            queue,
            quiet,
            interactive,
        }
    }

    /// Informational line (suppressed by --quiet).
    pub fn say(&self, msg: impl AsRef<str>) {
        if self.quiet {
            return;
        }
        self.print(msg.as_ref());
    }

    /// Always shown.
    pub fn error(&self, msg: impl AsRef<str>) {
        if self.interactive {
            let _ = self.mp.println(msg.as_ref());
        } else {
            eprintln!("{}", msg.as_ref());
        }
    }

    fn print(&self, msg: &str) {
        if self.interactive {
            let _ = self.mp.println(msg);
        } else {
            println!("{msg}");
        }
    }

    pub fn start_file(&self, name: &str, queue: Vec<String>, saved: i64) {
        self.file.set_prefix(format!("Converting {name}"));
        self.file.reset();
        self.file.set_position(0);
        let mut text = vec![style("Conversion Queue").magenta().bold().to_string()];
        text.extend(queue);
        text.push(String::new());
        text.push(format!("Total space saved: {}", format_size(saved)));
        self.queue.set_message(text.join("\n"));
    }

    pub fn set_progress(&self, frac: f64) {
        self.file
            .set_position((frac.clamp(0.0, 1.0) * 100.0) as u64);
    }

    pub fn finish_file(&self) {
        self.total.inc(1);
    }

    pub fn finish(&self) {
        self.file.finish_and_clear();
        self.queue.finish_and_clear();
        self.total.finish_and_clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes() {
        assert_eq!(format_size(0), "0 B");
        assert_eq!(format_size(1536), "1.50 KB");
        assert_eq!(format_size(5 * 1024 * 1024 * 1024), "5.00 GB");
        assert_eq!(format_size(-2048), "-2.00 KB (increase)");
    }

    #[test]
    fn names_are_sanitised() {
        assert_eq!(
            display_name(Path::new("/x/[a]\x1b[31m.mkv")),
            "[a]?[31m.mkv"
        );
    }

    #[test]
    fn queue_window() {
        let names: Vec<String> = (0..7).map(|i| format!("f{i}")).collect();
        let st = vec![
            Some(Status::Success),
            Some(Status::Failed),
            None,
            None,
            None,
            None,
            None,
        ];
        let l = queue_lines(3, &names, &st);
        assert_eq!(l.len(), 5);
        assert!(l[0].ends_with("f1") && l[4].ends_with("f5"));
        assert_eq!(queue_lines(0, &names, &st).len(), 3);
    }
}
