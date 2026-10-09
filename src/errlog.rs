//! Error log: truncated at start, removed at the end if nothing was written.
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const NAME: &str = "vidconv_errors.log";

pub struct ErrorLog {
    path: PathBuf,
    used: bool,
}

fn log_dir_fallback() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_STATE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/state")))?;
    let d = base.join("vidconv");
    std::fs::create_dir_all(&d).ok()?;
    Some(d)
}

/// "YYYY-MM-DD HH:MM:SS UTC" without a datetime dependency.
pub fn utc_timestamp(t: SystemTime) -> String {
    let secs = t
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0) as i64;
    let (days, rem) = (secs.div_euclid(86400), secs.rem_euclid(86400));
    // civil-from-days (H. Hinnant)
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!(
        "{y:04}-{m:02}-{d:02} {:02}:{:02}:{:02} UTC",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}

impl ErrorLog {
    pub fn new() -> Self {
        Self::at(PathBuf::from(NAME))
    }

    pub fn at(path: PathBuf) -> Self {
        let _ = std::fs::remove_file(&path);
        Self { path, used: false }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn log(&mut self, file: &Path, msg: &str) {
        let entry = format!(
            "[{}] {}\n{}\n\n",
            utc_timestamp(SystemTime::now()),
            file.display(),
            msg.trim_end()
        );
        let open = |p: &Path| OpenOptions::new().create(true).append(true).open(p);
        let f: Option<File> = open(&self.path).ok().or_else(|| {
            // cwd not writable: fall back to the user state dir.
            let p = log_dir_fallback()?.join(NAME);
            let f = open(&p).ok()?;
            self.path = p;
            Some(f)
        });
        match f {
            Some(mut f) => {
                let _ = f.write_all(entry.as_bytes());
                self.used = true;
            }
            None => eprint!("{entry}"),
        }
    }

    /// Remove the log unless errors occurred and something was recorded.
    pub fn finish(&self, had_errors: bool) {
        if !(had_errors && self.used) {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn timestamp_known_values() {
        assert_eq!(utc_timestamp(UNIX_EPOCH), "1970-01-01 00:00:00 UTC");
        assert_eq!(
            utc_timestamp(UNIX_EPOCH + Duration::from_secs(1_709_210_096)),
            "2024-02-29 12:34:56 UTC"
        );
    }

    #[test]
    fn log_lifecycle() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("e.log");
        let mut l = ErrorLog::at(p.clone());
        l.finish(true);
        assert!(!p.exists());
        l.log(Path::new("a.mkv"), "boom");
        l.finish(true);
        assert!(std::fs::read_to_string(&p).unwrap().contains("boom"));
    }
}
