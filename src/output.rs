//! Temp output, collision-free naming, atomic finalize and disk checks.
use std::ffi::{CString, OsString};
use std::fs::{self, File, Metadata};
use std::io;
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

static COUNTER: AtomicU32 = AtomicU32::new(0);

/// Hidden temp file next to the destination; removed on drop unless persisted.
pub struct TempOutput {
    path: PathBuf,
    armed: bool,
}

impl TempOutput {
    pub fn create(dir: &Path) -> io::Result<Self> {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = dir.join(format!(".vidconv-{}-{n}.tmp.mp4", std::process::id()));
        File::options().write(true).create_new(true).open(&path)?;
        Ok(Self { path, armed: true })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn size(&self) -> io::Result<u64> {
        Ok(fs::metadata(&self.path)?.len())
    }

    /// Moves the temp file to `dest` without overwriting an existing file.
    pub fn persist_noclobber(&mut self, dest: &Path) -> io::Result<()> {
        match fs::hard_link(&self.path, dest) {
            Ok(()) => {
                let _ = fs::remove_file(&self.path);
            }
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => return Err(e),
            // Filesystem without hard links: plain rename (destination was checked free).
            Err(_) => fs::rename(&self.path, dest)?,
        }
        self.armed = false;
        Ok(())
    }

    /// Atomically replaces `dest` (which may be the original input).
    pub fn persist_overwrite(&mut self, dest: &Path) -> io::Result<()> {
        fs::rename(&self.path, dest)?;
        self.armed = false;
        Ok(())
    }

    /// Copy permissions and mtime from the original so the result looks like an in-place edit.
    pub fn copy_attrs_from(&self, orig: &Metadata) {
        let _ = fs::set_permissions(&self.path, orig.permissions());
        if let (Ok(m), Ok(f)) = (
            orig.modified(),
            File::options().write(true).open(&self.path),
        ) {
            let _ = f.set_modified(m);
        }
    }
}

impl Drop for TempOutput {
    fn drop(&mut self) {
        if self.armed {
            let _ = fs::remove_file(&self.path);
        }
    }
}

/// `dir/stem[-N].ext` that does not exist yet.
pub fn unique_path(dir: &Path, stem: &std::ffi::OsStr, ext: &str) -> PathBuf {
    let make = |suffix: Option<u32>| {
        let mut name = OsString::from(stem);
        if let Some(n) = suffix {
            name.push(format!("-{n}"));
        }
        name.push(".");
        name.push(ext);
        dir.join(name)
    };
    let first = make(None);
    if fs::symlink_metadata(&first).is_err() {
        return first;
    }
    (1..)
        .map(|n| make(Some(n)))
        .find(|p| fs::symlink_metadata(p).is_err())
        .unwrap()
}

pub fn stem_with(stem: &std::ffi::OsStr, suffix: &str) -> OsString {
    let mut s = OsString::from(stem);
    s.push(suffix);
    s
}

pub fn free_space(dir: &Path) -> Option<u64> {
    let c = CString::new(dir.as_os_str().as_bytes()).ok()?;
    // SAFETY: zeroed statvfs is a valid out-parameter; `c` is NUL-terminated.
    let mut s: libc::statvfs = unsafe { std::mem::zeroed() };
    (unsafe { libc::statvfs(c.as_ptr(), &mut s) } == 0)
        .then(|| s.f_bavail as u64 * s.f_frsize as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsStr;

    #[test]
    fn unique_names() {
        let d = tempfile::tempdir().unwrap();
        let a = unique_path(d.path(), OsStr::new("x"), "mp4");
        assert_eq!(a, d.path().join("x.mp4"));
        fs::write(&a, b"1").unwrap();
        assert_eq!(
            unique_path(d.path(), OsStr::new("x"), "mp4"),
            d.path().join("x-1.mp4")
        );
    }

    #[test]
    fn temp_removed_on_drop_and_noclobber() {
        let d = tempfile::tempdir().unwrap();
        let p;
        {
            let t = TempOutput::create(d.path()).unwrap();
            p = t.path().to_path_buf();
            assert!(p.exists());
        }
        assert!(!p.exists());
        let dest = d.path().join("o.mp4");
        fs::write(&dest, b"mine").unwrap();
        let mut t = TempOutput::create(d.path()).unwrap();
        assert!(t.persist_noclobber(&dest).is_err());
        assert_eq!(fs::read(&dest).unwrap(), b"mine");
        t.persist_overwrite(&dest).unwrap();
        assert_eq!(fs::read(&dest).unwrap(), b"");
    }

    #[test]
    fn free_space_works() {
        assert!(free_space(Path::new(".")).is_some());
    }
}
