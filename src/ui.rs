//! Terminal UI: a ratatui inline dashboard on a TTY, plain lines everywhere else.
//!
//! The dashboard is display-only (no raw mode, no key handling), so Ctrl-C still raises SIGINT and
//! the two-stage shutdown in `main` keeps working. Log lines are inserted above the dashboard and
//! stay in the terminal scrollback.
use crate::encode::Progress;
use console::style;
use ratatui::backend::CrosstermBackend;
use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Position, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Gauge, Padding, Paragraph, Widget};
use ratatui::{Terminal, TerminalOptions, Viewport};
use std::io::IsTerminal;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

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

/// `H:MM:SS`.
fn clock(secs: f64) -> String {
    let s = secs.max(0.0) as u64;
    format!("{}:{:02}:{:02}", s / 3600, s / 60 % 60, s % 60)
}

/// Cut to `w` characters, ending in `…` when shortened.
fn trunc(s: &str, w: usize) -> String {
    if s.chars().count() <= w {
        s.to_string()
    } else if w == 0 {
        String::new()
    } else {
        let mut t: String = s.chars().take(w - 1).collect();
        t.push('…');
        t
    }
}

const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const VIEWPORT_HEIGHT: u16 = 15;
const TICK: Duration = Duration::from_millis(100);

/// One queued file as the dashboard shows it.
#[derive(Clone)]
pub struct Row {
    pub name: String,
    pub size: u64,
}

#[derive(Default)]
struct State {
    header: Vec<(String, String)>,
    scanning: bool,
    scan_done: usize,
    scan_total: usize,
    rows: Vec<Row>,
    /// Outcome and bytes saved, per row.
    results: Vec<Option<(Status, i64)>>,
    current: Option<usize>,
    stage: String,
    secs: f64,
    duration: f64,
    speed: Option<f64>,
    fps: Option<f64>,
    out_bytes: Option<u64>,
    file_started: Option<Instant>,
    total_bytes: u64,
    done_bytes: u64,
    saved: i64,
    started: Option<Instant>,
    tick: usize,
    /// Lines waiting to be inserted above the dashboard (`true` = error).
    pending: Vec<(String, bool)>,
}

impl State {
    fn frac(&self) -> Option<f64> {
        (self.duration > 0.0).then(|| (self.secs / self.duration).clamp(0.0, 1.0))
    }

    fn file_eta(&self) -> Option<f64> {
        let frac = self.frac()?;
        match self.speed {
            Some(s) if s > 0.0 => Some((self.duration - self.secs).max(0.0) / s),
            _ => {
                let el = self.file_started?.elapsed().as_secs_f64();
                (frac > 0.01).then(|| el * (1.0 - frac) / frac)
            }
        }
    }

    /// Fraction of all input bytes that is finished, counting the running file by its progress.
    fn overall_frac(&self) -> f64 {
        if self.total_bytes == 0 {
            return 0.0;
        }
        let cur = match (self.current, self.frac()) {
            (Some(i), Some(f)) => self.rows.get(i).map_or(0.0, |r| r.size as f64 * f),
            _ => 0.0,
        };
        ((self.done_bytes as f64 + cur) / self.total_bytes as f64).clamp(0.0, 1.0)
    }

    fn overall_eta(&self) -> Option<f64> {
        let f = self.overall_frac();
        let el = self.started?.elapsed().as_secs_f64();
        (f > 0.01).then(|| el * (1.0 - f) / f)
    }
}

/// Colour switch: plain styles under `NO_COLOR` / `TERM=dumb`.
#[derive(Clone, Copy)]
struct Pal(bool);

impl Pal {
    fn detect() -> Self {
        Pal(std::env::var_os("NO_COLOR").is_none_or(|v| v.is_empty())
            && std::env::var("TERM").map_or(true, |t| t != "dumb"))
    }
    fn fg(self, c: Color) -> Style {
        if self.0 {
            Style::new().fg(c)
        } else {
            Style::new()
        }
    }
    fn bold(self, c: Color) -> Style {
        self.fg(c).add_modifier(Modifier::BOLD)
    }
    fn dim(self) -> Style {
        Style::new().add_modifier(Modifier::DIM)
    }
}

fn gauge(ratio: f64, label: String, pal: Pal, color: Color, area: Rect, buf: &mut Buffer) {
    let g = Gauge::default()
        .ratio(ratio.clamp(0.0, 1.0))
        .label(Span::styled(
            label,
            Style::new().add_modifier(Modifier::BOLD),
        ))
        .use_unicode(true);
    let g = if pal.0 {
        g.gauge_style(Style::new().fg(color).bg(Color::Indexed(238)))
    } else {
        g.gauge_style(Style::new().add_modifier(Modifier::REVERSED))
    };
    g.render(area, buf);
}

/// Draw the dashboard. Returns where the cursor should rest.
fn render(st: &State, pal: Pal, area: Rect, buf: &mut Buffer) -> Position {
    let header_h = if st.header.is_empty() || area.height < 10 {
        0
    } else {
        4
    };
    let [head, stage, file, overall, queue, foot] = Layout::vertical([
        Constraint::Length(header_h),
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Min(0),
        Constraint::Length(1),
    ])
    .areas(area);
    let w = area.width as usize;

    if header_h > 0 {
        let block = Block::bordered()
            .border_type(BorderType::Rounded)
            .border_style(pal.dim())
            .padding(Padding::horizontal(1))
            .title(Span::styled(" vidconv ", pal.bold(Color::Magenta)));
        let inner = block.inner(head);
        block.render(head, buf);
        let lines: Vec<Line> = st
            .header
            .chunks(3)
            .map(|group| {
                let mut spans = Vec::new();
                for (i, (k, v)) in group.iter().enumerate() {
                    if i > 0 {
                        spans.push(Span::styled("  ·  ", pal.dim()));
                    }
                    spans.push(Span::styled(format!("{k} "), pal.dim()));
                    spans.push(Span::styled(v.clone(), pal.fg(Color::Cyan)));
                }
                Line::from(spans)
            })
            .collect();
        Paragraph::new(lines).render(inner, buf);
    }

    let spin = SPINNER[st.tick % SPINNER.len()];
    let label_w = 10u16;
    let bar_w = (area.width / 3).clamp(10, 36);
    let row = |a: Rect| {
        let [l, b, t] = Layout::horizontal([
            Constraint::Length(label_w),
            Constraint::Length(bar_w),
            Constraint::Min(0),
        ])
        .areas(a);
        (l, b, t)
    };

    let text = |s: String, a: Rect, buf: &mut Buffer| {
        Paragraph::new(Span::raw(trunc(&s, a.width as usize))).render(a, buf)
    };
    if st.scanning {
        Paragraph::new(Line::from(vec![
            Span::styled(format!(" {spin} "), pal.bold(Color::Yellow)),
            Span::styled("Scanning library…", pal.bold(Color::White)),
        ]))
        .render(stage, buf);
        let (l, b, t) = row(file);
        Paragraph::new(Span::styled(" Probing", pal.dim())).render(l, buf);
        let r = if st.scan_total == 0 {
            0.0
        } else {
            st.scan_done as f64 / st.scan_total as f64
        };
        gauge(
            r,
            format!("{}/{}", st.scan_done, st.scan_total),
            pal,
            Color::Yellow,
            b,
            buf,
        );
        text("  ffprobe on every candidate file".into(), t, buf);
    } else if let Some(cur) = st.current {
        let name = st.rows.get(cur).map_or("", |r| r.name.as_str());
        let head_len = 3 + st.stage.chars().count() + 2;
        let n = format!("  [{}/{}]", cur + 1, st.rows.len());
        Paragraph::new(Line::from(vec![
            Span::styled(format!(" {spin} "), pal.bold(Color::Yellow)),
            Span::styled(st.stage.clone(), pal.bold(Color::Yellow)),
            Span::raw("  "),
            Span::styled(
                trunc(name, w.saturating_sub(head_len + n.chars().count())),
                pal.bold(Color::White),
            ),
            Span::styled(n, pal.dim()),
        ]))
        .render(stage, buf);

        let (l, b, t) = row(file);
        Paragraph::new(Span::styled(" File", pal.dim())).render(l, buf);
        let frac = st.frac();
        gauge(
            frac.unwrap_or(0.0),
            frac.map_or("--%".into(), |f| format!("{:.0}%", f * 100.0)),
            pal,
            Color::Cyan,
            b,
            buf,
        );
        let mut parts: Vec<String> = Vec::new();
        if let Some(s) = st.speed {
            parts.push(format!("{s:.1}x"));
        }
        if let Some(f) = st.fps {
            parts.push(format!("{f:.0} fps"));
        }
        if let (Some(bytes), Some(f)) = (st.out_bytes, frac)
            && f > 0.02
        {
            parts.push(format!("~{}", format_size((bytes as f64 / f) as i64)));
        }
        if let Some(t0) = st.file_started {
            parts.push(clock(t0.elapsed().as_secs_f64()));
        }
        if let Some(e) = st.file_eta() {
            parts.push(format!("ETA {}", clock(e)));
        }
        text(format!("  {}", parts.join(" · ")), t, buf);
    }

    if !st.scanning && !st.rows.is_empty() {
        let (l, b, t) = row(overall);
        Paragraph::new(Span::styled(" Overall", pal.dim())).render(l, buf);
        let done = st.results.iter().flatten().count();
        gauge(
            st.overall_frac(),
            format!("{done}/{}", st.rows.len()),
            pal,
            Color::Green,
            b,
            buf,
        );
        let cur = st
            .current
            .zip(st.frac())
            .and_then(|(i, f)| st.rows.get(i).map(|r| (r.size as f64 * f) as u64))
            .unwrap_or(0);
        let mut s = format!(
            "  {} of {}",
            format_size((st.done_bytes + cur) as i64),
            format_size(st.total_bytes as i64)
        );
        if let Some(e) = st.overall_eta() {
            s.push_str(&format!(" · ETA {}", clock(e)));
        }
        text(s, t, buf);
    }

    // Queue panel: a window around the current file.
    if queue.height >= 3 && !st.rows.is_empty() {
        let block = Block::bordered()
            .border_type(BorderType::Rounded)
            .border_style(pal.dim())
            .title(Span::styled(" Queue ", pal.bold(Color::Magenta)));
        let inner = block.inner(queue);
        block.render(queue, buf);
        let n = inner.height as usize;
        let cur = st.current.unwrap_or(0);
        let start = cur
            .saturating_sub(n / 2)
            .min(st.rows.len().saturating_sub(n));
        let lines: Vec<Line> = (start..(start + n).min(st.rows.len()))
            .map(|i| queue_line(st, i, cur, inner.width as usize, pal))
            .collect();
        Paragraph::new(lines).render(inner, buf);
    }

    let mut foot_text = format!(
        " Saved {}",
        format_size(st.saved).replace(" (increase)", " more")
    );
    if let Some(t0) = st.started {
        foot_text.push_str(&format!(" · elapsed {}", clock(t0.elapsed().as_secs_f64())));
    }
    foot_text.push_str(" · Ctrl-C to stop safely");
    let foot_text = trunc(&foot_text, w);
    let x = foot.x + foot_text.chars().count() as u16;
    Paragraph::new(Span::styled(foot_text, pal.dim())).render(foot, buf);
    Position::new(x.min(foot.right().saturating_sub(1)), foot.y)
}

fn queue_line<'a>(st: &State, i: usize, cur: usize, w: usize, pal: Pal) -> Line<'a> {
    let row = &st.rows[i];
    let (marker, mstyle) = match st.results.get(i).copied().flatten() {
        Some((Status::Success, _)) => ("✓", pal.fg(Color::Green)),
        Some((Status::Failed, _)) => ("✗", pal.fg(Color::Red)),
        Some((Status::Skipped, _)) => ("↷", pal.fg(Color::Cyan)),
        None if Some(i) == st.current && i == cur => ("→", pal.bold(Color::Yellow)),
        None => (" ", Style::new()),
    };
    let info = match st.results.get(i).copied().flatten() {
        Some((Status::Success, saved)) => {
            let new = row.size as i64 - saved;
            let pct = if row.size > 0 {
                saved * 100 / row.size as i64
            } else {
                0
            };
            format!(
                "{} → {}  -{}%",
                format_size(row.size as i64),
                format_size(new),
                pct
            )
        }
        Some((Status::Failed, _)) => "failed".into(),
        Some((Status::Skipped, _)) => "skipped".into(),
        None => format_size(row.size as i64),
    };
    let name_w = w.saturating_sub(info.chars().count() + 3);
    let name = trunc(&row.name, name_w);
    let pad = " ".repeat(name_w.saturating_sub(name.chars().count()) + 1);
    let name_style = if Some(i) == st.current && st.results[i].is_none() {
        pal.bold(Color::White)
    } else if st.results[i].is_some() {
        pal.dim()
    } else {
        Style::new()
    };
    Line::from(vec![
        Span::styled(marker, mstyle),
        Span::raw(" "),
        Span::styled(name, name_style),
        Span::raw(pad),
        Span::styled(info, pal.dim()),
    ])
}

struct Shared {
    state: Mutex<State>,
    done: AtomicBool,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

type Term = Terminal<CrosstermBackend<std::io::Stderr>>;

/// Insert `lines` above the dashboard (they scroll into the terminal history).
fn insert_lines(term: &mut Term, lines: &[(String, bool)], pal: Pal) {
    let width = term.size().map_or(80, |s| s.width.max(1)) as usize;
    let mut out: Vec<Line> = Vec::new();
    for (msg, is_err) in lines {
        let style = if *is_err {
            pal.fg(Color::Red)
        } else {
            Style::new()
        };
        let chars: Vec<char> = msg.chars().collect();
        if chars.is_empty() {
            out.push(Line::default());
        }
        for chunk in chars.chunks(width) {
            out.push(Line::styled(chunk.iter().collect::<String>(), style));
        }
    }
    let height = out.len().min(u16::MAX as usize) as u16;
    let _ = term.insert_before(height, |buf| {
        Paragraph::new(out).render(buf.area, buf);
    });
}

fn render_loop(mut term: Term, sh: Arc<Shared>, pal: Pal) {
    loop {
        let finished = sh.done.load(Ordering::Acquire);
        {
            let mut st = sh.lock();
            st.tick += 1;
            let pending = std::mem::take(&mut st.pending);
            if !pending.is_empty() {
                insert_lines(&mut term, &pending, pal);
            }
            let _ = term.draw(|f| {
                let area = f.area();
                let pos = render(&st, pal, area, f.buffer_mut());
                f.set_cursor_position(pos);
            });
        }
        if finished {
            break;
        }
        std::thread::sleep(TICK);
    }
    // Remove the dashboard and leave the cursor where it started, for the caller's summary.
    let _ = term.clear();
    let origin = term.get_frame().area();
    let _ = term.set_cursor_position((origin.x, origin.y));
    let _ = term.show_cursor();
}

pub struct Ui {
    quiet: bool,
    /// `None` = plain mode (no TTY, `--quiet`, `--json`).
    shared: Option<Arc<Shared>>,
    thread: Mutex<Option<JoinHandle<()>>>,
}

impl Ui {
    pub fn new(quiet: bool) -> Self {
        let mut ui = Ui {
            quiet,
            shared: None,
            thread: Mutex::new(None),
        };
        if quiet || !std::io::stderr().is_terminal() {
            return ui;
        }
        let rows = ratatui::crossterm::terminal::size().map_or(24, |(_, r)| r);
        let height = VIEWPORT_HEIGHT.min(rows.saturating_sub(1)).max(3);
        let term = Terminal::with_options(
            CrosstermBackend::new(std::io::stderr()),
            TerminalOptions {
                viewport: Viewport::Inline(height),
            },
        );
        let Ok(term) = term else { return ui };
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                started: Some(Instant::now()),
                ..State::default()
            }),
            done: AtomicBool::new(false),
        });
        let sh = Arc::clone(&shared);
        let pal = Pal::detect();
        ui.thread = Mutex::new(Some(std::thread::spawn(move || render_loop(term, sh, pal))));
        ui.shared = Some(shared);
        ui
    }

    fn with(&self, f: impl FnOnce(&mut State)) {
        if let Some(sh) = &self.shared {
            f(&mut sh.lock());
        }
    }

    /// Settings banner (key/value pairs).
    pub fn config(&self, kvs: Vec<(&str, String)>) {
        if self.quiet {
            return;
        }
        if self.shared.is_none() {
            println!("{}", style("Configuration:").bold());
            for (k, v) in &kvs {
                println!("  {k}: {}", style(v).cyan());
            }
        }
        self.with(|s| s.header = kvs.into_iter().map(|(k, v)| (k.to_string(), v)).collect());
    }

    pub fn scan_progress(&self, done: usize, total: usize) {
        self.with(|s| {
            s.scanning = true;
            s.scan_done = done;
            s.scan_total = total;
        });
    }

    /// Discovery finished: show the queue.
    pub fn queue(&self, rows: Vec<Row>) {
        if !self.quiet && self.shared.is_none() {
            println!("  Files to process: {}", style(rows.len()).cyan());
        }
        self.with(|s| {
            s.scanning = false;
            s.total_bytes = rows.iter().map(|r| r.size).sum();
            s.results = vec![None; rows.len()];
            s.rows = rows;
        });
    }

    /// Informational line (suppressed by --quiet).
    pub fn say(&self, msg: impl AsRef<str>) {
        if self.quiet {
            return;
        }
        self.print(msg.as_ref(), false);
    }

    /// Always shown.
    pub fn error(&self, msg: impl AsRef<str>) {
        self.print(msg.as_ref(), true);
    }

    fn print(&self, msg: &str, is_err: bool) {
        match &self.shared {
            Some(sh) => sh.lock().pending.push((msg.to_string(), is_err)),
            None if is_err => eprintln!("{msg}"),
            None => println!("{msg}"),
        }
    }

    pub fn start_file(&self, idx: usize) {
        self.with(|s| {
            s.current = Some(idx);
            s.stage = "Preparing".into();
            s.secs = 0.0;
            s.duration = 0.0;
            s.speed = None;
            s.fps = None;
            s.out_bytes = None;
            s.file_started = Some(Instant::now());
        });
    }

    /// What the current file is doing (`Probing`, `Encoding (VAAPI)`, `Verifying`, …).
    pub fn stage(&self, stage: &str) {
        self.with(|s| {
            s.stage = stage.to_string();
            if stage.starts_with("Encoding") {
                s.secs = 0.0;
                s.speed = None;
                s.fps = None;
                s.out_bytes = None;
            }
        });
    }

    pub fn progress(&self, p: &Progress, duration: f64) {
        self.with(|s| {
            s.secs = p.secs;
            s.duration = duration;
            s.speed = p.speed;
            s.fps = p.fps;
            s.out_bytes = p.bytes;
        });
    }

    pub fn finish_file(&self, idx: usize, status: Status, saved: i64) {
        self.with(|s| {
            if let Some(slot) = s.results.get_mut(idx) {
                *slot = Some((status, saved));
            }
            s.done_bytes += s.rows.get(idx).map_or(0, |r| r.size);
            s.saved += saved;
            s.secs = 0.0;
            s.duration = 0.0;
        });
    }

    /// Tear the dashboard down. Safe to call twice; also runs on drop.
    pub fn finish(&self) {
        let Some(sh) = &self.shared else { return };
        sh.done.store(true, Ordering::Release);
        let handle = self.thread.lock().unwrap_or_else(|e| e.into_inner()).take();
        if let Some(h) = handle {
            let _ = h.join();
        }
    }
}

impl Drop for Ui {
    fn drop(&mut self) {
        self.finish();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dump(buf: &Buffer) -> String {
        let a = buf.area;
        (a.y..a.bottom())
            .map(|y| {
                (a.x..a.right())
                    .map(|x| buf[(x, y)].symbol())
                    .collect::<String>()
                    .trim_end()
                    .to_string()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn draw(st: &State, w: u16, h: u16) -> String {
        let area = Rect::new(0, 0, w, h);
        let mut buf = Buffer::empty(area);
        render(st, Pal(false), area, &mut buf);
        dump(&buf)
    }

    fn working() -> State {
        let rows: Vec<Row> = (0..6)
            .map(|i| Row {
                name: format!("movie{i}.mkv"),
                size: 4 << 30,
            })
            .collect();
        let mut results = vec![None; 6];
        results[0] = Some((Status::Success, 2 << 30));
        results[1] = Some((Status::Failed, 0));
        State {
            header: vec![
                ("Bitrate".into(), "2000 kbps".into()),
                ("Cutoff".into(), "2600 kbps".into()),
                ("HW".into(), "VAAPI".into()),
                ("Sort".into(), "size".into()),
            ],
            total_bytes: rows.iter().map(|r| r.size).sum(),
            done_bytes: 8 << 30,
            rows,
            results,
            current: Some(2),
            stage: "Encoding (VAAPI)".into(),
            secs: 50.0,
            duration: 100.0,
            speed: Some(2.5),
            fps: Some(60.0),
            out_bytes: Some(500 << 20),
            saved: 2 << 30,
            file_started: Some(Instant::now()),
            started: Some(Instant::now()),
            ..State::default()
        }
    }

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
    fn clock_and_trunc() {
        assert_eq!(clock(0.0), "0:00:00");
        assert_eq!(clock(3725.9), "1:02:05");
        assert_eq!(trunc("abcdef", 6), "abcdef");
        assert_eq!(trunc("abcdef", 4), "abc…");
        assert_eq!(trunc("é世界", 2), "é…");
    }

    #[test]
    fn eta_uses_encode_speed() {
        let st = working();
        // 50 s left at 2.5x
        assert_eq!(st.file_eta().map(|e| e.round() as u64), Some(20));
        // (8 GB done + 2 GB of the running 4 GB file) / 24 GB
        assert!((st.overall_frac() - 10.0 / 24.0).abs() < 1e-9);
    }

    #[test]
    fn scan_state_shows_spinner_and_count() {
        let st = State {
            scanning: true,
            scan_done: 124,
            scan_total: 380,
            ..State::default()
        };
        let out = draw(&st, 80, 15);
        assert!(out.contains("Scanning library"), "{out}");
        assert!(out.contains("124/380"), "{out}");
    }

    #[test]
    fn encode_state_shows_stats_and_queue() {
        let out = draw(&working(), 90, 15);
        for want in [
            "Encoding (VAAPI)",
            "movie2.mkv",
            "[3/6]",
            "50%",
            "2.5x",
            "60 fps",
            "~1000.00 MB",
            "3/6",
            "Queue",
            "✓ movie0.mkv",
            "4.00 GB → 2.00 GB  -50%",
            "✗ movie1.mkv",
            "→ movie2.mkv",
            "Saved 2.00 GB",
            "Ctrl-C",
        ] {
            assert!(out.contains(want), "missing {want:?} in:\n{out}");
        }
    }

    #[test]
    fn narrow_and_short_terminals_do_not_panic() {
        for (w, h) in [(20, 15), (40, 6), (1, 1), (60, 3), (200, 40)] {
            draw(&working(), w, h);
        }
        let out = draw(&working(), 60, 6);
        assert!(out.contains("Encoding"), "{out}");
        assert!(!out.contains("Queue"), "{out}");
    }
}
