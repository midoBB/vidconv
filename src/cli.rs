use clap::{Parser, ValueEnum};
use std::path::PathBuf;

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum SortBy {
    Date,
    Size,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum HwMode {
    /// VAAPI when a device works and the source is eligible (default)
    Auto,
    Off,
    /// VAAPI without checking that the device can decode the source codec
    Force,
}

fn existing_path(s: &str) -> Result<PathBuf, String> {
    let p = PathBuf::from(s);
    if p.exists() {
        Ok(p)
    } else {
        Err(format!("path '{s}' does not exist"))
    }
}

/// Video conversion tool with hardware/software encoding support
#[derive(Parser, Debug)]
#[command(name = "vidconv", version)]
pub struct Cli {
    /// Files or directories to convert
    #[arg(value_name = "INPUT_PATH", value_parser = existing_path)]
    pub inputs: Vec<PathBuf>,

    /// Bitrate in kbps (1000-10000)
    #[arg(short, long, env = "VIDCONV_BITRATE", default_value_t = 2500,
          value_parser = clap::value_parser!(u32).range(1000..=10000))]
    pub bitrate: u32,

    /// Cutoff for bitrate when converting many videos (1000-10000) [default: bitrate + 500]
    #[arg(short, long, env = "VIDCONV_CUTOFF",
          value_parser = clap::value_parser!(u32).range(1000..=10000))]
    pub cutoff: Option<u32>,

    /// Disable hardware acceleration
    #[arg(short, long, env = "VIDCONV_NO_HW")]
    pub no_hw: bool,

    /// Keep the original; write `<name>-720.mp4` next to it instead of replacing it
    #[arg(short, long, env = "VIDCONV_KEEP")]
    pub keep: bool,

    /// Process all video files in the current directory
    #[arg(short, long = "all", env = "VIDCONV_PROCESS_ALL")]
    pub all: bool,

    /// Sort video files by date or size (descending)
    #[arg(
        short,
        long,
        env = "VIDCONV_SORT_BY",
        value_enum,
        default_value = "date"
    )]
    pub sort_by: SortBy,

    /// Descend into sub-directories
    #[arg(short, long, env = "VIDCONV_RECURSIVE")]
    pub recursive: bool,

    /// Replace the original even if the result is not smaller
    #[arg(short, long)]
    pub force: bool,

    /// Write results here (originals are kept)
    #[arg(long, value_name = "DIR")]
    pub output_dir: Option<PathBuf>,

    /// Hardware acceleration mode
    #[arg(long, value_enum, default_value = "auto")]
    pub hw: HwMode,

    /// VAAPI render node (default: first /dev/dri/renderD*)
    #[arg(long, value_name = "PATH")]
    pub vaapi_device: Option<PathBuf>,

    /// Target height in pixels
    #[arg(long, default_value_t = 720, value_parser = clap::value_parser!(u32).range(144..=4320))]
    pub height: u32,

    /// Maximum frame rate
    #[arg(long, default_value_t = 24.0)]
    pub max_fps: f64,

    /// x264 CRF for software encoding
    #[arg(long, default_value_t = 20, value_parser = clap::value_parser!(u8).range(0..=51))]
    pub crf: u8,

    /// x264 preset for software encoding
    #[arg(long, default_value = "slow", value_parser = ["ultrafast", "superfast", "veryfast",
          "faster", "fast", "medium", "slow", "slower", "veryslow"])]
    pub preset: String,

    /// AAC audio bitrate in kbps
    #[arg(long, default_value_t = 64, value_parser = clap::value_parser!(u32).range(16..=512))]
    pub audio_bitrate: u32,

    /// Disable the sharpening filter
    #[arg(long)]
    pub no_sharpen: bool,

    /// Write fragmented MP4 instead of faststart MP4
    #[arg(long)]
    pub fragmented: bool,

    /// Only print errors
    #[arg(short, long)]
    pub quiet: bool,

    /// Print one JSON object per file and a final summary on stdout (implies --quiet)
    #[arg(long)]
    pub json: bool,

    /// Print a shell completion script and exit
    #[arg(long, value_name = "SHELL")]
    pub completions: Option<clap_complete::Shell>,
}

impl Cli {
    /// No human-oriented progress or status output.
    pub fn silent(&self) -> bool {
        self.quiet || self.json
    }

    pub fn effective_cutoff(&self) -> u32 {
        self.cutoff.unwrap_or(self.bitrate + 500)
    }
}
