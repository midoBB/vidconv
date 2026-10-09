# vidconv

Batch-shrink videos to 720p H.264/AAC MP4. Uses VAAPI hardware encoding for HEVC/AV1 sources when available, libx264 otherwise. Converted files carry a `vidconv=1` tag and are skipped on later runs.

## Requirements

`ffmpeg` and `ffprobe` on `PATH`. VAAPI is optional (auto-detected, with a software fallback). HDR tone mapping needs ffmpeg built with `zscale`/`tonemap`.

## Install

```sh
make install          # builds with cargo, installs to ~/.local plus shell completions
```

## Usage

```sh
vidconv movie.mkv                 # convert one file (replaces it with movie.mp4)
vidconv -k movie.mkv              # keep the original, write movie-720.mp4
vidconv --all                     # every video in the current directory
vidconv -r ~/Videos               # a directory tree
vidconv --output-dir out *.mkv    # write results elsewhere, originals untouched
```

With several inputs, a directory, or `--all`, files already tagged by vidconv and files below the bitrate cutoff (HEVC/AV1 excepted) are skipped.

| Option | Default | |
|---|---|---|
| `-b, --bitrate` | 2500 | target kbps (1000-10000) |
| `-c, --cutoff` | bitrate+500 | skip lower-bitrate files in multi mode |
| `-n, --no-hw` / `--hw auto\|off\|force` | auto | VAAPI use; `--vaapi-device PATH` picks the render node |
| `-k, --keep` | | keep the original |
| `-a, --all`, `-r, --recursive` | | scan current dir / descend into folders |
| `-s, --sort-by date\|size` | date | processing order (descending) |
| `-f, --force` | | replace even if the result is not smaller |
| `--output-dir DIR` | | write results there, keep originals |
| `--height`, `--max-fps`, `--crf`, `--preset`, `--audio-bitrate` | 720, 24, 20, slow, 64 | encoder knobs |
| `--no-sharpen`, `--fragmented`, `-q` | | misc |
| `--json` | | one JSON object per file (`status`: success/skipped/failed/ignored) and a final `summary` on stdout; implies `-q` |
| `--completions SHELL` | | print a completion script |

Every option also reads `VIDCONV_<NAME>` (e.g. `VIDCONV_BITRATE`, `VIDCONV_NO_HW`, `VIDCONV_PROCESS_ALL`, `VIDCONV_SORT_BY`).

Exit codes: 0 ok, 1 a file failed (details in `vidconv_errors.log`), 2 usage error, 130 interrupted. Ctrl-C stops cleanly and removes temporary files; a second Ctrl-C exits at once.

Symlinked inputs are never replaced: the result is written beside them as `<name>-720.mp4`.

## Development

```sh
make lint test
```

End-to-end tests need `ffmpeg`/`ffprobe`; they are skipped when missing.
