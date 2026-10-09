# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-10-09

Rewritten in Rust. Still shells out to `ffmpeg` and `ffprobe`; the `file` command is no longer needed.

### Changed

- Replacing an original now yields `<name>.mp4` (an `.mkv`/`.avi` input no longer keeps its extension while holding MP4 data); a different existing `<name>.mp4` is never overwritten
- `--keep` keeps the original and writes `<name>-720.mp4` next to it, never overwriting existing files (`-720-1`, ...)
- Default output is faststart MP4 (`--fragmented` restores the old layout)
- Exit code is 1 when any file failed, 130 when interrupted, 2 on usage errors; skipped files are no longer counted as successes
- Videos are detected with ffprobe and a list of video extensions when scanning directories
- Output never upscales; odd sizes are scaled to even dimensions instead of padded with black
- Only the first video and first audio stream are kept (multichannel audio is downmixed to stereo); subtitles, data streams, attachments and chapters are dropped; `creation_time` is preserved

### Added

- Rotation and non-square pixel aspect ratio are honoured when choosing portrait/landscape and size
- HDR (PQ/HLG) sources are tone-mapped to SDR when ffmpeg has `zscale` and `tonemap`
- Output is verified (decodes, duration matches) before the original is replaced; a result that is not smaller keeps the original unless `--force`
- Work happens in a hidden temp file that is always cleaned up, including on Ctrl-C (exit 130); a second Ctrl-C exits immediately
- Free-space and write-permission checks, a stall watchdog, and a VAAPI self-test with automatic render-node detection
- Probe failures are reported and logged instead of files silently disappearing; explicit directory arguments are scanned
- Options: `-r/--recursive`, `-f/--force`, `--output-dir`, `--hw auto|off|force`, `--vaapi-device`, `--height`, `--max-fps`, `--crf`, `--preset`, `--audio-bitrate`, `--no-sharpen`, `--fragmented`, `-q/--quiet`, `--json` (one JSON object per file plus a summary), `--completions <shell>`, `--version`
- Unit tests, golden tests that pin the ffmpeg arguments to those captured from the 1.x Python tool, and end-to-end tests against generated clips; CI runs fmt, clippy and tests

### Fixed

- Filenames containing `[...]`, control characters or a leading `-` no longer break output or tool calls
- Missing stream bitrate (MKV/WebM) is estimated from size and duration instead of forcing re-encode
- Probe failures (`0/0` frame rate, `N/A` duration, no video stream) no longer crash discovery or make files vanish silently
- Progress no longer jumps to 100% after a failure

## [1.0.17] - 2026-10-08

### Fixed

- Persist the `vidconv` marker in converted MP4 files (`use_metadata_tags`) so they are recognised as already converted
- Skip already converted files during discovery when processing multiple files or `--all`

## [1.0.16] - 2026-05-29

### Fixed

- Pad frames to even dimensions so libx264/VAAPI do not reject odd-sized output

## [1.0.15] - 2026-05-13

### Changed

- Improve video encoding quality (CRF-based x264 settings, unsharp filter, keyframe tuning)

### Added

- Check whether a file was already converted by vidconv

## [1.0.14] - 2026-04-14

### Fixed

- Ensure even dimensions when aspect-ratio scaling produces odd pixel counts

### Changed

- Use `uv` for virtualenv and pip management

## [1.0.13] - 2025-12-09

### Changed

- Parallelize video file scanning
- Cache video metadata during discovery and reuse it when converting

## [1.0.12] - 2025-11-19

### Changed

- Replace python-magic with system 'file' command

## [1.0.11] - 2025-09-10

### Fixed

- Improve error handling and logging for unexpected failures while processing a file

## [1.0.10] - 2025-08-08

### Fixed

- Prevent crashes from deleted or moved files
- Sanitize filenames

## [1.0.9] - 2025-03-19

### Changed

- Add sorting options for video files

## [1.0.8] - 2025-02-25

### Added

- Add audio transcoding to AAC with 64k bitrate

## [1.0.7] - 2025-02-20

### Changed

- Handle non-numeric bitrate values

## [1.0.6] - 2025-02-20

### Changed

- Handle framerate as a single value

## [1.0.5] - 2025-02-03

### Changed

- Add preflight required tools check

## [1.0.4] - 2025-02-03

### Changed

- Add elapsed time label to progress bar
- Revert "feat(progress): Add elapsed time label to progress bar"

### Fixed

- Reset progress bar on software encoding fallback

## [1.0.3] - 2025-02-03

### Changed

- Replace tqdm with Rich for enhanced UI
- Add space saved calculation and display

## [1.0.2] - 2025-02-02

### Changed

- Use mise to get python version in release workflow
- Add bitrate and duration to video metadata

## [1.0.1] - 2025-02-02

### Changed

- Sort videos by modification time and format progress bar

## [1.0.0] - 2025-02-02

### Changed

- Initial commit
- Add progress tracking and error handling
- Enhance signal handling and ffmpeg process management
- Automate release creation via GitHub Actions

[2.0.0]: https://github.com/midoBB/vidconv/compare/v1.0.17...v2.0.0
[1.0.17]: https://github.com/midoBB/vidconv/compare/v1.0.16..v1.0.17
[1.0.16]: https://github.com/midoBB/vidconv/compare/v1.0.15..v1.0.16
[1.0.15]: https://github.com/midoBB/vidconv/compare/v1.0.14..v1.0.15
[1.0.14]: https://github.com/midoBB/vidconv/compare/v1.0.13..v1.0.14
[1.0.13]: https://github.com/midoBB/vidconv/compare/v1.0.12..v1.0.13
[1.0.12]: https://github.com/midoBB/vidconv/compare/v1.0.11..v1.0.12
[1.0.11]: https://github.com/midoBB/vidconv/compare/v1.0.10..v1.0.11
[1.0.10]: https://github.com/midoBB/vidconv/compare/v1.0.9..v1.0.10
[1.0.9]: https://github.com/midoBB/vidconv/compare/v1.0.8..v1.0.9
[1.0.8]: https://github.com/midoBB/vidconv/compare/v1.0.7..v1.0.8
[1.0.7]: https://github.com/midoBB/vidconv/compare/v1.0.6..v1.0.7
[1.0.6]: https://github.com/midoBB/vidconv/compare/v1.0.5..v1.0.6
[1.0.5]: https://github.com/midoBB/vidconv/compare/v1.0.4..v1.0.5
[1.0.4]: https://github.com/midoBB/vidconv/compare/v1.0.3..v1.0.4
[1.0.3]: https://github.com/midoBB/vidconv/compare/v1.0.2..v1.0.3
[1.0.2]: https://github.com/midoBB/vidconv/compare/v1.0.1..v1.0.2
[1.0.1]: https://github.com/midoBB/vidconv/compare/v1.0.0..v1.0.1
