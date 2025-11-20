# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.13] - 2025-11-20

### Changed

- Parallelize video file scanning

## [1.0.12] - 2025-11-19

### Changed

- Replace python-magic with system 'file' command

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

[1.0.13]: https://github.com/midoBB/vidconv/compare/v1.0.12..v1.0.13
[1.0.12]: https://github.com/midoBB/vidconv/compare/v1.0.11..v1.0.12
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
