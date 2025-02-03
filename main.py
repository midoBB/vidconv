#!/usr/bin/env python3
import shutil
import signal
import subprocess
import sys
from enum import Enum
from pathlib import Path

import click
from magic import Magic
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

ERROR_LOG = "vidconv_errors.log"
# Global variables for signal handling
current_process = None
current_output = None
console = Console()


def signal_handler(sig, frame):
    """Handle signals and clean up resources"""
    global current_process, current_output

    if current_process is not None:
        # Terminate the running ffmpeg process
        current_process.terminate()
        try:
            current_process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            current_process.kill()

    # Clean up unfinished output file
    if current_output and current_output.exists():
        try:
            current_output.unlink()
            console.print(f"\nRemoved incomplete file: {current_output}")
        except Exception as e:
            console.print(f"\nError removing incomplete file: {e}")

    # Remove error log if empty
    if Path(ERROR_LOG).exists() and Path(ERROR_LOG).stat().st_size == 0:
        Path(ERROR_LOG).unlink()

    console.print("\nExiting due to interrupt...")
    sys.exit(1)


def format_size(bytes: int) -> str:
    """Convert bytes to a human-readable format."""
    value = float(bytes)
    if value < 0:
        suffix = " (increase)"
        value = -value
    else:
        suffix = ""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024.0:
            return f"{bytes / 1024.0 ** (['B', 'KB', 'MB', 'GB', 'TB'].index(unit)):.1f} {unit}{suffix}"
        value /= 1024.0
    return f"{bytes / 1024.0**5:.1f} PB{suffix}"


def get_video_files(directory):
    """Get video files in directory using python-magic"""
    mime = Magic(mime=True)
    videos = []
    with console.status("[bold green]Searching for video files..."):
        for file in directory.iterdir():
            if file.is_file():
                mimetype = mime.from_file(file)
                if mimetype.startswith("video/"):
                    videos.append(file)

    return sorted(videos, key=lambda f: f.stat().st_mtime, reverse=True)


class FFmpegResult(Enum):
    SUCCESS = 0
    SKIPPED = 1
    ERROR = 2


def get_only_video_files(inputs):
    """Get only video files from a list of inputs"""
    videos = []
    mime = Magic(mime=True)
    with console.status("[bold green]Searching for video files..."):
        for input in inputs:
            if input.is_file():
                mimetype = mime.from_file(input)
                if mimetype.startswith("video/"):
                    videos.append(input)
    return sorted(videos, key=lambda f: f.stat().st_mtime, reverse=True)


def convert_time_to_seconds(time_str):
    """Convert FFmpeg time string (HH:MM:SS.ms) to total seconds"""
    try:
        parts = time_str.split(":")
        hours, minutes, rest = 0, 0, "0"
        if len(parts) == 3:
            hours, minutes, rest = parts
        elif len(parts) == 2:
            minutes, rest = parts
        else:
            rest = parts[0]

        seconds_parts = rest.split(".")
        seconds = int(seconds_parts[0])
        milliseconds = (
            int(seconds_parts[1].ljust(3, "0")[:3]) if len(seconds_parts) > 1 else 0
        )
        return int(hours) * 3600 + int(minutes) * 60 + seconds + milliseconds / 1000
    except Exception:
        return 0


def run_ffmpeg_hw(
    input_file,
    output_file,
    width,
    height,
    bitrate,
    framerate,
    duration,
    progress: Progress,
    task_id: TaskID,
) -> FFmpegResult:
    """Run FFmpeg with hardware acceleration and track progress"""
    global current_process
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-v",
        "error",
        "-stats",
        "-init_hw_device",
        "vaapi=card:/dev/dri/renderD128",
        "-hwaccel",
        "vaapi",
        "-hwaccel_output_format",
        "vaapi",
        "-hwaccel_device",
        "card",
        "-i",
        str(input_file),
        "-filter_hw_device",
        "card",
        "-vf",
        f"scale_vaapi=w={width}:h={height}:force_original_aspect_ratio=decrease,hwdownload,format=nv12,hwupload",
        "-r",
        str(framerate),
        "-movflags",
        "frag_keyframe+empty_moov",
        "-c:v",
        "h264_vaapi",
        "-b:v",
        f"{bitrate}K",
        "-maxrate",
        f"{bitrate * 1.2}K",
        "-bufsize",
        f"{bitrate * 1.3}K",
        "-g",
        "48",
        "-keyint_min",
        "48",
        "-c:a",
        "copy",
        str(output_file),
    ]

    try:
        current_process = subprocess.Popen(
            cmd, stderr=subprocess.PIPE, text=True, bufsize=1
        )
        stderr_lines = []
        while True:
            line = current_process.stderr.readline()
            if not line:
                break
            line = line.strip()
            stderr_lines.append(line)
            if "time=" in line:
                time_part = line.split("time=", 1)[1].strip()
                time_tokens = time_part.split()
                if time_tokens:
                    time_str = time_tokens[0]
                    current_time = convert_time_to_seconds(time_str)
                    if duration > 0:
                        current_percent = min((current_time / duration) * 100, 100.0)
                        task = next(
                            (t for t in progress.tasks if t.id == task_id), None
                        )
                        if task is not None:
                            delta = current_percent - task.completed
                            if delta > 0:
                                progress.update(task_id, advance=delta)
        current_process.wait()
        progress.update(task_id, completed=100)
        if current_process.returncode != 0:
            log_error(input_file, "\n".join(stderr_lines))
            return FFmpegResult.ERROR
        return FFmpegResult.SUCCESS
    except Exception as e:
        log_error(input_file, str(e))
        return FFmpegResult.ERROR


def run_ffmpeg_sw(
    input_file,
    output_file,
    width,
    height,
    bitrate,
    framerate,
    duration,
    progress: Progress,
    task_id: TaskID,
) -> FFmpegResult:
    """Run FFmpeg with software encoding and track progress"""
    global current_process
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-v",
        "error",
        "-stats",
        "-i",
        str(input_file),
        "-vf",
        f"scale=w={width}:h={height}",
        "-r",
        str(framerate),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-profile:v",
        "high",
        "-level",
        "4.1",
        "-preset",
        "slow",
        "-movflags",
        "frag_keyframe+empty_moov",
        "-crf",
        "21",
        "-maxrate",
        f"{bitrate * 1.2}K",
        "-bufsize",
        f"{bitrate * 1.3}K",
        "-g",
        "48",
        "-keyint_min",
        "48",
        "-c:a",
        "copy",
        str(output_file),
    ]

    try:
        current_process = subprocess.Popen(
            cmd, stderr=subprocess.PIPE, text=True, bufsize=1
        )
        stderr_lines = []
        while True:
            line = current_process.stderr.readline()
            if not line:
                break
            line = line.strip()
            stderr_lines.append(line)
            if "time=" in line:
                time_part = line.split("time=", 1)[1].strip()
                time_tokens = time_part.split()
                if time_tokens:
                    time_str = time_tokens[0]
                    current_time = convert_time_to_seconds(time_str)
                    if duration > 0:
                        current_percent = min((current_time / duration) * 100, 100.0)
                        task = next(
                            (t for t in progress.tasks if t.id == task_id), None
                        )
                        if task is not None:
                            delta = current_percent - task.completed
                            if delta > 0:
                                progress.update(task_id, advance=delta)
        current_process.wait()
        progress.update(task_id, completed=100)
        if current_process.returncode != 0:
            log_error(input_file, "\n".join(stderr_lines))
            return FFmpegResult.ERROR
        return FFmpegResult.SUCCESS
    except Exception as e:
        log_error(input_file, str(e))
        return FFmpegResult.ERROR


def get_video_metadata(file_path: Path) -> tuple[str, int, int, float, int, float]:
    """
    Retrieves video metadata using ffprobe, including codec, resolution, frame rate,
    bitrate, and duration.

    Args:
        file_path (Path): The path to the video file.

    Returns:
        Tuple[str, int, int, float, int, float]: A tuple containing:
            - codec name (str)
            - width (int)
            - height (int)
            - frame rate (float)
            - bitrate (kbps) (int)
            - duration (seconds) (float)

    Raises:
        RuntimeError: If ffprobe returns a non-zero exit code.
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,r_frame_rate,bit_rate",
        "-show_entries",
        "format=duration",
        "-of",
        "csv=p=0",
        str(file_path),
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"FFprobe error: {result.stderr.decode()}")

    metadata = result.stdout.decode().strip().split("\n")
    stream_data, duration = metadata if len(metadata) == 2 else (metadata[0], "0")
    codec, width, height, framerate, bitrate = stream_data.split(",")

    # Parse frame rate
    if "/" in framerate:
        num, den = framerate.split("/")
        framerate = float(num) / float(den)
    else:
        framerate = float(framerate)

    # Convert bitrate to kbps
    bitrate_kbps = int(bitrate) // 1000 if bitrate.isdigit() else 0

    return codec, int(width), int(height), framerate, bitrate_kbps, float(duration)


def log_error(file_path, error_msg):
    """Log errors to the error log file"""
    with open(ERROR_LOG, "a") as f:
        f.write(f"Error processing file: {file_path}\n")
        f.write(f"Error message:\n{error_msg}\n")
        f.write("-" * 40 + "\n")


def get_queue_display(
    current_index: int, video_files: list[Path], statuses: dict[int, str]
) -> list[str]:
    """
    Return a list of strings for display showing two files before, the current file,
    and two after, using different symbols for previously processed files based on their status.

    Args:
        current_index (int): Index of the file currently processing.
        video_files (list[Path]): List of video files.
        statuses (dict[int, str]): Mapping from file index to status, where the status can be:
            "success", "failed", or "skipped".

    Returns:
        list[str]: List of strings to display in the queue.
    """
    start = max(0, current_index - 2)
    end = min(len(video_files), current_index + 3)
    lines = []

    for idx in range(start, end):
        if idx >= len(video_files):
            continue
        if idx < current_index:
            status = statuses.get(idx, "pending")
            if status == FFmpegResult.SUCCESS:
                prefix = "[green]✓[/green]"
            elif status == FFmpegResult.ERROR:
                prefix = "[red]✗[/red]"
            elif status == FFmpegResult.SKIPPED:
                prefix = "[cyan]S[/cyan]"
            else:
                prefix = "[white]?[/white]"
        elif idx == current_index:
            prefix = "[yellow]→[/yellow]"
        else:
            prefix = "  "
        lines.append(f"{prefix} {video_files[idx].name}")
    return lines


def process_file(
    file_path: Path,
    bitrate: int,
    cutoff: int,
    no_hw: bool,
    keep: bool,
    process_all: bool,
    progress: Progress,
    task_id: TaskID,
) -> tuple[FFmpegResult, int]:
    """
    Processes a single video file and returns the result along with space saved.
    """
    global current_process, current_output
    original_size = file_path.stat().st_size
    space_saved = 0

    try:
        codec, width, height, framerate, file_bitrate, duration = get_video_metadata(
            file_path
        )
    except RuntimeError as e:
        console.print(f"Skipping {file_path.name}: {str(e)}")
        return FFmpegResult.ERROR, 0

    if process_all:
        if file_bitrate < cutoff and codec not in ("hevc", "av1"):
            console.print(f"Skipping {file_path.name}")
            return FFmpegResult.SKIPPED, 0

    output_path = file_path.with_stem(f"{file_path.stem}-720").with_suffix(".mp4")
    current_output = output_path

    if height > width:
        new_height = 720
        new_width = int(width * new_height / height)
        new_width += new_width % 2
    else:
        new_width = 1280
        new_height = 720

    if framerate > 24:
        framerate = 24

    try:
        if not no_hw and codec in ("hevc", "av1"):
            result = run_ffmpeg_hw(
                file_path,
                output_path,
                new_width,
                new_height,
                bitrate,
                framerate,
                duration,
                progress,
                task_id,
            )
            if result != FFmpegResult.SUCCESS:
                if output_path.exists():
                    output_path.unlink()
                console.print("Hardware encoding failed, trying software...")
                progress.update(task_id, completed=0)
                result = run_ffmpeg_sw(
                    file_path,
                    output_path,
                    new_width,
                    new_height,
                    bitrate,
                    framerate,
                    duration,
                    progress,
                    task_id,
                )
        else:
            result = run_ffmpeg_sw(
                file_path,
                output_path,
                new_width,
                new_height,
                bitrate,
                framerate,
                duration,
                progress,
                task_id,
            )
    except KeyboardInterrupt:
        return FFmpegResult.ERROR, 0
    finally:
        current_process = None
        current_output = None

    if result == FFmpegResult.SUCCESS:
        if not keep:
            shutil.move(output_path, file_path)
            new_size = file_path.stat().st_size
        else:
            new_size = output_path.stat().st_size
        space_saved = original_size - new_size
        console.print(f"Successfully processed: {file_path.name}")
    elif result == FFmpegResult.SKIPPED:
        space_saved = 0
    else:
        if output_path.exists():
            output_path.unlink()
        console.print(f"Failed to process: {file_path.name}")
        space_saved = 0

    return result, space_saved


def check_required_tools():
    """Check if required tools are installed and available in the system PATH."""
    required_tools = {
        "ffmpeg": "FFmpeg (required for video processing)",
        "ffprobe": "FFprobe (required for video processing)",
    }
    missing_tools = []

    for tool, description in required_tools.items():
        if not shutil.which(tool):
            missing_tools.append(f"{tool} ({description})")

    if missing_tools:
        console.print(
            "[red]Error: The following required tools are not installed:[/red]"
        )
        for tool in missing_tools:
            console.print(f"  - {tool}")
        console.print("\nPlease install the missing tools and try again.")
        sys.exit(1)


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.argument("input_path", required=False, type=click.Path(exists=True), nargs=-1)
@click.option(
    "--bitrate",
    "-b",
    type=click.IntRange(1000, 10000),
    default=2500,
    help="Bitrate in kbps (1000-10000)",
)
@click.option(
    "--cutoff",
    "-c",
    type=click.IntRange(1000, 10000),
    default=None,
    help="Cutoff for bitrate when converting all videos (1000-10000) [default: bitrate + 500]",
)
@click.option("--no-hw", "-n", is_flag=True, help="Disable hardware acceleration")
@click.option("--keep", "-k", is_flag=True, help="Keep intermediate files")
@click.option(
    "--all",
    "-a",
    "process_all",
    is_flag=True,
    help="Process all video files in directory",
)
def main(input_path, bitrate, cutoff, no_hw, keep, process_all):
    """Video conversion tool with hardware/software encoding support"""
    check_required_tools()
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    if cutoff is None:
        cutoff = bitrate + 500
    # Initialize error log
    if Path(ERROR_LOG).exists():
        Path(ERROR_LOG).unlink()

    processed_files = []
    errors = 0
    statuses = {}
    total_space_saved = 0

    if process_all:
        input_path = Path.cwd()
        video_files = get_video_files(input_path)
        if not video_files:
            console.print("No video files found in current directory.")
            return
    elif input_path:
        if len(input_path) > 1:
            process_all = True
        video_files = get_only_video_files([Path(file) for file in input_path])
    else:
        raise click.UsageError("Must specify INPUT_PATH or use --all")

    console.print("[bold]Configuration:[/bold]")
    console.print(f"  Bitrate: [cyan]{bitrate} kbps[/cyan]")
    console.print(f"  Cutoff: [cyan]{cutoff} kbps[/cyan]")
    console.print(f"  Hardware Acceleration: [cyan]{not no_hw}[/cyan]")
    console.print(f"  Keep Intermediate Files: [cyan]{keep}[/cyan]")
    console.print(f"  Process All: [cyan]{process_all}[/cyan]")
    console.print(f"  Files to process: [cyan]{len(video_files)}[/cyan]")

    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        expand=True,
    )

    # Use Live to display both the progress and the conversion queue
    with Live(
        Group(
            progress,
            Panel(
                "Queue initializing...",
                title="Conversion Queue",
                border_style="magenta",
            ),
        ),
        refresh_per_second=4,
        console=console,
    ) as live:
        total_task = progress.add_task(
            "[cyan]Processing videos...", total=len(video_files)
        )
        for i, video in enumerate(video_files):
            file_task = progress.add_task(
                f"Converting {video.name}", total=100, start=False
            )
            progress.start_task(file_task)
            queue_display = get_queue_display(i, video_files, statuses)
            queue_display.append("")
            queue_display.append(f"Total space saved: {format_size(total_space_saved)}")
            queue_lines = "\n".join(queue_display)
            queue_panel = Panel(
                queue_lines, title="Conversion Queue", border_style="magenta"
            )
            live.update(Group(progress, queue_panel))
            status, space_saved = process_file(
                video, bitrate, cutoff, no_hw, keep, process_all, progress, file_task
            )
            total_space_saved += space_saved
            progress.remove_task(file_task)
            statuses[i] = status
            if status in (FFmpegResult.SUCCESS, FFmpegResult.SKIPPED):
                processed_files.append(video.name)
            else:
                errors += 1
            progress.update(total_task, advance=1)

    # Print summary
    console.print(f"\nProcessed {len(processed_files)} files successfully")
    console.print(f"Total space saved: {format_size(total_space_saved)}")
    if errors > 0:
        console.print(f"Encountered {errors} errors - see {ERROR_LOG} for details")
    else:
        if Path(ERROR_LOG).exists():
            Path(ERROR_LOG).unlink()


if __name__ == "__main__":
    main(auto_envvar_prefix="VIDCONV")
