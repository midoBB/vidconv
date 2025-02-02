#!/usr/bin/env python3
import shutil
import subprocess
from pathlib import Path

import click
from magic import Magic
from tqdm import tqdm

ERROR_LOG = "vidconv_errors.log"


def get_video_files(directory):
    """Get video files in directory using python-magic"""
    mime = Magic(mime=True)
    videos = []

    for file in directory.iterdir():
        if file.is_file():
            mimetype = mime.from_file(file)
            if mimetype.startswith("video/"):
                videos.append(file)

    return videos


def run_ffmpeg_hw(input_file, output_file, width, height, bitrate, framerate):
    """Run FFmpeg with hardware acceleration"""
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
        subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        log_error(input_file, e.stderr.decode())
        return False


def run_ffmpeg_sw(input_file, output_file, width, height, bitrate, framerate):
    """Run FFmpeg with software encoding"""
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
        subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        log_error(input_file, e.stderr.decode())
        return False


def get_video_info(file_path: Path) -> tuple[str, int, int, float]:
    """
    Retrieves video stream information using ffprobe.

    This function uses ffprobe to extract the codec name, width, height, and frame rate
    of the first video stream in the given file. It parses the output of ffprobe,
    handling cases where the frame rate is expressed as a fraction.

    Args:
        file_path (Path): The path to the video file.

    Returns:
        tuple[str, int, int, float]: A tuple containing the video codec name, width,
                                     height, and frame rate.

    Raises:
        RuntimeError: If ffprobe returns a non-zero exit code, indicating an error.
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,r_frame_rate",
        "-of",
        "csv=p=0",
        str(file_path),
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"FFprobe error: {result.stderr.decode()}")

    codec, width, height, framerate = result.stdout.decode().strip().split(",")
    if "/" in framerate:
        num, den = framerate.split("/")
        framerate = float(num) / float(den)
    else:
        framerate = float(framerate)
    return codec, int(width), int(height), framerate


def get_bitrate(file_path: Path) -> int:
    """
    Retrieves the video bitrate in kbps using ffprobe.

    Args:
        file_path (Path): The path to the video file.

    Returns:
        int: The video bitrate in kbps.

    Raises:
        RuntimeError: If ffprobe returns a non-zero exit code, indicating an error.
    """
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=bit_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(file_path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"FFprobe error: {result.stderr.decode()}")

    bitrate_bps = int(result.stdout.decode().strip())
    return bitrate_bps // 1000


def log_error(file_path, error_msg):
    """Log errors to the error log file"""
    with open(ERROR_LOG, "a") as f:
        f.write(f"Error processing file: {file_path}\n")
        f.write(f"Error message:\n{error_msg}\n")
        f.write("-" * 40 + "\n")


def process_file(
    file_path: Path,
    bitrate: int,
    cutoff: int,
    no_hw: bool,
    keep: bool,
    process_all: bool,
    progress_bar: tqdm,
) -> bool:
    """
    Processes a single video file, converting it to a 720p resolution.

    This function attempts to convert the input video file to a 720p resolution,
    using hardware acceleration if available and enabled. It handles both horizontal
    and vertical videos, ensuring the output maintains the correct aspect ratio.
    The function also manages the output file path, and can optionally keep or
    replace the original file.

    Args:
        file_path (Path): The path to the input video file.
        bitrate (int): The target bitrate for the output video in kbps.
        cutoff (int): The cutoff bitrate for the output video in kbps.
        no_hw (bool): If True, disables hardware acceleration.
        keep (bool): If True, keeps the intermediate output file; otherwise, replaces the original.
        process_all (bool): If True, we are processing all videos in the directory.
        progress_bar (tqdm): A tqdm progress bar object for displaying progress.


    Returns:
      bool: True if the processing was successful, False otherwise.
    """

    try:
        codec, width, height, framerate = get_video_info(file_path)
    except RuntimeError as e:
        progress_bar.write(f"Skipping {file_path.name}: {str(e)}")
        return False

    if process_all:
        try:
            file_bitrate = get_bitrate(file_path)
        except RuntimeError as e:
            progress_bar.write(f"Skipping {file_path.name}: {str(e)}")
            return False
        if file_bitrate < cutoff and codec not in ("hevc", "av1"):
            progress_bar.write(
                f"Skipping {file_path.name}: bitrate {file_bitrate} is less than cutoff {cutoff} and codec is {codec}"
            )
            return True

    # Determine output path
    output_path = file_path.with_stem(f"{file_path.stem}-720")
    output_path = output_path.with_suffix(".mp4")

    # Determine orientation
    if height > width:  # Vertical
        new_height = 720
        new_width = int(width * new_height / height)
        new_width += new_width % 2  # Ensure even width
    else:  # Horizontal
        new_width = 1280
        new_height = 720

    if framerate > 24:
        framerate = 24

    # Try hardware encoding first if enabled
    success = False
    if not no_hw and codec in ("hevc", "av1"):
        success = run_ffmpeg_hw(
            file_path, output_path, new_width, new_height, bitrate, framerate
        )
        if not success:
            progress_bar.write(
                f"Hardware encoding failed for {file_path.name}, trying software..."
            )

    if not success:
        success = run_ffmpeg_sw(
            file_path, output_path, new_width, new_height, bitrate, framerate
        )

    if success and not keep:
        shutil.move(output_path, file_path)

    return success


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
    if cutoff is None:
        cutoff = bitrate + 500
    # Initialize error log
    if Path(ERROR_LOG).exists():
        Path(ERROR_LOG).unlink()

    processed_files = []
    errors = 0

    if process_all:
        input_path = Path.cwd()
        video_files = get_video_files(input_path)

        if not video_files:
            click.echo("No video files found in current directory.")
            return
    elif input_path:
        video_files = [Path(file) for file in input_path]
    else:
        raise click.UsageError("Must specify INPUT_PATH or use --all")

    with tqdm(total=len(video_files), desc="Processing videos") as pbar:
        for video in video_files:
            pbar.set_postfix(file=video.name)
            result = process_file(
                video, bitrate, cutoff, no_hw, keep, process_all, pbar
            )
            if result:
                processed_files.append(video.name)
            else:
                errors += 1
            pbar.update(1)

    # Print summary
    click.echo(f"\nProcessed {len(processed_files)} files successfully")
    if errors > 0:
        click.echo(f"Encountered {errors} errors - see {ERROR_LOG} for details")
    else:
        if Path(ERROR_LOG).exists():
            Path(ERROR_LOG).unlink()


if __name__ == "__main__":
    main(auto_envvar_prefix="VIDCONV")
