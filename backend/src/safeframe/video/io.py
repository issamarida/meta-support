"""Video probing, decoding and encoding.

Decoding uses OpenCV (which applies rotation metadata, so frames come out the
way they are displayed). Encoding pipes raw frames into FFmpeg, which writes
H.264 at the source's exact frame rate and resolution and muxes the original
audio back in.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from safeframe.pipeline.types import VideoInfo

log = logging.getLogger(__name__)

# Audio codecs that can be stream-copied into an MP4 container unchanged.
MP4_COPYABLE_AUDIO = {"aac", "mp3", "alac", "ac3", "eac3", "opus", "flac"}


class VideoError(RuntimeError):
    """The input is not a readable video, or FFmpeg failed."""


def require_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise VideoError(f"{tool} not found on PATH; install FFmpeg")


def _parse_rate(rate: str | None) -> Fraction | None:
    if not rate or rate in {"0/0", "0"}:
        return None
    try:
        value = Fraction(rate)
    except (ValueError, ZeroDivisionError):
        return None
    return value if value > 0 else None


def probe(path: Path) -> VideoInfo:
    """Read stream metadata with ffprobe. Raises VideoError for non-video input."""
    require_ffmpeg()
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise VideoError(f"ffprobe could not read the file: {result.stderr.strip()[:300]}")
    data = json.loads(result.stdout or "{}")
    streams = data.get("streams", [])
    video = next(
        (
            s
            for s in streams
            if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")
        ),
        None,
    )
    if video is None:
        raise VideoError("no video stream found")
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    rate = _parse_rate(video.get("avg_frame_rate")) or _parse_rate(video.get("r_frame_rate"))
    if rate is None:
        raise VideoError("could not determine frame rate")
    # Prefer r_frame_rate when it agrees with the average (exact NTSC rationals).
    r_rate = _parse_rate(video.get("r_frame_rate"))
    if r_rate is not None and abs(float(r_rate) - float(rate)) < 0.01:
        rate = r_rate
    rate = rate.limit_denominator(1001 * 1000)

    duration = float(video.get("duration") or data.get("format", {}).get("duration") or 0.0)
    nb_frames = int(video.get("nb_frames") or 0)
    frame_count = nb_frames or round(duration * float(rate))

    width, height = int(video["width"]), int(video["height"])
    if _rotation(video) in (90, 270):
        width, height = height, width  # decoders auto-rotate, so report display size

    return VideoInfo(
        width=width,
        height=height,
        fps=float(rate),
        fps_rational=f"{rate.numerator}/{rate.denominator}",
        frame_count=max(frame_count, 0),
        duration=duration,
        has_audio=audio is not None,
        audio_codec=audio.get("codec_name") if audio else None,
    )


def _rotation(stream: dict) -> int:
    rotate = stream.get("tags", {}).get("rotate")
    if rotate is None:
        for side in stream.get("side_data_list", []):
            if "rotation" in side:
                rotate = side["rotation"]
                break
    try:
        return int(float(rotate or 0)) % 360
    except ValueError:
        return 0


def iter_frames(path: Path) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_index, BGR frame) in display orientation."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoError(f"OpenCV could not open {path.name}")
    try:
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield index, frame
            index += 1
    finally:
        cap.release()


def read_frame(path: Path, index: int) -> np.ndarray | None:
    """Random access to a single frame (used for thumbnails)."""
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


class VideoWriter:
    """Encode BGR frames to MP4 with FFmpeg, muxing audio from ``audio_source``.

    Output keeps the source resolution and exact frame rate. Container metadata
    (which can include GPS location and device identifiers) is deliberately
    *not* copied: the output is meant to be shared.
    """

    def __init__(
        self,
        output: Path,
        info: VideoInfo,
        audio_source: Path | None,
        crf: int = 18,
        preset: str = "medium",
    ) -> None:
        require_ffmpeg()
        self.output = output
        self.info = info
        w, h = info.width, info.height
        # yuv420p is the most compatible, but needs even dimensions.
        pix_fmt = "yuv420p" if w % 2 == 0 and h % 2 == 0 else "yuv444p"
        if pix_fmt != "yuv420p":
            log.warning("odd frame size %dx%d: encoding yuv444p to keep exact resolution", w, h)

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{w}x{h}",
            "-framerate",
            info.fps_rational,
            "-i",
            "pipe:0",
        ]
        with_audio = audio_source is not None and info.has_audio
        if with_audio:
            cmd += ["-i", str(audio_source), "-map", "0:v:0", "-map", "1:a?"]
            if (info.audio_codec or "").lower() in MP4_COPYABLE_AUDIO:
                cmd += ["-c:a", "copy"]
            else:
                cmd += ["-c:a", "aac", "-b:a", "192k"]
        cmd += [
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            pix_fmt,
            "-r",
            info.fps_rational,
            "-map_metadata",
            "-1",
            "-map_chapters",
            "-1",
            "-movflags",
            "+faststart",
            str(output),
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self._frames = 0

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[:2] != (self.info.height, self.info.width):
            raise VideoError(
                f"frame size {frame.shape[1]}x{frame.shape[0]} != "
                f"{self.info.width}x{self.info.height}"
            )
        assert self._proc.stdin is not None
        try:
            self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError as exc:
            raise VideoError(f"ffmpeg exited early: {self._stderr()}") from exc
        self._frames += 1

    def close(self) -> None:
        assert self._proc.stdin is not None
        if not self._proc.stdin.closed:
            self._proc.stdin.close()
        code = self._proc.wait()
        if code != 0:
            raise VideoError(f"ffmpeg failed ({code}): {self._stderr()}")

    def abort(self) -> None:
        self._proc.kill()
        self._proc.wait()
        self.output.unlink(missing_ok=True)

    def _stderr(self) -> str:
        assert self._proc.stderr is not None
        return self._proc.stderr.read().decode(errors="replace").strip()[-500:]

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


def make_preview(source: Path, output: Path, info: VideoInfo, max_height: int = 720) -> None:
    """Write a small, browser-playable H.264 proxy for the review screen.

    Frame timing matches the source (constant ``fps_rational``) so overlay
    frame indices line up with ``currentTime * fps`` in the browser.
    """
    require_ffmpeg()
    scale = (
        f"scale=-2:'min({max_height},ih)'"
        if info.height > max_height
        else "scale=trunc(iw/2)*2:trunc(ih/2)*2"
    )
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vf",
        scale,
        "-r",
        info.fps_rational,
        "-fps_mode",
        "cfr",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "26",
        "-pix_fmt",
        "yuv420p",
        "-an",
        "-map_metadata",
        "-1",
        "-movflags",
        "+faststart",
        str(output),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise VideoError(f"preview encode failed: {result.stderr.strip()[-300:]}")
