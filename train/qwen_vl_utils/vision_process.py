from __future__ import annotations

import base64
import copy
import logging
import math
import os
import re
import sys
import time
import warnings
from functools import lru_cache
from io import BytesIO
from typing import Optional

import requests
import torch
import torchvision
from packaging import version
from PIL import Image, ImageDraw, ImageFont
import numpy as np 
from bisect import bisect_right
from pathlib import Path
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode


logger = logging.getLogger(__name__)

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = int(os.getenv("FPS_MAX_FRAMES", 32))
print(f"Setting FPS_MAX_FRAMES={FPS_MAX_FRAMES}")

# Toggle for timestamp writing (off by default)
WRITE_TIMESTAMPS_ON_FRAMES = os.getenv("WRITE_TIMESTAMPS_ON_FRAMES", "False").lower() == "true"
print(f"Setting WRITE_TIMESTAMPS_ON_FRAMES={WRITE_TIMESTAMPS_ON_FRAMES}")

# ===== Subtitles: toggle & style =====
WRITE_SUBTITLES_ON_FRAMES = os.getenv("WRITE_SUBTITLES_ON_FRAMES", "False").lower() == "true"
# Style knobs (kept as constants for simplicity; tune if desired)
SUBTITLE_FONT_SCALE = 0.06      # ~6% of min(H, W)
SUBTITLE_MIN_PX     = 20
SUBTITLE_MAX_PX     = 96
SUBTITLE_MAX_WIDTH_FRAC = 0.90  # wrap text to 90% of frame width
SUBTITLE_BOX_ALPHA  = 0.55      # semi-transparent black box
SUBTITLE_MARGIN_SCALE = 0.25    # margin ≈ 25% of font size
print(f"Setting WRITE_SUBTITLES_ON_FRAMES={WRITE_SUBTITLES_ON_FRAMES}")

# Set the maximum number of video token inputs.
# Here, 128K represents the maximum number of input tokens for the VLLM model.
# Remember to adjust it according to your own configuration.
VIDEO_TOTAL_PIXELS = int(float(os.environ.get('VIDEO_MAX_PIXELS', 128000 * 28 * 28 * 0.9)))
logger.info(f"set VIDEO_TOTAL_PIXELS: {VIDEO_TOTAL_PIXELS}")


def format_hhmmss(seconds: float) -> str:
    total = int(seconds)
    hh = total // 3600
    mm = (total % 3600) // 60
    ss = total % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def _choose_text_color_top_left(frame_hwc_uint8: np.ndarray) -> tuple[int, int, int]:
    """
    Decide text color based on top-left patch:
    - Use WHITE if background is dark or red-ish
    - Otherwise use RED
    """
    h, w = frame_hwc_uint8.shape[:2]
    ph, pw = min(40, h), min(200, w)  # small patch near the text
    patch = frame_hwc_uint8[:ph, :pw, :].astype(np.float32)

    r = patch[..., 0].mean()
    g = patch[..., 1].mean()
    b = patch[..., 2].mean()
    # Perceptual luminance
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b

    is_dark = luminance < 80.0
    is_red_bg = (r > 120.0) and (r - max(g, b) > 40.0)

    if is_dark or is_red_bg:
        return (255, 255, 255)  # white
    else:
        return (255, 0, 0)      # red


def _get_font_and_margin_for_frame(frame_hwc_uint8: np.ndarray) -> tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, int, int]:
    """
    Choose a font size based on frame size:
    - font_size ≈ 5% of the shorter side, clamped to [16, 72]
    - returns (font, margin_px, stroke_w)
    """
    h, w = frame_hwc_uint8.shape[:2]
    base = min(h, w)
    font_size = int(round(base * 0.05))  # 5% of min side
    font_size = max(16, min(72, font_size))

    # Try a built-in TTF shipped with Pillow; fall back to default bitmap font
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    margin = max(5, font_size // 4)
    stroke_w = max(1, font_size // 12)  # thin outline for legibility (very light)
    return font, margin, stroke_w


def draw_timestamp_top_left_hwc_uint8(frame_hwc_uint8: np.ndarray, text: str) -> np.ndarray:
    """
    Draw HH:MM:SS in the top-left corner of a uint8 HWC frame, using an adaptive font size.
    """
    img = Image.fromarray(frame_hwc_uint8, mode="RGB")
    draw = ImageDraw.Draw(img)

    color = _choose_text_color_top_left(frame_hwc_uint8)
    font, margin, stroke_w = _get_font_and_margin_for_frame(frame_hwc_uint8)

    # Use Pillow's stroke for subtle edge definition (keeps minimal code, better visibility).
    draw.text((margin, margin), text, fill=color, font=font, stroke_width=stroke_w, stroke_fill=(0, 0, 0))

    return np.array(img, copy=False)

# ===== Subtitles helpers =====

def _srt_time_to_seconds(ts: str) -> float:
    # "HH:MM:SS,mmm"
    hh, mm, rest = ts.split(":")
    ss, ms = rest.split(",")
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0


@lru_cache(maxsize=256)
def parse_srt_cached(path_str: str, mtime: float):
    """
    Parse .srt file into a list of cues: [(start, end, [lines]), ...]
    Cache keyed by (path, mtime) so updates bust cache automatically.
    """
    path = str(path_str)
    try:
        raw = open(path, "r", encoding="utf-8").read()
    except UnicodeDecodeError:
        raw = open(path, "r", encoding="utf-8-sig").read()
    # Normalize newlines
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")

    blocks = [b.strip() for b in raw.split("\n\n") if b.strip()]
    cues = []
    for b in blocks:
        lines = b.split("\n")
        if not lines:
            continue
        # Some SRTs have an index line first; timing is always on the next line containing '-->'
        timing_idx = -1
        for i, ln in enumerate(lines[:3]):  # usually at line 1 or 2
            if "-->" in ln:
                timing_idx = i
                break
        if timing_idx < 0:
            continue
        timing = lines[timing_idx].strip()
        try:
            left, right = [x.strip() for x in timing.split("-->")]
            start = _srt_time_to_seconds(left)
            end = _srt_time_to_seconds(right)
        except Exception:
            continue
        text_lines = [ln.strip() for ln in lines[timing_idx + 1:] if ln.strip()]
        # strip simple tags
        text_lines = [re.sub(r"</?([biu]|i|b|u)>", "", t, flags=re.IGNORECASE) for t in text_lines]
        if text_lines:
            cues.append((start, end, text_lines))
    # Sort just in case
    cues.sort(key=lambda x: x[0])
    return cues


class SubtitleIndex:
    """Fast monotonic-time subtitle lookup."""
    def __init__(self, cues):
        self.cues = cues
        self.starts = [c[0] for c in cues]  # start times

    def get(self, t: float, last_idx: int = 0, tol: float = 0.05):  # 50 ms
        if not self.cues:
            return None, last_idx
        i = bisect_right(self.starts, t + tol, lo=last_idx) - 1
        if 0 <= i < len(self.cues):
            s, e, lines = self.cues[i]
            if s - tol <= t < e + tol:
                return "\n".join(lines), i
            if t < s:
                j = max(0, i - 1)
                s2, e2, lines2 = self.cues[j]
                if s2 - tol <= t < e2 + tol:
                    return "\n".join(lines2), j
        # also catch the very first cue if t is just before it
        if i < 0 and self.cues and (self.cues[0][0] - t) <= tol:
            return "\n".join(self.cues[0][2]), 0
        
        return None, max(0, i)


def _measure_text(draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, text: str):
    # returns (width, height)
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=0)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _wrap_to_width(draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, text: str, max_width: int):
    # Greedy word-wrap, preserving explicit line breaks
    out_lines = []
    paras = text.split("\n")
    for para in paras:
        words = para.split()
        if not words:
            out_lines.append("")
            continue
        cur = words[0]
        for w in words[1:]:
            cand = cur + " " + w
            w_px, _ = _measure_text(draw, font, cand)
            if w_px <= max_width:
                cur = cand
            else:
                out_lines.append(cur)
                cur = w
        out_lines.append(cur)
    return out_lines


def _get_subtitle_font_and_metrics(frame_hwc_uint8: np.ndarray):
    h, w = frame_hwc_uint8.shape[:2]
    base = min(h, w)
    font_size = int(round(base * SUBTITLE_FONT_SCALE))
    font_size = max(SUBTITLE_MIN_PX, min(SUBTITLE_MAX_PX, font_size))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    margin = max(6, int(font_size * SUBTITLE_MARGIN_SCALE))
    stroke_w = max(1, font_size // 12)
    return font, margin, stroke_w, font_size


def draw_subtitle_bottom_center_hwc_uint8(frame_hwc_uint8: np.ndarray, text: str) -> np.ndarray:
    """
    Draw wrapped subtitle at bottom-center with semi-transparent box.
    """
    if not text:
        return frame_hwc_uint8

    # Work in RGBA to composite background box
    img = Image.fromarray(frame_hwc_uint8, mode="RGB").convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    draw = ImageDraw.Draw(img)  # will use for text drawing

    font, margin, stroke_w, font_size = _get_subtitle_font_and_metrics(frame_hwc_uint8)
    W, H = img.size
    max_w = int(W * SUBTITLE_MAX_WIDTH_FRAC)
    # Wrap text
    wrapped = _wrap_to_width(draw, font, text, max_w)

    # Measure block size
    line_hs = []
    line_ws = []
    for ln in wrapped:
        w_px, h_px = _measure_text(draw, font, ln if ln else " ")
        line_hs.append(h_px)
        line_ws.append(w_px)
    text_h = sum(line_hs) + int(0.2 * font_size) * max(0, len(wrapped) - 1)
    text_w = max(line_ws) if line_ws else 0

    # Position: bottom-center with margin
    box_left = max(0, (W - text_w) // 2 - margin)
    box_right = min(W, box_left + text_w + 2 * margin)
    box_bottom = H - margin
    box_top = max(0, box_bottom - text_h - 2 * margin)

    # Background box (semi-transparent black)
    alpha = int(255 * SUBTITLE_BOX_ALPHA)
    odraw.rectangle([box_left, box_top, box_right, box_bottom], fill=(0, 0, 0, alpha))

    # Composite overlay
    img = Image.alpha_composite(img, overlay)
    draw = ImageDraw.Draw(img)

    # Draw lines centered
    y = box_top + margin
    for i, ln in enumerate(wrapped):
        w_px, h_px = _measure_text(draw, font, ln if ln else " ")
        x = (W - w_px) // 2
        # white text with subtle black stroke for readability
        draw.text((x, y), ln, fill=(255, 255, 255), font=font, stroke_width=stroke_w, stroke_fill=(0, 0, 0))
        y += h_px + int(0.2 * font_size)

    return np.array(img.convert("RGB"), copy=False)


def _resolve_srt_path_for_video(video_path: str | Path) -> Optional[Path]:
    """
    Subtitles are stored next to the video, same basename, .srt extension.
    """
    srt = Path(video_path).with_suffix(".srt")
    return srt if srt.exists() else None

# ===== Subtitles helpers =====

def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def to_rgb(pil_image: Image.Image) -> Image.Image:
    if pil_image.mode == 'RGBA':
        white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
        white_background.paste(pil_image, mask=pil_image.split()[3])  # Use alpha channel as mask
        return white_background
    else:
        return pil_image.convert("RGB")


def fetch_image(ele: dict[str, str | Image.Image], size_factor: int = IMAGE_FACTOR) -> Image.Image:
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        # fix memory leak issue while using BytesIO
        with requests.get(image, stream=True) as response:
            response.raise_for_status()
            with BytesIO(response.content) as bio:
                image_obj = copy.deepcopy(Image.open(bio))
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            # fix memory leak issue while using BytesIO
            with BytesIO(data) as bio:
                image_obj = copy.deepcopy(Image.open(bio))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    image = to_rgb(image_obj)
    ## resize
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=size_factor,
        )
    else:
        width, height = image.size
        min_pixels = ele.get("min_pixels", MIN_PIXELS)
        max_pixels = ele.get("max_pixels", MAX_PIXELS)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    image = image.resize((resized_width, resized_height))

    return image


def smart_nframes(
    ele: dict,
    total_frames: int,
    video_fps: int | float,
) -> int:
    """calculate the number of frames for video used for model inputs.

    Args:
        ele (dict): a dict contains the configuration of video.
            support either `fps` or `nframes`:
                - nframes: the number of frames to extract for model inputs.
                - fps: the fps to extract frames for model inputs.
                    - min_frames: the minimum number of frames of the video, only used when fps is provided.
                    - max_frames: the maximum number of frames of the video, only used when fps is provided.
        total_frames (int): the original total number of frames of the video.
        video_fps (int | float): the original fps of the video.

    Raises:
        ValueError: nframes should in interval [FRAME_FACTOR, total_frames].

    Returns:
        int: the number of frames for video used for model inputs.
    """
    assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"
    if "nframes" in ele:
        nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
    else:
        fps = ele.get("fps", FPS)
        min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
        max_frames = floor_by_factor(ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR)
        nframes = total_frames / video_fps * fps
        if nframes > total_frames:
            logger.warning(f"smart_nframes: nframes[{nframes}] > total_frames[{total_frames}]")
        nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
        nframes = floor_by_factor(nframes, FRAME_FACTOR)
    if not (FRAME_FACTOR <= nframes and nframes <= total_frames):
        raise ValueError(f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}.")
    return nframes


def _read_video_torchvision(
    ele: dict,
) -> (torch.Tensor, float):
    """read video using torchvision.io.read_video

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    video_path = ele["video"]
    if version.parse(torchvision.__version__) < version.parse("0.19.0"):
        if "http://" in video_path or "https://" in video_path:
            warnings.warn("torchvision < 0.19.0 does not support http/https video path, please upgrade to 0.19.0.")
        if "file://" in video_path:
            video_path = video_path[7:]
    video, audio, info = io.read_video(
        video_path,
        start_pts=ele.get("video_start", 0.0),
        end_pts=ele.get("video_end", None),
        pts_unit="sec",
        output_format="TCHW",
    )
    total_frames, video_fps = video.size(0), info["video_fps"]
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(0, total_frames - 1, nframes).round().long()
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    video = video[idx]
    
    # ===== timestamp + subtitles in one pass to avoid extra conversions =====
    if WRITE_TIMESTAMPS_ON_FRAMES or WRITE_SUBTITLES_ON_FRAMES:
        clip_start = float(ele.get("video_start", 0.0))
        srt_path = ele["subtitles"] if "subtitles" in ele else _resolve_srt_path_for_video(video_path) if WRITE_SUBTITLES_ON_FRAMES else None
        subidx = None
        last = 0
        if srt_path is not None:
            try:
                cues = parse_srt_cached(str(srt_path), os.path.getmtime(srt_path))
                subidx = SubtitleIndex(cues) if cues else None
            except Exception as e:
                logger.warning(f"Subtitle overlay skipped for {video_path}: {e}")
                subidx = None
        elif WRITE_SUBTITLES_ON_FRAMES:
            logger.info(f"No .srt found for {video_path}; skipping subtitle overlay.")

        device = video.device
        dtype = video.dtype
        T, C, H, W = video.shape
        idx_list = idx.tolist()

        for k in range(T):
            # Convert CHW tensor -> HWC numpy uint8
            frame_chw = video[k].to("cpu")
            frame_hwc = frame_chw.permute(1, 2, 0).contiguous().numpy()
            frame_hwc = frame_hwc.astype(np.uint8)

            if WRITE_TIMESTAMPS_ON_FRAMES:
                t_abs = clip_start + (int(idx_list[k]) / float(video_fps))
                ts_text = format_hhmmss(t_abs)
                frame_hwc = draw_timestamp_top_left_hwc_uint8(frame_hwc, ts_text)

            if subidx is not None:
                t_abs = clip_start + (int(idx_list[k]) / float(video_fps))
                sub_text, last = subidx.get(t_abs, last)
                if sub_text:
                    frame_hwc = draw_subtitle_bottom_center_hwc_uint8(frame_hwc, sub_text)

            # Back to CHW torch (preserve dtype/device)
            frame_chw_new = torch.from_numpy(frame_hwc).permute(2, 0, 1)
            video[k] = frame_chw_new.to(dtype=dtype, device=device)
    # ===== timestamp + subtitles in one pass to avoid extra conversions

    return video, sample_fps


def is_decord_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("decord") is not None


def calculate_video_frame_range(
    ele: dict,
    total_frames: int,
    video_fps: float,
) -> tuple[int, int, int]:
    """
    Calculate the start and end frame indices based on the given time range.

    Args:
        ele (dict): A dictionary containing optional 'video_start' and 'video_end' keys (in seconds).
        total_frames (int): Total number of frames in the video.
        video_fps (float): Frames per second of the video.

    Returns:
        tuple: A tuple containing (start_frame, end_frame, frame_count).

    Raises:
        ValueError: If input parameters are invalid or the time range is inconsistent.
    """
    # Validate essential parameters
    if video_fps <= 0:
        raise ValueError("video_fps must be a positive number")
    if total_frames <= 0:
        raise ValueError("total_frames must be a positive integer")

    # Get start and end time in seconds
    video_start = ele.get("video_start", None)
    video_end = ele.get("video_end", None)
    if video_start is None and video_end is None:
        return 0, total_frames - 1, total_frames

    max_duration = total_frames / video_fps
    # Process start frame
    if video_start is not None:
        video_start_clamped = max(0.0, min(video_start, max_duration))
        start_frame = math.ceil(video_start_clamped * video_fps)
    else:
        start_frame = 0
    # Process end frame
    if video_end is not None:
        video_end_clamped = max(0.0, min(video_end, max_duration))
        end_frame = math.floor(video_end_clamped * video_fps)
        end_frame = min(end_frame, total_frames - 1)
    else:
        end_frame = total_frames - 1

    # Validate frame order
    if start_frame >= end_frame:
        raise ValueError(
            f"Invalid time range: Start frame {start_frame} (at {video_start_clamped if video_start is not None else 0}s) "
            f"exceeds end frame {end_frame} (at {video_end_clamped if video_end is not None else max_duration}s). "
            f"Video duration: {max_duration:.2f}s ({total_frames} frames @ {video_fps}fps)"
        )

    logger.info(f"calculate video frame range: {start_frame=}, {end_frame=}, {total_frames=} from {video_start=}, {video_end=}, {video_fps=:.3f}")
    return start_frame, end_frame, end_frame - start_frame + 1


def _read_video_decord(
    ele: dict,
) -> (torch.Tensor, float):
    """read video using decord.VideoReader

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    import decord
    video_path = ele["video"]
    vr = decord.VideoReader(video_path)
    total_frames, video_fps = len(vr), vr.get_avg_fps()
    start_frame, end_frame, total_frames = calculate_video_frame_range(
        ele,
        total_frames,
        video_fps,
    )
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(start_frame, end_frame, nframes).round().long().tolist()
    video = vr.get_batch(idx).asnumpy()   # (T, H, W, C) uint8

    # Fetch HWC frames as uint8 numpy and write timestamps if enabled
    if WRITE_TIMESTAMPS_ON_FRAMES:
        timestamps_sec = [i / float(video_fps) for i in idx]  # absolute timeline
        for k in range(len(idx)):
            text = format_hhmmss(timestamps_sec[k])
            video[k] = draw_timestamp_top_left_hwc_uint8(video[k], text)
    
    # ===== subtitles =====
    if WRITE_SUBTITLES_ON_FRAMES:
        # print(f"***********\nele: {ele}\n***********")
        srt_path = ele["subtitles"] if "subtitles" in ele else _resolve_srt_path_for_video(video_path)
        # print(f"***********\nUsing subtitles from: {srt_path}\n***********")
        if srt_path is not None:
            try:
                cues = parse_srt_cached(str(srt_path), os.path.getmtime(srt_path))
                subidx = SubtitleIndex(cues)
                last = 0
                for k, i_src in enumerate(idx):
                    t = float(i_src) / float(video_fps)  # absolute time
                    sub_text, last = subidx.get(t, last)
                    if sub_text:
                        video[k] = draw_subtitle_bottom_center_hwc_uint8(video[k], sub_text)
            except Exception as e:
                logger.warning(f"Subtitle overlay skipped for {video_path}: {e}")
        else:
            logger.info(f"No .srt found for {video_path}; skipping subtitle overlay.")
    # ===== subtitles =====
    
    video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    return video, sample_fps


def is_torchcodec_available() -> bool:
    """Check if torchcodec is available and properly installed."""
    try:
        import importlib.util
        if importlib.util.find_spec("torchcodec") is None:
            return False
        from torchcodec.decoders import VideoDecoder
        return True
    except (ImportError, AttributeError, Exception):
        return False


def _read_video_torchcodec(
    ele: dict,
) -> (torch.Tensor, float):
    """read video using torchcodec.decoders.VideoDecoder

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    from torchcodec.decoders import VideoDecoder
    TORCHCODEC_NUM_THREADS = int(os.environ.get('TORCHCODEC_NUM_THREADS', 8))
    logger.info(f"set TORCHCODEC_NUM_THREADS: {TORCHCODEC_NUM_THREADS}")
    video_path = ele["video"]
    decoder = VideoDecoder(video_path, num_ffmpeg_threads=TORCHCODEC_NUM_THREADS)
    video_fps = decoder.metadata.average_fps
    total_frames = decoder.metadata.num_frames
    start_frame, end_frame, total_frames = calculate_video_frame_range(
        ele,
        total_frames,
        video_fps,
    )
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(start_frame, end_frame, nframes).round().long().tolist()
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    
    # Obtain frames HWC uint8 and write timestamps if enabled
    video = decoder.get_frames_at(indices=idx).data  # typically numpy HWC uint8
    if isinstance(video, torch.Tensor):
        video_np = video.detach().cpu().numpy()
    else:
        video_np = video

    if WRITE_TIMESTAMPS_ON_FRAMES:
        timestamps_sec = [i / float(video_fps) for i in idx]
        for k in range(len(idx)):
            text = format_hhmmss(timestamps_sec[k])
            video_np[k] = draw_timestamp_top_left_hwc_uint8(video_np[k], text)

    # ===== subtitles =====
    if WRITE_SUBTITLES_ON_FRAMES:
        srt_path = ele["subtitles"] if "subtitles" in ele else _resolve_srt_path_for_video(video_path)
        if srt_path is not None:
            try:
                cues = parse_srt_cached(str(srt_path), os.path.getmtime(srt_path))
                subidx = SubtitleIndex(cues)
                last = 0
                for k, i_src in enumerate(idx):
                    t = float(i_src) / float(video_fps)
                    sub_text, last = subidx.get(t, last)
                    if sub_text:
                        video_np[k] = draw_subtitle_bottom_center_hwc_uint8(video_np[k], sub_text)
            except Exception as e:
                logger.warning(f"Subtitle overlay skipped for {video_path}: {e}")
        else:
            logger.info(f"No .srt found for {video_path}; skipping subtitle overlay.")
    # ===== subtitles =====

    # ensure tensor TCHW
    video_t = torch.tensor(video_np).permute(0, 3, 1, 2)  # TCHW
    
    return video_t, sample_fps


VIDEO_READER_BACKENDS = {
    "decord": _read_video_decord,
    "torchvision": _read_video_torchvision,
    "torchcodec": _read_video_torchcodec,
}

FORCE_QWENVL_VIDEO_READER = os.getenv("FORCE_QWENVL_VIDEO_READER", None)


@lru_cache(maxsize=1)
def get_video_reader_backend() -> str:
    if FORCE_QWENVL_VIDEO_READER is not None:
        video_reader_backend = FORCE_QWENVL_VIDEO_READER
    elif is_torchcodec_available():
        video_reader_backend = "torchcodec"
    elif is_decord_available():
        video_reader_backend = "decord"
    else:
        video_reader_backend = "torchvision"
    print(f"qwen-vl-utils using {video_reader_backend} to read video.", file=sys.stderr)
    return video_reader_backend


def fetch_video(ele: dict, image_factor: int = IMAGE_FACTOR, return_video_sample_fps: bool = False) -> torch.Tensor | list[Image.Image]:
    if isinstance(ele["video"], str):
        video_reader_backend = get_video_reader_backend()
        try:
            video, sample_fps = VIDEO_READER_BACKENDS[video_reader_backend](ele)
        except Exception as e:
            logger.warning(f"video_reader_backend {video_reader_backend} error, use torchvision as default, msg: {e}")
            video, sample_fps = VIDEO_READER_BACKENDS["torchvision"](ele)

        nframes, _, height, width = video.shape
        min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
        total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
        max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
        max_pixels_supposed = ele.get("max_pixels", max_pixels)
        if max_pixels_supposed > max_pixels:
            logger.warning(f"The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}].")
        max_pixels = min(max_pixels_supposed, max_pixels)
        if "resized_height" in ele and "resized_width" in ele:
            resized_height, resized_width = smart_resize(
                ele["resized_height"],
                ele["resized_width"],
                factor=image_factor,
            )
        else:
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=image_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        video = transforms.functional.resize(
            video,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).float()
        if return_video_sample_fps:
            return video, sample_fps
        return video
    else:
        assert isinstance(ele["video"], (list, tuple))
        process_info = ele.copy()
        process_info.pop("type", None)
        process_info.pop("video", None)
        images = [
            fetch_image({"image": video_element, **process_info}, size_factor=image_factor)
            for video_element in ele["video"]
        ]
        nframes = ceil_by_factor(len(images), FRAME_FACTOR)
        if len(images) < nframes:
            images.extend([images[-1]] * (nframes - len(images)))
        if return_video_sample_fps:
            return images, process_info.pop("fps", 2.0)
        return images


def extract_vision_info(conversations: list[dict] | list[list[dict]]) -> list[dict]:
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if (
                        "image" in ele
                        or "image_url" in ele
                        or "video" in ele
                        or ele.get("type","") in ("image", "image_url", "video")
                    ):
                        vision_infos.append(ele)
    return vision_infos


def process_vision_info(
    conversations: list[dict] | list[list[dict]],
    return_video_kwargs: bool = False,
) -> tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None, Optional[dict]]:

    vision_infos = extract_vision_info(conversations)
    ## Read images or videos
    image_inputs = []
    video_inputs = []
    video_sample_fps_list = []
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info))
        elif "video" in vision_info:
            video_input, video_sample_fps = fetch_video(vision_info, return_video_sample_fps=True)
            video_sample_fps_list.append(video_sample_fps)
            video_inputs.append(video_input)
        else:
            raise ValueError("image, image_url or video should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    if return_video_kwargs:
        return image_inputs, video_inputs, {'fps': video_sample_fps_list}
    return image_inputs, video_inputs
