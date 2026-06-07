"""
video_ocr_search.py
-------------------
Extract a frame every N seconds from one or more video files,
run OCR on each frame, and stop the moment a keyword is found.

Dependencies:
    pip install pillow pytesseract
    System: ffmpeg, tesseract-ocr

    Ubuntu/Debian:
        sudo apt install ffmpeg tesseract-ocr
    macOS (Homebrew):
        brew install ffmpeg tesseract
    Windows:
        - ffmpeg: https://ffmpeg.org/download.html
        - Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
"""

import subprocess
import sys
import io
import re
import argparse
from pathlib import Path

# Pillow and pytesseract are only needed at runtime, not import time,
# but we import them here so the user gets a clear error message early.
try:
    from PIL import Image
    import pytesseract
except ImportError:
    sys.exit(
        "Missing dependencies. Run:\n"
        "  pip install pillow pytesseract\n"
        "and make sure Tesseract is installed on your system."
    )


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------


def extract_frame_at(video_path: str, timestamp: float) -> Image.Image | None:
    """
    Ask ffmpeg to decode exactly one frame at `timestamp` seconds and return
    it as a Pillow Image — without writing anything to disk.

    Why pipe instead of saving files?
    ----------------------------------
    Saving tens of thousands of PNGs to disk is slow (I/O bound) and messy.
    Instead we tell ffmpeg to write a single PNG to stdout (`pipe:1`) and
    read it straight into memory. The round-trip is:

        video file → ffmpeg → raw PNG bytes → Python bytes buffer → PIL Image

    Why `image2` with `-vframes 1`?
    --------------------------------
    `-ss` (seek) before `-i` (input) is a *fast seek* — ffmpeg jumps near the
    target without decoding every prior frame.  Then `-vframes 1` captures
    exactly one frame and exits, so we never decode more than we need.

    Returns None if ffmpeg fails (e.g. timestamp beyond video length).
    """
    cmd = [
        "ffmpeg",
        "-ss",
        str(timestamp),  # fast-seek to timestamp
        "-i",
        video_path,  # input file
        "-vframes",
        "1",  # capture exactly one frame
        "-f",
        "image2",  # output format: single image
        "-vcodec",
        "png",  # lossless PNG keeps text sharp for OCR
        "pipe:1",  # write to stdout instead of a file
        "-loglevel",
        "error",  # suppress progress noise
    ]

    result = subprocess.run(cmd, capture_output=True)

    if result.returncode != 0 or not result.stdout:
        # The timestamp is likely past the end of the video — not an error.
        return None

    # Wrap the raw bytes in a BytesIO so PIL can read them like a file.
    return Image.open(io.BytesIO(result.stdout))


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------


def ocr_image(image: Image.Image) -> str:
    """
    Run Tesseract OCR on a PIL Image and return the recognised text.

    Preprocessing choices:
    ----------------------
    - Convert to greyscale: colour carries no information useful to Tesseract
      and processing one channel is faster than three.
    - `--psm 3`: "Fully automatic page segmentation" — the default, but
      explicit is better than implicit.  Use `--psm 6` if your frames are
      single uniform blocks of text (e.g. a title card).
    - `--oem 3`: Use the LSTM neural-net engine, which is more accurate than
      the legacy engine on most modern text.

    No thresholding or sharpening is applied here; Tesseract handles that
    internally. Add `image = image.filter(ImageFilter.SHARPEN)` before this
    call if you find accuracy poor on blurry frames.
    """
    grey = image.convert("L")  # L = 8-bit greyscale
    config = "--psm 3 --oem 3"
    return pytesseract.image_to_string(grey, config=config)


# ---------------------------------------------------------------------------
# Keyword matching
# ---------------------------------------------------------------------------


def text_contains_keyword(text: str, keyword: str, whole_word: bool) -> bool:
    """
    Return True if `text` contains `keyword` (case-insensitive).

    `whole_word=True` uses a regex word-boundary (`\b`) so that searching for
    "cat" does not match "concatenate".  This is on by default because OCR
    output sometimes runs words together, leading to false positives.
    """
    if whole_word:
        pattern = r"\b" + re.escape(keyword) + r"\b"
        return bool(re.search(pattern, text, re.IGNORECASE))
    return keyword.lower() in text.lower()


# ---------------------------------------------------------------------------
# Video duration
# ---------------------------------------------------------------------------


def get_video_duration(video_path: str) -> float:
    """
    Use ffprobe to find the total duration of the video in seconds.

    ffprobe is bundled with ffmpeg and is the canonical way to inspect
    container metadata without decoding any video.

    Returns 0.0 if the duration cannot be determined (e.g. live streams).
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


# ---------------------------------------------------------------------------
# Per-video search
# ---------------------------------------------------------------------------


def search_video(
    video_path: str,
    keyword: str,
    interval: float = 10.0,
    whole_word: bool = True,
    verbose: bool = False,
) -> dict | None:
    """
    Scan a single video for `keyword`, sampling one frame every `interval`
    seconds.

    Returns a dict with match details on success, or None if not found.

    Why a generator-style loop instead of building a list of timestamps?
    ----------------------------------------------------------------------
    We stop as soon as we find the keyword, so there's no point computing
    all timestamps up front.  A `while` loop with early `return` is clearer
    and more efficient than a list comprehension followed by a break.
    """
    duration = get_video_duration(video_path)
    if duration <= 0:
        print(
            f"  [!] Could not determine duration for {video_path}; "
            "will scan until frames run out."
        )

    timestamp = 0.0
    frame_number = 0

    while True:
        if verbose:
            print(f"  [{video_path}] Checking t={timestamp:.1f}s …", end="\r")

        frame = extract_frame_at(video_path, timestamp)

        if frame is None:
            # ffmpeg returned nothing → we've passed the end of the video.
            break

        text = ocr_image(frame)

        if text_contains_keyword(text, keyword, whole_word):
            return {
                "video": video_path,
                "timestamp_seconds": timestamp,
                "timestamp_human": format_timestamp(timestamp),
                "frame_number": frame_number,
                "ocr_text_snippet": text.strip(),
            }

        frame_number += 1
        timestamp += interval

        # If we know the duration, stop cleanly rather than waiting for a
        # None frame (saves one extra ffmpeg call at the very end).
        if duration > 0 and timestamp > duration:
            break

    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def format_timestamp(seconds: float) -> str:
    """Convert a float number of seconds to HH:MM:SS.mmm format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def check_dependencies() -> None:
    """Fail fast if ffmpeg/ffprobe/tesseract are not on PATH."""
    for tool in ("ffmpeg", "ffprobe", "tesseract"):
        result = subprocess.run(
            ["which", tool],  # use `where` on Windows
            capture_output=True,
        )
        if result.returncode != 0:
            sys.exit(
                f"Required tool not found on PATH: {tool}\n"
                "Install it and make sure it's accessible from your terminal."
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OCR video frames every N seconds and stop when a keyword is found.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "videos",
        nargs="+",
        help="One or more video file paths to search.",
    )
    parser.add_argument(
        "-k",
        "--keyword",
        required=True,
        help="The word or phrase to search for.",
    )
    parser.add_argument(
        "-i",
        "--interval",
        type=float,
        default=10.0,
        help="Seconds between sampled frames.",
    )
    parser.add_argument(
        "--no-whole-word",
        action="store_true",
        help="Match keyword as a substring, not a whole word.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print progress for each frame.",
    )
    args = parser.parse_args()

    check_dependencies()

    whole_word = not args.no_whole_word
    found = False

    for video_path in args.videos:
        if not Path(video_path).exists():
            print(f"[SKIP] File not found: {video_path}")
            continue

        print(f"\n[→] Scanning: {video_path}")
        match = search_video(
            video_path=video_path,
            keyword=args.keyword,
            interval=args.interval,
            whole_word=whole_word,
            verbose=args.verbose,
        )

        if match:
            print(f"\n✓  KEYWORD FOUND — stopping search.\n")
            print(f"  Video     : {match['video']}")
            print(
                f"  Timestamp : {match['timestamp_human']} "
                f"({match['timestamp_seconds']:.1f}s)"
            )
            print(f"  Frame #   : {match['frame_number']}")
            print(f"\n  OCR snippet:\n  {'-'*40}")
            for line in match["ocr_text_snippet"].splitlines():
                print(f"  {line}")
            print(f"  {'-'*40}")
            found = True
            break  # ← stop processing remaining videos
        else:
            print(f"  Keyword not found in {video_path}.")

    if not found:
        print(
            f'\n✗  Keyword "{args.keyword}" was not found in any of the provided videos.'
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
