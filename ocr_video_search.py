"""
video_ocr_search.py
-------------------
Extract a frame every N seconds from one or more video files or directories,
run OCR on each frame, and stop the moment a keyword is found.

Inputs can be any mix of:
  - individual video file paths
  - directories (scanned for video files; use --recursive for subdirectories)

Non-video files inside directories are silently skipped — ffprobe is used to
confirm each file actually contains a video stream before it is processed.

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
# Video file validation
# ---------------------------------------------------------------------------


def is_video_file(path: str) -> bool:
    """
    Return True if `path` contains at least one video stream, according to
    ffprobe.

    Why ffprobe instead of checking file extensions?
    -------------------------------------------------
    Extensions are unreliable: a `.mp4` could be audio-only, a `.dat` could
    be a video, and users often have mixed directories with images, subtitles,
    NFO files, and thumbnails sitting alongside actual videos.

    ffprobe reads the container's stream metadata — a fast operation that does
    not decode any media — and tells us definitively whether a video stream
    exists.  We ask it to count video streams (`select=v`) and check that the
    count is at least 1.

    The call is intentionally lightweight:
      -read_intervals "%+#1"  → read only the first packet's worth of data
      -show_streams           → emit stream metadata
      -select_streams v:0     → only the first video stream (if any)
    If ffprobe returns no output or exits non-zero, we treat the file as
    non-video and skip it without raising an exception.
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",  # suppress all info except errors
        "-select_streams",
        "v:0",  # look only at the first video stream
        "-show_entries",
        "stream=codec_type",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        "-read_intervals",
        "%+#1",  # read just enough to find stream info
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # ffprobe prints "video" if a video stream was found; nothing otherwise.
    return result.stdout.strip() == "video"


# ---------------------------------------------------------------------------
# Input resolution (files + directories → validated video list)
# ---------------------------------------------------------------------------


def resolve_inputs(raw_inputs: list[str], recursive: bool = False) -> list[Path]:
    """
    Accept a mixed list of file paths and directory paths; return an ordered
    list of Path objects that ffprobe has confirmed are video files.

    Processing rules:
    -----------------
    - A plain file path: validated with is_video_file(); kept or skipped.
    - A directory path: all files inside are collected (recursively if
      --recursive is set), sorted alphabetically, then each is validated.
    - Anything that doesn't exist: a warning is printed and it's skipped.

    Why sort directory contents?
    ----------------------------
    os.scandir / glob return files in filesystem order, which is essentially
    arbitrary on most systems.  Alphabetical order means episode01.mp4 is
    always scanned before episode02.mp4, making results reproducible.
    """
    resolved: list[Path] = []

    for raw in raw_inputs:
        p = Path(raw)

        if not p.exists():
            print(f"[SKIP] Not found: {p}")
            continue

        if p.is_file():
            # Single file — validate and add.
            if is_video_file(str(p)):
                resolved.append(p)
            else:
                print(f"[SKIP] No video stream detected: {p}")

        elif p.is_dir():
            # Directory — collect all files, sort, validate each.
            pattern = "**/*" if recursive else "*"
            candidates = sorted(f for f in p.glob(pattern) if f.is_file())

            if not candidates:
                print(f"[SKIP] Directory is empty: {p}")
                continue

            print(
                f"[DIR]  {p}  — found {len(candidates)} file(s), "
                "checking for video streams…"
            )

            valid_count = 0
            for candidate in candidates:
                if is_video_file(str(candidate)):
                    resolved.append(candidate)
                    valid_count += 1
                else:
                    # Print only in verbose-ish context; always skip silently
                    # to avoid spamming when a directory has many non-video files.
                    pass

            print(f"         {valid_count} video file(s) will be scanned.")

        else:
            # Symlinks to devices, sockets, etc. — just skip.
            print(f"[SKIP] Not a regular file or directory: {p}")

    return resolved


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
        description=(
            "OCR video frames every N seconds and stop when a keyword is found.\n"
            "Inputs can be individual video files, directories, or a mix of both."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Video file(s) and/or director(ies) to search.",
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
        "-r",
        "--recursive",
        action="store_true",
        help="When a directory is given, also search subdirectories.",
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

    # Expand directories and validate all inputs via ffprobe before scanning
    # anything.  This surfaces problems (empty dirs, no video files) upfront
    # rather than mid-run, and gives the user a clear picture of what will
    # actually be scanned.
    print("Resolving inputs…")
    video_paths = resolve_inputs(args.inputs, recursive=args.recursive)

    if not video_paths:
        sys.exit(
            "\n✗  No valid video files found in the provided inputs.\n"
            "   Make sure ffprobe can open the files and they contain a video stream."
        )

    print(f"\n{len(video_paths)} video(s) queued for scanning.\n")

    whole_word = not args.no_whole_word
    found = False

    for idx, video_path in enumerate(video_paths, start=1):
        print(f"[{idx}/{len(video_paths)}] Scanning: {video_path}")
        match = search_video(
            video_path=str(video_path),
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
            break  # stop processing remaining videos
        else:
            print(f"  Keyword not found.\n")

    if not found:
        print(
            f'\n✗  Keyword "{args.keyword}" was not found in any of the {len(video_paths)} scanned video(s).'
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
