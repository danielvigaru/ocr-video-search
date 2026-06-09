# ocr_video_search

Scans video files frame by frame (every N seconds), runs OCR on each frame, and stops the moment a keyword is found — reporting the exact timestamp and video it appeared in.

Useful for finding when a word, error message, or title card appears across a large set of recordings without watching them manually.

## Requirements

**Python packages**

```bash
pip install -r requirements.txt
```

**System tools**

```bash
# Ubuntu/Debian
sudo apt install ffmpeg tesseract-ocr

# macOS
brew install ffmpeg tesseract

# Windows
# ffmpeg:    https://ffmpeg.org/download.html
# Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
```

## Usage

```
python ocr_video_search.py <inputs> -k KEYWORD [options]
```

`<inputs>` can be any mix of video files and directories.

## Flags

| Flag                | Default      | Description                                        |
| ------------------- | ------------ | -------------------------------------------------- |
| `-k`, `--keyword`   | _(required)_ | Word or phrase to search for                       |
| `-i`, `--interval`  | `10.0`       | Seconds between sampled frames                     |
| `-r`, `--recursive` | off          | Also scan subdirectories when a directory is given |
| `--no-whole-word`   | off          | Match as a substring instead of a whole word       |
| `-v`, `--verbose`   | off          | Print progress for each frame as it's checked      |

## Examples

```bash
# Search a single file
python ocr_video_search.py lecture.mp4 -k "Chapter 3"

# Search all videos in a directory
python ocr_video_search.py /recordings/ -k "ERROR"

# Search a directory and all its subdirectories
python ocr_video_search.py /recordings/ -k "ERROR" --recursive

# Mix files and directories
python ocr_video_search.py intro.mp4 /recordings/season2/ finale.mkv -k "disclaimer"

# Sample every 5 seconds instead of 10
python ocr_video_search.py /recordings/ -k "WARNING" -i 5

# Match as substring — e.g. "err" matches "error", "stderr"
python ocr_video_search.py recording.mp4 -k "err" --no-whole-word

# Show per-frame progress
python ocr_video_search.py long_video.mp4 -k "password" -v
```

## Output

On a match, the script prints the video filename, timestamp, and an OCR text snippet, then exits:

```
[1/6] Scanning: recordings/session3.mp4

✓  KEYWORD FOUND — stopping search.

  Video     : recordings/session3.mp4
  Timestamp : 00:12:40.000 (760.0s)
  Frame #   : 76

  OCR snippet:
  ----------------------------------------
  FATAL ERROR: connection refused
  ----------------------------------------
```

If no match is found across all inputs, the script exits with code `1`.

## Notes

- Non-video files in directories are silently skipped. Detection is done via ffprobe (stream inspection), not file extension, so mixed directories are handled safely.
- Keyword matching is case-insensitive.
- Scanning stops at the **first** match — remaining videos in the queue are not processed.
- Frames are never written to disk; ffmpeg pipes them directly into memory.

## AI disclaimer

This script was created using Claude Sonnet
