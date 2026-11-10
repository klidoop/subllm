"""Unified lossless subtitle parser for .srt, .vtt, and .ass/.ssa formats."""

import re
from pathlib import Path

from subllm.core.models import SubtitleCue

# SRT / VTT timestamp pattern: HH:MM:SS,mmm or HH:MM:SS.mmm (hours optional in VTT)
SRT_VTT_TIME_RE = re.compile(r"(?:(?P<h>\d{1,2}):)?(?P<m>\d{2}):(?P<s>\d{2})[,\.](?P<ms>\d{3})")
# ASS timestamp pattern: H:MM:SS.cc (centiseconds)
ASS_TIME_RE = re.compile(r"(?P<h>\d+):(?P<m>\d{2}):(?P<s>\d{2})\.(?P<cs>\d{2})")


def parse_timestamp_ms(ts_str: str) -> int:
    """Convert an SRT/VTT (HH:MM:SS,mmm) or ASS (H:MM:SS.cc) timestamp into milliseconds."""
    ts_str = ts_str.strip()
    m_srt = SRT_VTT_TIME_RE.match(ts_str)
    if m_srt:
        hours = int(m_srt.group("h") or 0)
        minutes = int(m_srt.group("m"))
        seconds = int(m_srt.group("s"))
        millis = int(m_srt.group("ms"))
        return ((hours * 3600) + (minutes * 60) + seconds) * 1000 + millis

    m_ass = ASS_TIME_RE.match(ts_str)
    if m_ass:
        hours = int(m_ass.group("h"))
        minutes = int(m_ass.group("m"))
        seconds = int(m_ass.group("s"))
        centis = int(m_ass.group("cs"))
        return ((hours * 3600) + (minutes * 60) + seconds) * 1000 + (centis * 10)

    raise ValueError(f"Unrecognized subtitle timestamp format: {ts_str!r}")


def format_srt_timestamp(ms: int) -> str:
    """Format milliseconds into standard SRT timestamp HH:MM:SS,mmm."""
    ms = max(0, ms)
    total_sec, millis = divmod(ms, 1000)
    hours, rem = divmod(total_sec, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def format_vtt_timestamp(ms: int) -> str:
    """Format milliseconds into standard WebVTT timestamp HH:MM:SS.mmm."""
    return format_srt_timestamp(ms).replace(",", ".")


def format_ass_timestamp(ms: int) -> str:
    """Format milliseconds into standard ASS timestamp H:MM:SS.cc (centiseconds)."""
    ms = max(0, ms)
    total_sec, millis = divmod(ms, 1000)
    centis = millis // 10
    hours, rem = divmod(total_sec, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}.{centis:02d}"


def _attach_inter_cue_gaps(cues: list[SubtitleCue]) -> list[SubtitleCue]:
    """Compute silence gap (in seconds) between each cue and the immediately following cue."""
    for i in range(len(cues) - 1):
        gap_ms = cues[i + 1].start_ms - cues[i].end_ms
        cues[i].gap_to_next_sec = round(max(0.0, gap_ms / 1000.0), 3)
    if cues:
        cues[-1].gap_to_next_sec = None
    return cues


def parse_srt_or_vtt(content: str) -> list[SubtitleCue]:
    """Parse .srt or .vtt content into a list of SubtitleCue objects with inter-cue gaps."""
    # Normalize BOM and line endings
    normalized = content.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    # Remove WEBVTT header block if present
    normalized = re.sub(r"^WEBVTT[^\n]*\n+", "", normalized, flags=re.IGNORECASE)

    blocks = re.split(r"\n\s*\n", normalized.strip())
    cues: list[SubtitleCue] = []

    for block in blocks:
        lines = [line.strip() for line in block.strip().split("\n") if line.strip()]
        if not lines or lines[0].startswith(("NOTE", "STYLE", "#")):
            continue

        # Locate the line containing the arrow '-->'
        arrow_line_idx = next((i for i, ln in enumerate(lines) if "-->" in ln), None)
        if arrow_line_idx is None:
            continue

        time_line = lines[arrow_line_idx]
        parts = time_line.split("-->")
        if len(parts) < 2:
            continue

        start_str = parts[0].strip().split()[0]
        end_str = parts[1].strip().split()[0]
        start_ms = parse_timestamp_ms(start_str)
        end_ms = parse_timestamp_ms(end_str)

        text_lines = lines[arrow_line_idx + 1 :]
        if not text_lines:
            continue

        raw_text = "\n".join(text_lines)
        cues.append(
            SubtitleCue(
                index=len(cues) + 1,
                start_ms=start_ms,
                end_ms=end_ms,
                raw_text=raw_text,
            )
        )

    return _attach_inter_cue_gaps(cues)


def parse_ass(content: str) -> list[SubtitleCue]:
    """Parse .ass / .ssa Dialogue lines into SubtitleCue objects, preserving speaker and tags."""
    normalized = content.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    cues: list[SubtitleCue] = []

    in_events = False
    format_fields: list[str] = [
        "Layer",
        "Start",
        "End",
        "Style",
        "Name",
        "MarginL",
        "MarginR",
        "MarginV",
        "Effect",
        "Text",
    ]

    for line in normalized.split("\n"):
        stripped = line.strip()
        if stripped.lower() == "[events]":
            in_events = True
            continue
        if in_events and stripped.startswith("[") and stripped.endswith("]"):
            break
        if not in_events:
            continue

        if stripped.lower().startswith("format:"):
            raw_fields = stripped.split(":", 1)[1].split(",")
            format_fields = [f.strip() for f in raw_fields]
            continue

        if stripped.startswith("Dialogue:"):
            payload = stripped.split(":", 1)[1].lstrip()
            # Split up to (len(format_fields) - 1) commas so commas inside Text are preserved!
            max_splits = len(format_fields) - 1
            parts = payload.split(",", max_splits)
            if len(parts) < len(format_fields):
                continue

            row = dict(zip(format_fields, parts, strict=False))
            start_ms = parse_timestamp_ms(row.get("Start", "0:00:00.00"))
            end_ms = parse_timestamp_ms(row.get("End", "0:00:00.00"))
            speaker = row.get("Name", "").strip()
            # Convert ASS soft/hard line breaks (\N or \n) into standard newline
            raw_text = row.get("Text", "").replace("\\N", "\n").replace("\\n", "\n")

            cues.append(
                SubtitleCue(
                    index=len(cues) + 1,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    raw_text=raw_text,
                    speaker=speaker,
                )
            )

    return _attach_inter_cue_gaps(cues)


def parse_subtitle_file(path: str | Path) -> list[SubtitleCue]:
    """Auto-detect subtitle format (.srt, .vtt, .ass, .ssa) and parse into SubtitleCue list."""
    file_path = Path(path)
    content = file_path.read_text(encoding="utf-8")
    ext = file_path.suffix.lower()

    if ext in {".ass", ".ssa"} or "[Events]" in content:
        return parse_ass(content)
    return parse_srt_or_vtt(content)
