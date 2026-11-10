"""Time-gap aware scene chunker and cross-cue split-sentence detector."""

import re

from subllm.config import ProfileConfig
from subllm.core.models import SceneBlock, SubtitleCue

# Terminal punctuation indicating that a sentence is grammatically complete at the end of a cue
TERMINAL_PUNCT_RE = re.compile(r'[.!?。！？…"\'”’♪\]\)]\s*$')


def is_split_sentence_continuation(
    curr_cue: SubtitleCue, next_cue: SubtitleCue, max_merge_gap_sec: float = 0.8
) -> bool:
    """Determine if `curr_cue` and `next_cue` form a single sentence split across two timecodes.

    Heuristics:
    1. Both cues contain actual dialogue (not pure sound effects like `[Radio static]`).
    2. If both cues have explicit speaker labels and they differ, they cannot be a continuation.
    3. The silence gap between `curr_cue` and `next_cue` is <= `max_merge_gap_sec`.
    4. `curr_cue.clean_text` does NOT end with terminal punctuation (., !, ?, etc.).
    """
    curr_text = curr_cue.clean_text.strip()
    next_text = next_cue.clean_text.strip()
    if not curr_text or not next_text:
        return False

    # Ignore bracketed sound effects, music cues, or all-caps/top-positioned title cards
    if (curr_text.startswith("[") and curr_text.endswith("]")) or "♪" in curr_text:
        return False
    if curr_text.isupper() or "\\an8" in curr_cue.leading_ass_tags:
        return False

    # Different speakers cannot be the same split sentence
    if curr_cue.speaker and next_cue.speaker and curr_cue.speaker != next_cue.speaker:
        return False

    gap = curr_cue.gap_to_next_sec
    if gap is None or gap > max_merge_gap_sec:
        return False

    return not bool(TERMINAL_PUNCT_RE.search(curr_text))


def annotate_linked_sentence_groups(
    cues: list[SubtitleCue],
    max_merge_gap_sec: float = 0.8,
    max_group_size: int = 4,
) -> list[SubtitleCue]:
    """Assign shared `linked_group_id` to consecutive cues that split a single sentence."""
    group_id = 0
    current_group_len = 0

    # Reset existing group IDs
    for c in cues:
        c.linked_group_id = None

    for i in range(len(cues) - 1):
        if is_split_sentence_continuation(
            cues[i], cues[i + 1], max_merge_gap_sec=max_merge_gap_sec
        ):
            if cues[i].linked_group_id is None or current_group_len >= max_group_size:
                group_id += 1
                cues[i].linked_group_id = group_id
                current_group_len = 1
            cues[i + 1].linked_group_id = cues[i].linked_group_id
            current_group_len += 1
        else:
            current_group_len = 0

    return cues


def merge_continuation_cues(
    cues: list[SubtitleCue],
    max_chars: int = 125,
    max_merge_gap_sec: float = 0.8,
) -> list[SubtitleCue]:
    """Physically merge split continuation cues into natural full-sentence cues.

    Inspired by `SRTProcessor.merge_multiline_subtitles` in `TranslateBooksWithLLMs`:
    When a sentence is split across consecutive cues (`is_split_sentence_continuation`)
    and their combined length is `<= max_chars`, merge the text and extend `end_ms` to
    the continuation cue's end time instead of forcing an inverted target sentence to be
    artificially chopped back into tiny fragments.
    """
    if not cues:
        return []

    merged: list[SubtitleCue] = [cues[0].model_copy(deep=True)]
    for next_cue in cues[1:]:
        curr = merged[-1]
        combined_clean = f"{curr.clean_text.strip()} {next_cue.clean_text.strip()}"
        if (
            len(combined_clean) <= max_chars
            and is_split_sentence_continuation(curr, next_cue, max_merge_gap_sec=max_merge_gap_sec)
        ):
            curr.raw_text = f"{curr.raw_text.strip()} {next_cue.raw_text.strip()}"
            curr.end_ms = next_cue.end_ms
            curr.gap_to_next_sec = next_cue.gap_to_next_sec
        else:
            merged.append(next_cue.model_copy(deep=True))

    # Re-index sequentially from 1..N after physical merging
    for idx, cue in enumerate(merged, start=1):
        cue.index = idx
    return merged


def build_scene_blocks(
    cues: list[SubtitleCue],
    profile: ProfileConfig,
    context_window: int = 2,
) -> list[SceneBlock]:
    """Partition cues into scene-coherent translation blocks without breaking linked sentences.

    Unlike naive fixed-size chunkers, this algorithm breaks blocks when:
    - A scene boundary is detected (`gap_to_next_sec >= profile.scene_gap_sec`) AND we have at least 3 cues, OR
    - We reach `profile.lines_per_block` AND the current cue is NOT in the middle of a `linked_group_id`.
    """
    if not cues:
        return []

    annotate_linked_sentence_groups(cues, max_merge_gap_sec=profile.merge_gap_sec)

    raw_batches: list[list[SubtitleCue]] = []
    current_batch: list[SubtitleCue] = []

    for i, cue in enumerate(cues):
        current_batch.append(cue)
        is_last = i == len(cues) - 1
        if is_last:
            break

        next_cue = cues[i + 1]
        # Never split right in the middle of a linked multi-cue sentence!
        in_same_linked_group = (
            cue.linked_group_id is not None and cue.linked_group_id == next_cue.linked_group_id
        )
        if in_same_linked_group:
            continue

        gap = cue.gap_to_next_sec or 0.0
        is_scene_break = gap >= profile.scene_gap_sec and len(current_batch) >= 2
        reached_max_lines = len(current_batch) >= profile.lines_per_block

        if is_scene_break or reached_max_lines:
            raw_batches.append(current_batch)
            current_batch = []

    if current_batch:
        raw_batches.append(current_batch)

    # Wrap into SceneBlock objects with sliding context_before and context_after
    blocks: list[SceneBlock] = []
    for block_idx, batch in enumerate(raw_batches):
        ctx_before: list[str] = []
        if block_idx > 0:
            prev_cues = raw_batches[block_idx - 1][-context_window:]
            ctx_before = [c.clean_text for c in prev_cues if c.clean_text]

        ctx_after: list[str] = []
        if block_idx < len(raw_batches) - 1:
            next_cues = raw_batches[block_idx + 1][:context_window]
            ctx_after = [c.clean_text for c in next_cues if c.clean_text]

        blocks.append(
            SceneBlock(
                block_id=block_idx,
                cues=batch,
                context_before=ctx_before,
                context_after=ctx_after,
            )
        )

    return blocks
