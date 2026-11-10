"""Uniform interior sample selection and sentence-boundary capping for Sample & Compare.

Modeled after `src/core/sampling.py` in `hydropix/TranslateBooksWithLLMs`:
1. `select_sample_indices`: Uniformly distributes `n` samples across the interior `[1, T-2]`
   of the subtitle blocks (skipping opening and closing credits at `0` and `T-1`).
2. `cap_sample_block`: Caps a sample block's cues so that it never cuts off mid-sentence.
"""

from subllm.core.chunker import TERMINAL_PUNCT_RE
from subllm.core.models import SceneBlock, SubtitleCue


def select_sample_indices(total_blocks: int, n: int = 5) -> list[int]:
    """Pick up to `n` block indices distributed uniformly inside `(0, total_blocks - 1)`.

    Excludes the first block (`0`, often opening credits/title cards) and last block
    (`total_blocks - 1`, often closing credits) whenever `total_blocks > 2`.
    """
    if total_blocks <= 0 or n <= 0:
        return []
    if total_blocks <= 2:
        return list(range(min(total_blocks, n)))

    interior = list(range(1, total_blocks - 1))
    if n >= len(interior):
        return interior

    seen: set[int] = set()
    picked: list[int] = []
    for i in range(n):
        idx = round((i + 1) * (total_blocks - 1) / (n + 1))
        idx = max(1, min(total_blocks - 2, idx))
        if idx not in seen:
            seen.add(idx)
            picked.append(idx)
    return picked


def cap_sample_cues(cues: list[SubtitleCue], max_chars: int = 240) -> list[SubtitleCue]:
    """Select a prefix of `cues` up to `max_chars` without ever cutting mid-sentence.

    If the last included cue does not end with terminal punctuation (`.!?…`), continues
    including subsequent cues until a complete sentence boundary is reached.
    """
    if not cues or max_chars <= 0:
        return cues

    selected: list[SubtitleCue] = []
    char_count = 0
    for cue in cues:
        selected.append(cue)
        char_count += len(cue.clean_text)
        if char_count >= max_chars and TERMINAL_PUNCT_RE.search(cue.clean_text.strip()):
            break

    return selected


def extract_uniform_samples(
    blocks: list[SceneBlock], n: int = 5, max_chars: int = 240
) -> list[SceneBlock]:
    """Extract `n` evenly spaced interior `SceneBlock` samples with complete sentence endings."""
    indices = select_sample_indices(len(blocks), n=n)
    sampled_blocks: list[SceneBlock] = []
    for idx in indices:
        blk = blocks[idx]
        capped_cues = cap_sample_cues(blk.cues, max_chars=max_chars)
        sampled_blocks.append(
            SceneBlock(
                block_id=blk.block_id,
                cues=capped_cues,
                context_before=blk.context_before,
                context_after=blk.context_after,
            )
        )
    return sampled_blocks
