"""Orchestrator for the 3-Pass Cinema Subtitle Translation Pipeline."""

import hashlib
from collections.abc import Callable

from subllm.config import ProfileConfig
from subllm.core.chunker import annotate_linked_sentence_groups, build_scene_blocks
from subllm.core.models import MovieBible, SceneBlock, SubtitleCue
from subllm.llm.client import CheckpointStore, LLMProviderClient
from subllm.pipeline.prompts import (
    build_bible_prompt,
    build_block_translation_prompt,
    build_condense_prompt,
    parse_bible_json,
    parse_block_translation_output,
)


def compute_job_key(
    cues: list[SubtitleCue], source_lang: str, target_lang: str, profile: ProfileConfig
) -> str:
    """Compute a deterministic hash key for SQLite checkpointing."""
    fingerprint = "|".join(f"{c.index}:{c.start_ms}-{c.end_ms}:{c.clean_text}" for c in cues[:50])
    raw = f"{len(cues)}:{source_lang}->{target_lang}:{profile.name}:{fingerprint}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


async def extract_movie_bible(
    cues: list[SubtitleCue],
    source_lang: str,
    target_lang: str,
    llm: LLMProviderClient,
    checkpoint_store: CheckpointStore | None = None,
    job_key: str = "",
) -> MovieBible:
    """Pass 1: Extract global MovieBible (synopsis, characters, glossary) from the script."""
    if checkpoint_store and job_key:
        cached = checkpoint_store.load_bible(job_key)
        if cached:
            return parse_bible_json(cached)

    prompt = build_bible_prompt(cues, source_lang=source_lang, target_lang=target_lang)
    raw_resp = await llm.complete(prompt, temperature=0.2)
    bible = parse_bible_json(raw_resp)

    if checkpoint_store and job_key:
        checkpoint_store.save_bible(job_key, bible.model_dump_json())

    return bible


def _split_at_clause_boundary(text: str, target_len: int) -> tuple[str, str]:
    """Split `text` near `target_len` at a natural clause/punctuation boundary without dangling particles."""
    if len(text) <= 2:
        return text, ""
    target = max(1, min(len(text) - 1, target_len))
    # Search within a +-5 char window around target for a natural clause break
    best_cut = target
    best_dist = 999
    for pos in range(max(1, target - 5), min(len(text), target + 6)):
        if text[pos - 1] in "，、；：。！？ ":
            dist = abs(pos - target)
            if dist < best_dist:
                best_dist = dist
                best_cut = pos
    # Never leave a dangling particle ('的', '地', '得', '了', '着') at the start of the second part
    while best_cut < len(text) and text[best_cut] in "的地得了着过，、。；":
        best_cut += 1
    head = text[:best_cut].rstrip("，、；。 ").strip()
    tail = text[best_cut:].lstrip("，、；。 ").strip()
    return (head or text), tail


def _detect_broken_segmentation(parsed: dict[int, str]) -> list[int]:
    """Return cue indices that start with a dangling particle (e.g. '的...') indicating broken enjambment."""
    broken: list[int] = []
    for idx, text in parsed.items():
        stripped = text.strip()
        if len(stripped) >= 2 and stripped[0] in "的地得":
            broken.append(idx)
    return broken


def _repair_missing_linked_cues(
    block: SceneBlock,
    parsed: dict[int, str],
) -> dict[int, str]:
    """If the LLM merged a linked group into its first cue, split at natural syntactic boundaries."""
    expected_indices = [c.index for c in block.cues]
    missing = [idx for idx in expected_indices if idx not in parsed]
    if not missing:
        return parsed

    repaired = dict(parsed)
    groups: dict[int, list[SubtitleCue]] = {}
    for c in block.cues:
        if c.linked_group_id is not None:
            groups.setdefault(c.linked_group_id, []).append(c)

    for _, group_cues in groups.items():
        present = [c for c in group_cues if c.index in repaired]
        absent = [c for c in group_cues if c.index not in repaired]
        if len(present) == 1 and absent:
            remaining = repaired[present[0].index]
            for idx_in_grp, c in enumerate(group_cues):
                if idx_in_grp == len(group_cues) - 1 or not remaining:
                    repaired[c.index] = remaining.strip() or repaired[present[0].index]
                else:
                    rem_dur = sum(gc.duration_sec for gc in group_cues[idx_in_grp:])
                    ratio = c.duration_sec / max(0.1, rem_dur)
                    target_len = max(1, int(round(len(remaining) * ratio)))
                    head, remaining = _split_at_clause_boundary(remaining, target_len)
                    repaired[c.index] = head

    return repaired


async def translate_scene_block(
    block: SceneBlock,
    source_lang: str,
    target_lang: str,
    profile: ProfileConfig,
    llm: LLMProviderClient,
    bible: MovieBible | None = None,
    max_marker_retries: int = 2,
) -> dict[int, str]:
    """Pass 2: Translate a single scene block with `[ID]` marker & sense-group segmentation validation."""
    expected_indices = [c.index for c in block.cues if c.clean_text.strip()]
    if not expected_indices:
        return {c.index: "" for c in block.cues}

    feedback = ""
    best_parsed: dict[int, str] = {}

    for _attempt in range(max_marker_retries + 1):
        prompt = build_block_translation_prompt(
            block=block,
            source_lang=source_lang,
            target_lang=target_lang,
            profile=profile,
            bible=bible,
            retry_feedback=feedback,
        )
        raw_out = await llm.complete(prompt, temperature=0.2)
        parsed = parse_block_translation_output(raw_out, expected_indices=expected_indices)
        parsed = _repair_missing_linked_cues(block, parsed)

        if len(parsed) > len(best_parsed):
            best_parsed = parsed

        missing = [idx for idx in expected_indices if idx not in parsed]
        broken = _detect_broken_segmentation(parsed)
        if not missing and not broken:
            return parsed

        issues: list[str] = []
        if missing:
            missing_str = ", ".join(f"[{m}]" for m in missing)
            issues.append(f"omitted required cue markers: {missing_str}")
        if broken:
            broken_str = ", ".join(f"[{b}]" for b in broken)
            issues.append(
                f"cues {broken_str} start with a dangling particle ('的/地/得'); "
                "each cue must be a self-contained phrase"
            )
        feedback = "Your previous response had segmentation issues: " + "; ".join(issues) + "."

    # Fallback for any stubbornly missing cue: preserve clean_text
    for c in block.cues:
        if c.index not in best_parsed:
            best_parsed[c.index] = c.clean_text

    return best_parsed


async def condense_overflow_cues(
    cues: list[SubtitleCue],
    target_lang: str,
    profile: ProfileConfig,
    llm: LLMProviderClient,
    batch_size: int = 12,
) -> int:
    """Pass 3: Identify cues violating CPS/CPL and rewrite them to fit reading-speed budgets."""
    overflowing = [
        c
        for c in cues
        if c.translated_text.strip() and (c.cps > profile.max_cps or c.cpl > profile.max_cpl)
    ]
    if not overflowing:
        return 0

    condensed_count = 0
    cue_by_idx = {c.index: c for c in overflowing}

    for i in range(0, len(overflowing), batch_size):
        batch = overflowing[i : i + batch_size]
        expected_ids = [c.index for c in batch]
        prompt = build_condense_prompt(batch, target_lang=target_lang, profile=profile)
        raw_out = await llm.complete(prompt, temperature=0.1)
        parsed = parse_block_translation_output(raw_out, expected_indices=expected_ids)

        for idx, shorter_text in parsed.items():
            cue = cue_by_idx[idx]
            # Accept the condensed version if it is genuinely shorter or improves CPS/CPL
            if len(shorter_text) <= len(cue.translated_text):
                cue.condensed_text = shorter_text
                condensed_count += 1

    return condensed_count


def select_test_sample_cues(
    cues: list[SubtitleCue],
    sample_size: int = 5,
    start_index: int | None = None,
    profile: ProfileConfig | None = None,
) -> list[SubtitleCue]:
    """Select a contiguous sample of cues for pre-flight translation quality inspection.

    - Never slices a multi-cue split sentence (`linked_group_id`) in half.
    - When `start_index` is None (`smart` mode), picks the most diagnostic window containing
      cross-cue split sentences and dense dialogue.
    """
    if not cues:
        return []

    max_link_gap = profile.merge_gap_sec if profile else 0.8
    annotate_linked_sentence_groups(cues, max_merge_gap_sec=max_link_gap)

    if len(cues) <= sample_size:
        return list(cues)

    if start_index is not None:
        start_pos = next((i for i, c in enumerate(cues) if c.index >= start_index), 0)
    else:
        # Smart mode: score each window by linked cues + character density
        best_pos = 0
        best_score = -1.0
        for pos in range(0, len(cues) - sample_size + 1):
            window = cues[pos : pos + sample_size]
            linked_bonus = sum(15.0 for c in window if c.linked_group_id is not None)
            char_score = sum(len(c.clean_text) / max(0.8, c.duration_sec) for c in window)
            # Avoid starting right inside the middle of a linked group
            starts_mid_group = (
                pos > 0
                and window[0].linked_group_id is not None
                and cues[pos - 1].linked_group_id == window[0].linked_group_id
            )
            penalty = 50.0 if starts_mid_group else 0.0
            score = linked_bonus + char_score - penalty
            if score > best_score:
                best_score = score
                best_pos = pos
        start_pos = best_pos

    # Expand backwards if start_pos lands mid-linked-group
    while (
        start_pos > 0
        and cues[start_pos].linked_group_id is not None
        and cues[start_pos - 1].linked_group_id == cues[start_pos].linked_group_id
    ):
        start_pos -= 1

    end_pos = min(len(cues), start_pos + sample_size)
    # Expand forwards if end_pos cuts a linked group in half
    while (
        end_pos < len(cues)
        and cues[end_pos - 1].linked_group_id is not None
        and cues[end_pos].linked_group_id == cues[end_pos - 1].linked_group_id
    ):
        end_pos += 1

    return cues[start_pos:end_pos]


async def run_translation_pipeline(
    cues: list[SubtitleCue],
    source_lang: str,
    target_lang: str,
    profile: ProfileConfig,
    llm: LLMProviderClient,
    checkpoint_store: CheckpointStore | None = None,
    existing_bible: MovieBible | None = None,
    progress_callback: Callable[[str, float], None] | None = None,
    sample_size: int = 0,
    sample_start: int | None = None,
    merge_continuations: bool = False,
) -> tuple[list[SubtitleCue], MovieBible, list[SceneBlock]]:
    """Execute the 3-Pass Cinema Subtitle Pipeline (full movie or pre-flight test sample)."""
    from subllm.core.chunker import merge_continuation_cues

    working_cues = (
        merge_continuation_cues(cues, max_chars=125, max_merge_gap_sec=profile.merge_gap_sec)
        if merge_continuations
        else cues
    )
    job_key = compute_job_key(working_cues, source_lang, target_lang, profile)

    # Pass 1: Global Movie Bible (always scans full movie cues so terminology is complete)
    if existing_bible is not None:
        bible = existing_bible
    elif profile.enable_bible_pass:
        if progress_callback:
            progress_callback(
                "Pass 1/3: 正在分析全片剧情基调、人物口吻与术语表 (Movie Bible)...", 0.08
            )
        bible = await extract_movie_bible(
            cues=working_cues,
            source_lang=source_lang,
            target_lang=target_lang,
            llm=llm,
            checkpoint_store=checkpoint_store,
            job_key=job_key,
        )
    else:
        bible = MovieBible()

    # If running a pre-flight test sample, slice target cues after MovieBible extraction
    active_cues = (
        select_test_sample_cues(
            working_cues, sample_size=sample_size, start_index=sample_start, profile=profile
        )
        if sample_size > 0
        else working_cues
    )

    # Scene-aware chunking
    blocks = build_scene_blocks(active_cues, profile=profile)
    cached_blocks = (
        checkpoint_store.load_blocks(job_key) if (checkpoint_store and sample_size == 0) else {}
    )
    cue_map = {c.index: c for c in active_cues}

    # Pass 2: Time-Aware Block Translation
    total_blocks = max(1, len(blocks))
    for idx, block in enumerate(blocks):
        if progress_callback:
            frac = 0.15 + 0.65 * ((idx + 1) / total_blocks)
            progress_callback(
                f"Pass 2/3: 正在翻译场景分块 {idx + 1}/{total_blocks} (含跨时间轴断句对齐)...",
                frac,
            )

        if block.block_id in cached_blocks:
            block_trans = cached_blocks[block.block_id]
        else:
            block_trans = await translate_scene_block(
                block=block,
                source_lang=source_lang,
                target_lang=target_lang,
                profile=profile,
                llm=llm,
                bible=bible,
            )
            if checkpoint_store and sample_size == 0:
                checkpoint_store.save_block(job_key, block.block_id, block_trans)

        for cue_idx, text in block_trans.items():
            if cue_idx in cue_map:
                cue_map[cue_idx].translated_text = text

    # Pass 3: CPS/CPL Quality Gate & Auto-Condensation
    if profile.enable_condense_pass:
        if progress_callback:
            progress_callback("Pass 3/3: 正在检测 CPS/CPL 超标字幕并执行紧凑化重写...", 0.90)
        await condense_overflow_cues(
            cues=active_cues,
            target_lang=target_lang,
            profile=profile,
            llm=llm,
        )

    if progress_callback:
        progress_callback("翻译流水线已完成！", 1.0)

    return active_cues, bible, blocks
