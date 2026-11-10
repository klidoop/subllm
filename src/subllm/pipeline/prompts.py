"""Prompt templates and parsers for the 3-Pass Cinema Subtitle Pipeline."""

import json
import re

from subllm.config import ProfileConfig
from subllm.core.models import MovieBible, SceneBlock, SubtitleCue

CUE_LINE_OUTPUT_RE = re.compile(r"^\[(?P<idx>\d+)\]\s*(?P<text>.*)$")


def build_bible_prompt(
    cues: list[SubtitleCue],
    source_lang: str,
    target_lang: str,
    max_sample_cues: int = 120,
) -> str:
    """Build Pass-1 prompt to extract MovieBible (genre, synopsis, character voices, glossary)."""
    # Evenly sample across the entire movie if there are more than max_sample_cues
    if len(cues) <= max_sample_cues:
        sampled = cues
    else:
        step = len(cues) / max_sample_cues
        sampled = [cues[int(i * step)] for i in range(max_sample_cues)]

    script_lines = []
    for c in sampled:
        spk = f"({c.speaker}) " if c.speaker else ""
        script_lines.append(f"[{c.index}] {spk}{c.clean_text}")
    script_excerpt = "\n".join(script_lines)

    return f"""You are a veteran cinema subtitle translator and script supervisor.
Analyze the following subtitle excerpt in {source_lang} and prepare a structured "Movie Bible" for translating into {target_lang}.

Return ONLY valid JSON matching this exact schema:
{{
  "genre_and_tone": "Brief genre, setting, and stylistic tone guide in {target_lang}",
  "synopsis": "2-3 sentence plot summary in {target_lang}",
  "characters": [
    {{
      "name": "Original name",
      "translated_name": "{target_lang} name",
      "role_and_tone": "Role, personality, and speaking style",
      "pronoun_and_address": "How they address others (e.g. polite vs casual pronouns)"
    }}
  ],
  "glossary": {{
    "Source Proper Noun or Term": "Canonical {target_lang} Translation"
  }}
}}

SUBTITLE EXCERPT:
{script_excerpt}
"""


def format_cue_for_block_prompt(cue: SubtitleCue, profile: ProfileConfig) -> str:
    """Format a single cue with physical timing metadata for Pass-2 LLM input."""
    budget = cue.max_budget_chars(target_cps=profile.max_cps, max_cpl=profile.max_cpl)
    meta_parts = [f"dur:{cue.duration_sec:.1f}s", f"max_chars:{budget}"]
    if cue.linked_group_id is not None:
        meta_parts.append(f"link:G{cue.linked_group_id}")
    if cue.gap_to_next_sec is not None and cue.gap_to_next_sec >= profile.scene_gap_sec:
        meta_parts.append(f"scene_gap_after:{cue.gap_to_next_sec:.1f}s")
    if cue.speaker:
        meta_parts.append(f"speaker:{cue.speaker}")

    meta_str = " | ".join(meta_parts)
    # Replace internal line breaks with a single space for prompt clarity
    one_line_text = " ".join(cue.clean_text.splitlines())
    return f"[{cue.index} | {meta_str}] {one_line_text}"


def build_block_translation_prompt(
    block: SceneBlock,
    source_lang: str,
    target_lang: str,
    profile: ProfileConfig,
    bible: MovieBible | None = None,
    retry_feedback: str = "",
) -> str:
    """Build Pass-2 time-aware block translation prompt with cross-cue alignment instructions."""
    bible_section = ""
    if bible:
        glossary_items = ", ".join(f'"{k}" -> "{v}"' for k, v in bible.glossary.items())
        char_items = "; ".join(
            f"{c.name} ({c.translated_name}): {c.role_and_tone}" for c in bible.characters
        )
        bible_section = f"""GLOBAL MOVIE BIBLE:
- Genre & Tone: {bible.genre_and_tone}
- Synopsis: {bible.synopsis}
- Characters: {char_items or "N/A"}
- Mandatory Glossary: {glossary_items or "N/A"}
"""

    ctx_before_str = "\n".join(f"  {line}" for line in block.context_before) or "  (Start of film)"
    ctx_after_str = "\n".join(f"  {line}" for line in block.context_after) or "  (End of film)"

    cue_lines = "\n".join(format_cue_for_block_prompt(c, profile) for c in block.cues)
    expected_ids = ", ".join(f"[{c.index}]" for c in block.cues)

    retry_banner = f"\nCRITICAL RETRY CORRECTION: {retry_feedback}\n" if retry_feedback else ""

    return f"""You are an elite cinema subtitle translator translating from {source_lang} to {target_lang}.
{retry_banner}
{bible_section}
TIMING & CROSS-CUE SEGMENTATION RULES (CRITICAL):
1. Every input line starts with `[ID | dur:... | max_chars:...]`.
2. **Self-Contained Sense Groups (`单屏意群独立，严禁跨屏破句`)**: Every single `[ID]` cue MUST be a grammatically self-contained breathing unit (a complete clause, or a complete verb-object / modifier-noun phrase).
   - NEVER start a cue with dangling particles like `的`, `地`, `得`, `了`, `着`.
   - NEVER split a verb and its short object across two cues, or mix the tail of one clause with the head of the next clause inside the same cue (e.g. avoid `"夜幕协议"，只因密码`).
3. **Audio-Visual Anchor Sync (`声画锚点同步顺译`)**: When cues share `link:G<N>` (a sentence split across multiple timestamps), keep numbers, proper nouns, and key entities in the SAME `[ID]` cue where they are spoken in the source audio. Anticipate inverted verbs/negations locally so each cue matches its own timestamp's content while flowing seamlessly into the next cue.
4. **Cinema Punctuation & Speed (`max_chars`)**: Omit trailing commas or periods at the end of a cue (keep only `?`, `!`, `…`), and keep each cue <= `max_chars` characters (and <= {profile.max_cpl} chars/line).
5. **Strict Output Format**: Output ONLY `[ID] translated text`, one cue per line, for all expected IDs: {expected_ids}. Do NOT drop, merge, or invent any `[ID]`.

CONTEXT BEFORE (for reference only, do NOT translate):
{ctx_before_str}

CUES TO TRANSLATE:
{cue_lines}

CONTEXT AFTER (for reference only, do NOT translate):
{ctx_after_str}

OUTPUT (strictly `[ID] translation` for each cue):
"""


def parse_block_translation_output(raw_output: str, expected_indices: list[int]) -> dict[int, str]:
    """Extract `[ID] translated text` pairs from LLM response."""
    results: dict[int, str] = {}
    expected_set = set(expected_indices)

    for raw_line in raw_output.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("```"):
            continue
        m = CUE_LINE_OUTPUT_RE.match(line)
        if m:
            idx = int(m.group("idx"))
            text = m.group("text").strip()
            # In case the LLM echoed back metadata like [101 | dur:1.5s] text
            if idx in expected_set and text:
                results[idx] = text
        else:
            # Handle accidental echo of `[101 | ...]`
            m_meta = re.match(r"^\[(?P<idx>\d+)\s*\|[^\]]+\]\s*(?P<text>.*)$", line)
            if m_meta:
                idx = int(m_meta.group("idx"))
                text = m_meta.group("text").strip()
                if idx in expected_set and text:
                    results[idx] = text

    return results


def build_condense_prompt(
    overflow_cues: list[SubtitleCue],
    target_lang: str,
    profile: ProfileConfig,
) -> str:
    """Build Pass-3 prompt to tighten translations that violated CPS or CPL limits."""
    lines = []
    for c in overflow_cues:
        budget = c.max_budget_chars(target_cps=profile.max_cps, max_cpl=profile.max_cpl)
        one_line_src = " ".join(c.clean_text.splitlines())
        one_line_draft = " ".join(c.translated_text.splitlines())
        lines.append(
            f"[{c.index} | dur:{c.duration_sec:.1f}s | current_cps:{c.cps} | target_max_chars:{budget}] "
            f"Source: {one_line_src} || Draft: {one_line_draft}"
        )
    joined_lines = "\n".join(lines)
    expected_ids = ", ".join(f"[{c.index}]" for c in overflow_cues)

    return f"""You are a cinema subtitle editor performing a reading-speed condensation pass in {target_lang}.
The following subtitle drafts are too long for their on-screen duration (CPS > {profile.max_cps} or line width > {profile.max_cpl}).

Rewrite each draft in idiomatic, spoken {target_lang} so that:
1. It preserves the core dramatic subtext and emotion.
2. Its character count is strictly <= `target_max_chars` (and <= {profile.max_cpl} chars per single line).
3. Output ONLY `[ID] condensed_translation` for: {expected_ids}.

OVERFLOWING CUES:
{joined_lines}

CONDENSED OUTPUT:
"""


def parse_bible_json(raw_output: str) -> MovieBible:
    """Safely parse Pass-1 LLM output into a validated MovieBible instance."""
    cleaned = raw_output.strip()
    # Strip markdown code fences if present
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    start_brace = cleaned.find("{")
    end_brace = cleaned.rfind("}")
    if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
        cleaned = cleaned[start_brace : end_brace + 1]

    try:
        data = json.loads(cleaned)
        return MovieBible.model_validate(data)
    except Exception:
        return MovieBible()
