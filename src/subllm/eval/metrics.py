"""Quantitative subtitle quality evaluation (CPS, CPL, Marker Alignment, Glossary Consistency)."""

import math

from pydantic import BaseModel, Field

from subllm.config import ProfileConfig
from subllm.core.models import MovieBible, SubtitleCue


class EvalReport(BaseModel):
    """Comprehensive, honest benchmark report measuring subtitle quality before vs. after Pass 3."""

    total_cues: int
    translated_cues: int
    alignment_recall_pct: float = Field(
        description="Percentage of cues with valid aligned translations"
    )
    linked_cues_count: int = Field(
        description="Number of cues belonging to cross-cue split sentences"
    )

    # Initial Pass-2 Draft Timing Metrics
    draft_cps_mean: float
    draft_cps_p50: float
    draft_cps_p95: float
    draft_cps_max: float
    draft_cps_violation_rate_pct: float
    draft_cpl_violation_rate_pct: float

    # Final Post-Pass-3 Timing Metrics
    final_cps_mean: float
    final_cps_p50: float
    final_cps_p95: float
    final_cps_max: float
    final_cps_violation_rate_pct: float
    final_cpl_violation_rate_pct: float
    condensed_cues_count: int

    # Terminology Consistency
    glossary_checks: int
    glossary_hits: int
    glossary_hit_rate_pct: float


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Compute percentile `q` in [0, 100] from a pre-sorted list."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return round(sorted_vals[0], 2)
    pos = (q / 100.0) * (len(sorted_vals) - 1)
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return round(sorted_vals[lower], 2)
    weight = pos - lower
    val = sorted_vals[lower] * (1.0 - weight) + sorted_vals[upper] * weight
    return round(val, 2)


def _draft_cps(cue: SubtitleCue) -> float:
    text = cue.translated_text.strip() or cue.clean_text
    chars = SubtitleCue._visible_char_count(text)
    return round(chars / cue.duration_sec, 2)


def _draft_cpl(cue: SubtitleCue) -> int:
    text = cue.translated_text.strip() or cue.clean_text
    lines = text.splitlines() or [text]
    return max((SubtitleCue._visible_char_count(ln) for ln in lines), default=0)


def evaluate_subtitle_quality(
    cues: list[SubtitleCue],
    profile: ProfileConfig,
    bible: MovieBible | None = None,
) -> EvalReport:
    """Compute honest timing, layout, and glossary compliance metrics across all cues."""
    total = len(cues)
    if total == 0:
        return EvalReport(
            total_cues=0,
            translated_cues=0,
            alignment_recall_pct=100.0,
            linked_cues_count=0,
            draft_cps_mean=0.0,
            draft_cps_p50=0.0,
            draft_cps_p95=0.0,
            draft_cps_max=0.0,
            draft_cps_violation_rate_pct=0.0,
            draft_cpl_violation_rate_pct=0.0,
            final_cps_mean=0.0,
            final_cps_p50=0.0,
            final_cps_p95=0.0,
            final_cps_max=0.0,
            final_cps_violation_rate_pct=0.0,
            final_cpl_violation_rate_pct=0.0,
            condensed_cues_count=0,
            glossary_checks=0,
            glossary_hits=0,
            glossary_hit_rate_pct=100.0,
        )

    translated_count = sum(1 for c in cues if c.translated_text.strip())
    linked_count = sum(1 for c in cues if c.linked_group_id is not None)
    condensed_count = sum(1 for c in cues if c.condensed_text.strip())

    # Draft metrics (Pass 2 output before Pass 3 condensation)
    draft_cps_list = sorted(_draft_cps(c) for c in cues)
    draft_cpl_list = [_draft_cpl(c) for c in cues]
    draft_cps_viol = sum(1 for v in draft_cps_list if v > profile.max_cps)
    draft_cpl_viol = sum(1 for v in draft_cpl_list if v > profile.max_cpl)

    # Final metrics (After Pass 3 condensation)
    final_cps_list = sorted(c.cps for c in cues)
    final_cpl_list = [c.cpl for c in cues]
    final_cps_viol = sum(1 for v in final_cps_list if v > profile.max_cps)
    final_cpl_viol = sum(1 for v in final_cpl_list if v > profile.max_cpl)

    # Glossary consistency check (evaluated at the semantic unit level so cross-cue
    # word-order inversions within the same linked_group_id are properly credited)
    glossary_checks = 0
    glossary_hits = 0
    if bible and bible.glossary:
        group_tgt_map: dict[int, str] = {}
        for c in cues:
            if c.linked_group_id is not None:
                prev = group_tgt_map.get(c.linked_group_id, "")
                group_tgt_map[c.linked_group_id] = f"{prev} {c.active_translation}".strip()

        for c in cues:
            src_lower = c.clean_text.lower()
            tgt_scope = (
                group_tgt_map[c.linked_group_id]
                if c.linked_group_id is not None
                else c.active_translation
            )
            for src_term, tgt_term in bible.glossary.items():
                if src_term.lower() in src_lower:
                    glossary_checks += 1
                    if tgt_term in tgt_scope:
                        glossary_hits += 1

    glossary_rate = (
        round((glossary_hits / glossary_checks) * 100.0, 1) if glossary_checks > 0 else 100.0
    )

    return EvalReport(
        total_cues=total,
        translated_cues=translated_count,
        alignment_recall_pct=round((translated_count / total) * 100.0, 1),
        linked_cues_count=linked_count,
        draft_cps_mean=round(sum(draft_cps_list) / total, 2),
        draft_cps_p50=_percentile(draft_cps_list, 50),
        draft_cps_p95=_percentile(draft_cps_list, 95),
        draft_cps_max=round(max(draft_cps_list), 2),
        draft_cps_violation_rate_pct=round((draft_cps_viol / total) * 100.0, 1),
        draft_cpl_violation_rate_pct=round((draft_cpl_viol / total) * 100.0, 1),
        final_cps_mean=round(sum(final_cps_list) / total, 2),
        final_cps_p50=_percentile(final_cps_list, 50),
        final_cps_p95=_percentile(final_cps_list, 95),
        final_cps_max=round(max(final_cps_list), 2),
        final_cps_violation_rate_pct=round((final_cps_viol / total) * 100.0, 1),
        final_cpl_violation_rate_pct=round((final_cpl_viol / total) * 100.0, 1),
        condensed_cues_count=condensed_count,
        glossary_checks=glossary_checks,
        glossary_hits=glossary_hits,
        glossary_hit_rate_pct=glossary_rate,
    )
