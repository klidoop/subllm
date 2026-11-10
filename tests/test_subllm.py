"""Unit and integration test suite for subllm."""

from pathlib import Path

import pytest

from subllm.config import ProfileConfig, SubtitleProfile
from subllm.core.chunker import annotate_linked_sentence_groups, build_scene_blocks
from subllm.core.parser import parse_subtitle_file
from subllm.eval.metrics import evaluate_subtitle_quality
from subllm.llm.client import DemoCinemaLLMClient
from subllm.pipeline.translator import run_translation_pipeline
from subllm.render.ass_styler import render_ass, render_srt, render_vtt

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_parse_srt_preserves_timestamps_gaps_and_tags() -> None:
    """Verify SRT parser extracts exact ms timestamps, inter-cue gaps, and formatting tags."""
    cues = parse_subtitle_file(FIXTURES_DIR / "sample_movie.srt")
    assert len(cues) == 8

    # Cue 1 has {\an8} and <i>...</i>
    assert cues[0].start_ms == 1200
    assert cues[0].end_ms == 3000
    assert cues[0].duration_sec == 1.8
    assert cues[0].leading_ass_tags == r"{\an8}"
    assert cues[0].is_italic is True
    assert cues[0].clean_text == "BERLIN - WINTER 1989"

    # Check scene break gap between Cue 5 (end 12.4s) and Cue 6 (start 17.5s) -> 5.1s gap
    assert cues[4].gap_to_next_sec == 5.1


def test_cross_cue_split_sentence_detection_and_scene_chunking() -> None:
    """Verify cues 3, 4, 5 (one split sentence) share a linked_group_id and are never split across blocks."""
    cues = parse_subtitle_file(FIXTURES_DIR / "sample_movie.srt")
    profile = ProfileConfig.from_preset(SubtitleProfile.CINEMA)

    annotate_linked_sentence_groups(cues, max_merge_gap_sec=profile.merge_gap_sec)

    # Cues 3, 4, 5 form: "The reason why the Nightfall Protocol / was never activated by Director Vance / is because..."
    assert cues[2].linked_group_id is not None
    assert cues[2].linked_group_id == cues[3].linked_group_id == cues[4].linked_group_id
    # Cue 2 ends with a period, so it must NOT be linked to Cue 3
    assert cues[1].linked_group_id is None

    # Scene break of 5.1s after Cue 5 should split blocks between Cue 5 and Cue 6
    blocks = build_scene_blocks(cues, profile=profile)
    assert len(blocks) == 2
    assert [c.index for c in blocks[0].cues] == [1, 2, 3, 4, 5]
    assert [c.index for c in blocks[1].cues] == [6, 7, 8]


def test_parse_ass_and_vtt_formats() -> None:
    """Verify both .ass (with speaker Name field) and .vtt formats parse cleanly."""
    ass_cues = parse_subtitle_file(FIXTURES_DIR / "sample_styled.ass")
    assert len(ass_cues) == 6
    assert ass_cues[1].speaker == "Elena"
    assert ass_cues[5].leading_ass_tags == r"{\pos(960,1020)}"

    vtt_cues = parse_subtitle_file(FIXTURES_DIR / "sample_asr.vtt")
    assert len(vtt_cues) == 4
    assert vtt_cues[0].start_ms == 500
    assert vtt_cues[0].clean_text == "Welcome back to Lecture 4 on"


@pytest.mark.asyncio
async def test_three_pass_pipeline_and_cps_condensation() -> None:
    """Verify Pass 1 Bible extraction, Pass 2 translation, Pass 3 CPS condensation, and honest metrics."""
    cues = parse_subtitle_file(FIXTURES_DIR / "sample_movie.srt")
    profile = ProfileConfig.from_preset(SubtitleProfile.CINEMA)
    llm = DemoCinemaLLMClient()

    translated_cues, bible, _ = await run_translation_pipeline(
        cues=cues,
        source_lang="English",
        target_lang="Simplified Chinese",
        profile=profile,
        llm=llm,
    )

    # Pass 1 Bible checks
    assert "夜幕协议" in bible.glossary.values()
    assert len(bible.characters) == 2

    # Pass 2 & tag restoration check on Cue 1
    assert translated_cues[0].render_translated_with_tags() == r"{\an8}<i>1989年冬 · 柏林</i>"

    # Pass 3 Condensation check on Cue 7 (fast 1.35s line)
    cue7 = translated_cues[6]
    assert cue7.condensed_text != ""
    assert len(cue7.condensed_text) < len(cue7.translated_text)

    # Evaluation metrics check
    report = evaluate_subtitle_quality(translated_cues, profile=profile, bible=bible)
    assert report.alignment_recall_pct == 100.0
    assert report.condensed_cues_count >= 1
    assert report.final_cps_max < report.draft_cps_max
    assert report.glossary_hit_rate_pct == 100.0


def test_render_srt_vtt_and_styled_ass() -> None:
    """Verify exporters produce valid .srt, .vtt, and styled 1080p bilingual .ass output with custom fonts."""
    from subllm.render.ass_styler import AssStyleConfig, hex_to_ass_color

    # Verify RGB (#FFD838) -> BGR (&H0038D8FF) conversion
    assert hex_to_ass_color("#FFD838", alpha=0) == "&H0038D8FF"

    cues = parse_subtitle_file(FIXTURES_DIR / "sample_movie.srt")
    cues[0].translated_text = "1989年冬 · 柏林"

    srt_out = render_srt(cues[:1], bilingual=True)
    assert "00:00:01,200 --> 00:00:03,000" in srt_out
    assert r"{\an8}<i>1989年冬 · 柏林</i>" in srt_out

    vtt_out = render_vtt(cues[:1], bilingual=True)
    assert vtt_out.startswith("WEBVTT")
    assert "00:00:01.200 --> 00:00:03.000" in vtt_out

    custom_style = AssStyleConfig(
        zh_font="Source Han Serif SC",
        en_font="Georgia",
        zh_size=60,
        en_size=34,
        zh_color_hex="#FFD838",
        bilingual_order="en_top",
    )
    ass_out = render_ass(cues[:1], bilingual=True, style_config=custom_style)
    assert "Style: MainZH,Source Han Serif SC,60,&H0038D8FF" in ass_out
    assert "Style: SubSRC,Georgia,34" in ass_out
    assert r"{\rSubSRC}BERLIN - WINTER 1989\N{\rMainZH}{\an8}{\i1}1989年冬 · 柏林{\i0}" in ass_out


@pytest.mark.asyncio
async def test_preflight_sample_translation_preserves_linked_groups() -> None:
    """Verify --sample-cues smart selection picks cross-cue sentences without slicing linked groups in half."""
    cues = parse_subtitle_file(FIXTURES_DIR / "tanz_mit_dem_teufel_2001_de.srt")
    profile = ProfileConfig.from_preset(SubtitleProfile.CINEMA)
    llm = DemoCinemaLLMClient()

    # Request a 4-cue sample; Cues 3..6 form a 4-cue linked German Nebensatz
    sampled_cues, bible, _ = await run_translation_pipeline(
        cues=cues,
        source_lang="German",
        target_lang="Simplified Chinese",
        profile=profile,
        llm=llm,
        sample_size=4,
    )

    assert len(sampled_cues) < len(cues)
    assert "奥特克" in bible.glossary.values()
    # Ensure Cues 3, 4, 5, 6 (Link Group G1) are kept intact together
    sampled_indices = [c.index for c in sampled_cues]
    assert {3, 4, 5, 6}.issubset(set(sampled_indices))
    assert all(c.final_text.strip() != "" for c in sampled_cues)


def test_merge_continuation_cues_and_uniform_sampling() -> None:
    """Verify physical continuation cue merging and uniform interior sampling."""
    from subllm.core.chunker import merge_continuation_cues
    from subllm.core.sampling import cap_sample_cues, select_sample_indices

    cues = parse_subtitle_file(FIXTURES_DIR / "tanz_mit_dem_teufel_2001_de.srt")
    # Cues 3..6 are a 4-cue split sentence; merging with max_chars=110 should combine
    # #3+#4 into one cue and #5+#6 into one cue, reducing total cue count from 12 to 10.
    merged = merge_continuation_cues(cues, max_chars=110, max_merge_gap_sec=0.8)
    assert len(merged) < len(cues)
    # Check that merged cue 3 now spans from cue 3 start (6150ms) to cue 4 end (10350ms)
    assert merged[2].start_ms == 6150
    assert merged[2].end_ms >= 10350
    assert "Wenn Ihr Vater die einundzwanzig Millionen D-Mark" in merged[2].clean_text
    assert "bis morgen Mittag in großen, unregistrierten Scheinen" in merged[2].clean_text

    # Uniform interior sampling excludes first (0) and last (T-1) blocks
    indices = select_sample_indices(total_blocks=10, n=3)
    assert indices == [2, 4, 7]
    assert 0 not in indices and 9 not in indices

    # Sentence-boundary cap never stops on a cue that lacks terminal punctuation
    capped = cap_sample_cues(cues[2:6], max_chars=30)
    assert len(capped) == 4  # Extends all the way through Cue 6 (which ends with '.')


