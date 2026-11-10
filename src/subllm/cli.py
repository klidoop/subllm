"""Typer CLI for subllm: inspect subtitle physics, run 3-pass translation, and evaluate CPS/CPL."""

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from subllm.config import ProfileConfig, SubtitleProfile, settings
from subllm.core.chunker import build_scene_blocks
from subllm.core.parser import parse_subtitle_file
from subllm.eval.metrics import evaluate_subtitle_quality
from subllm.llm.client import CheckpointStore, DemoCinemaLLMClient, UnifiedLLMClient
from subllm.pipeline.translator import run_translation_pipeline
from subllm.render.ass_styler import AssStylePreset, render_ass, render_srt, render_vtt

app = typer.Typer(
    name="subllm",
    help="🎬 subllm: Time-aware, movie-grade LLM subtitle translation engine (.srt / .ass / .vtt)",
    add_completion=False,
)
console = Console()


@app.command("inspect")
def inspect_command(
    input_file: Path = typer.Argument(..., exists=True, help="Path to .srt, .ass, or .vtt file"),
    profile: SubtitleProfile = typer.Option(
        SubtitleProfile.CINEMA, "--profile", "-p", help="Preset profile (cinema, asr, lecture)"
    ),
) -> None:
    """Inspect time-gap scene segmentation and cross-cue split sentences without calling an LLM."""
    cfg = ProfileConfig.from_preset(profile)
    cues = parse_subtitle_file(input_file)
    blocks = build_scene_blocks(cues, profile=cfg)

    linked_count = sum(1 for c in cues if c.linked_group_id is not None)
    console.print(
        Panel.fit(
            f"[bold cyan]File:[/bold cyan] {input_file.name}  |  "
            f"[bold green]Total Cues:[/bold green] {len(cues)}  |  "
            f"[bold yellow]Scene Blocks:[/bold yellow] {len(blocks)}  |  "
            f"[bold magenta]Cross-Cue Linked Lines:[/bold magenta] {linked_count}",
            title=f"🎬 subllm Physics Inspector ({profile.value} mode)",
        )
    )

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Block", justify="right", width=6)
    table.add_column("Cue #", justify="right", width=6)
    table.add_column("Dur (s)", justify="right", width=8)
    table.add_column("Gap (s)", justify="right", width=8)
    table.add_column("Link Group", justify="center", width=10)
    table.add_column("Max Chars", justify="right", width=9)
    table.add_column("Source Text (Clean)", overflow="fold")

    for blk in blocks:
        for c in blk.cues:
            gap_str = f"{c.gap_to_next_sec:.2f}s" if c.gap_to_next_sec is not None else "END"
            if c.gap_to_next_sec and c.gap_to_next_sec >= cfg.scene_gap_sec:
                gap_str = f"[bold red]{gap_str} (SCENE)[/bold red]"
            link_str = (
                f"[bold magenta]G{c.linked_group_id}[/bold magenta]" if c.linked_group_id else "-"
            )
            budget = c.max_budget_chars(cfg.max_cps, cfg.max_cpl)
            table.add_row(
                str(blk.block_id),
                str(c.index),
                f"{c.duration_sec:.2f}s",
                gap_str,
                link_str,
                str(budget),
                c.clean_text,
            )

    console.print(table)


@app.command("translate")
def translate_command(
    input_file: Path = typer.Argument(
        ..., exists=True, help="Input subtitle file (.srt/.ass/.vtt)"
    ),
    output_file: Path | None = typer.Option(
        None, "--output", "-o", help="Output file path (.ass, .srt, or .vtt)"
    ),
    source_lang: str = typer.Option("English", "--source-lang", "-s", help="Source language"),
    target_lang: str = typer.Option(
        "Simplified Chinese", "--target-lang", "-t", help="Target language"
    ),
    profile: SubtitleProfile = typer.Option(
        SubtitleProfile.CINEMA, "--profile", "-p", help="Preset profile (cinema, asr, lecture)"
    ),
    provider: str = typer.Option(
        settings.subllm_provider,
        "--provider",
        help="LLM provider (gemini, openai, deepseek, openrouter, ollama)",
    ),
    model: str = typer.Option(settings.subllm_model, "--model", "-m", help="Model name"),
    api_key: str = typer.Option("", "--api-key", "-k", help="Override API key"),
    bilingual: bool = typer.Option(True, "--bilingual/--mono", help="Export bilingual subtitles"),
    ass_preset: AssStylePreset = typer.Option(
        AssStylePreset.FANSUB_CLASSIC, "--ass-preset", help="Visual preset when exporting .ass"
    ),
    zh_font: str = typer.Option("", "--zh-font", help="Custom Chinese/target font family"),
    en_font: str = typer.Option("", "--en-font", help="Custom English/source font family"),
    zh_size: int = typer.Option(0, "--zh-size", help="Custom target font size (24-96, 1080p scale)"),
    zh_color: str = typer.Option("", "--zh-color", help="Custom target font color in #RRGGBB"),
    sample_cues: int = typer.Option(
        0,
        "--sample-cues",
        "-n",
        help="Test-translate only N sample cues (0 = full movie, >0 = pre-flight quality test)",
    ),
    sample_start: int | None = typer.Option(
        None,
        "--sample-start",
        help="Starting cue index for --sample-cues (omit to auto-pick best split-sentence window)",
    ),
    merge_continuations: bool = typer.Option(
        True,
        "--merge-continuations/--keep-fragmented",
        help="Merge cross-cue continuation fragments before translation and extend timestamps",
    ),
    demo: bool = typer.Option(
        False, "--demo", help="Run with built-in offline demo simulator (no API key needed)"
    ),
) -> None:
    """Run the 3-Pass Cinema Subtitle Pipeline and output a quality benchmark report."""
    from subllm.render.ass_styler import AssStyleConfig

    cfg = ProfileConfig.from_preset(profile)
    cues = parse_subtitle_file(input_file)

    if demo:
        llm = DemoCinemaLLMClient()
        console.print("[yellow]⚡ Running in offline --demo simulation mode.[/yellow]")
    else:
        keys = [k.strip() for k in api_key.split(",") if k.strip()] or settings.active_api_keys
        llm = UnifiedLLMClient(
            provider=provider,
            model=model,
            api_keys=keys,
            api_endpoint=settings.subllm_api_endpoint,
        )

    store = CheckpointStore(settings.cache_db_path)

    def _on_progress(msg: str, frac: float) -> None:
        console.print(f"[cyan][{int(frac * 100):3d}%][/cyan] {msg}")

    cues, bible, _ = asyncio.run(
        run_translation_pipeline(
            cues=cues,
            source_lang=source_lang,
            target_lang=target_lang,
            profile=cfg,
            llm=llm,
            checkpoint_store=store if not demo else None,
            progress_callback=_on_progress,
            sample_size=sample_cues,
            sample_start=sample_start,
            merge_continuations=merge_continuations,
        )
    )

    if sample_cues > 0:
        sample_table = Table(
            title=f"🧪 Pre-Flight Sample Quality Inspection ({len(cues)} Cues Tested · {provider}/{model})",
            show_header=True,
            header_style="bold magenta",
            show_lines=True,
        )
        sample_table.add_column("Cue #", justify="right", width=10)
        sample_table.add_column("Link", justify="center", width=6)
        sample_table.add_column("Source Line", overflow="fold")
        sample_table.add_column("Pass 2 (Cross-Cue Draft)", overflow="fold", style="yellow")
        sample_table.add_column("Pass 3 Final (CPS)", overflow="fold", style="bold green")

        for c in cues:
            link_tag = f"G{c.linked_group_id}" if c.linked_group_id else "-"
            cps_color = "green" if c.cps <= cfg.max_cps else "yellow"
            sample_table.add_row(
                f"#{c.index}\n[dim]{c.duration_sec:.2f}s[/dim]",
                link_tag,
                c.clean_text,
                c.translated_text,
                f"{c.final_text}\n[{cps_color}]({c.cps} CPS)[/{cps_color}]",
            )
        console.print(sample_table)

    report = evaluate_subtitle_quality(cues, profile=cfg, bible=bible)

    # Determine output path and format
    if output_file is None:
        output_file = input_file.with_name(f"{input_file.stem}.zh.ass")

    ext = output_file.suffix.lower()
    if ext == ".ass":
        style_cfg = AssStyleConfig.from_preset(ass_preset)
        if zh_font:
            style_cfg.zh_font = zh_font
        if en_font:
            style_cfg.en_font = en_font
        if zh_size >= 24:
            style_cfg.zh_size = zh_size
        if zh_color:
            style_cfg.zh_color_hex = zh_color
        rendered = render_ass(
            cues,
            bilingual=bilingual,
            preset=ass_preset,
            style_config=style_cfg,
            title=input_file.stem,
        )
    elif ext == ".vtt":
        rendered = render_vtt(cues, bilingual=bilingual)
    else:
        rendered = render_srt(cues, bilingual=bilingual)

    output_file.write_text(rendered, encoding="utf-8")

    # Display Movie Bible & Honest Evaluation Report
    console.print(
        Panel(
            f"[bold]Genre & Tone:[/bold] {bible.genre_and_tone}\n"
            f"[bold]Synopsis:[/bold] {bible.synopsis}\n"
            f"[bold]Glossary ({len(bible.glossary)} terms):[/bold] "
            + ", ".join(f"{k}→{v}" for k, v in bible.glossary.items()),
            title="📖 Pass 1: Extracted Movie Bible",
            border_style="green",
        )
    )

    metrics_table = Table(
        title="📊 Honest Subtitle Quality Benchmark (Before vs. After Pass 3)", show_header=True
    )
    metrics_table.add_column("Metric", style="bold")
    metrics_table.add_column("Pass 2 Draft", justify="right")
    metrics_table.add_column("Pass 3 Final", justify="right", style="bold green")
    metrics_table.add_column("Target Threshold", justify="right", style="dim")

    metrics_table.add_row(
        "CPS P50 / P95 / Max",
        f"{report.draft_cps_p50} / {report.draft_cps_p95} / {report.draft_cps_max}",
        f"{report.final_cps_p50} / {report.final_cps_p95} / {report.final_cps_max}",
        f"<= {cfg.max_cps} chars/s",
    )
    metrics_table.add_row(
        "CPS Violation Rate (%)",
        f"{report.draft_cps_violation_rate_pct}%",
        f"{report.final_cps_violation_rate_pct}%",
        "0.0%",
    )
    metrics_table.add_row(
        "CPL Overflow Rate (%)",
        f"{report.draft_cpl_violation_rate_pct}%",
        f"{report.final_cpl_violation_rate_pct}%",
        f"<= {cfg.max_cpl} chars/line",
    )
    metrics_table.add_row(
        "Cue Alignment Recall",
        f"{report.alignment_recall_pct}%",
        f"{report.alignment_recall_pct}%",
        "100.0%",
    )
    metrics_table.add_row(
        "Glossary Consistency",
        "-",
        f"{report.glossary_hit_rate_pct}% ({report.glossary_hits}/{report.glossary_checks})",
        "100.0%",
    )
    console.print(metrics_table)
    console.print(
        f"\n✅ Saved translated subtitles to: [bold underline green]{output_file}[/bold underline green]"
    )


if __name__ == "__main__":
    app()
