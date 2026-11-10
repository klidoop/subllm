"""Render translated SubtitleCue lists into .srt, .vtt, and customizable cinema-styled .ass files."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from subllm.core.models import SubtitleCue
from subllm.core.parser import (
    format_ass_timestamp,
    format_srt_timestamp,
    format_vtt_timestamp,
)


class AssStylePreset(StrEnum):
    """Visual styling presets for rendered .ass cinema subtitles."""

    FANSUB_CLASSIC = "fansub_classic"  # Crisp white Chinese + subtle steel-blue English below
    CINEMA_GOLD = "cinema_gold"        # Classic film-festival warm yellow + muted white source
    ART_HOUSE_SERIF = "art_house_serif"  # Literary/art-house serif font (Songti/Source Han Serif)
    MINIMAL_BOX = "minimal_box"        # Translucent dark background box for high-contrast scenes


def hex_to_ass_color(hex_rgb: str, alpha: int = 0) -> str:
    """Convert standard HTML hex '#RRGGBB' + alpha (0=opaque, 255=transparent) to ASS '&HAABBGGRR'.

    Note: ASS uses reversed byte order (BGR instead of RGB) and inverted alpha (00 is fully opaque).
    """
    clean = hex_rgb.lstrip("#").strip()
    if len(clean) != 6:
        clean = "FFFFFF"
    r, g, b = clean[0:2].upper(), clean[2:4].upper(), clean[4:6].upper()
    a_clamped = max(0, min(255, alpha))
    return f"&H{a_clamped:02X}{b}{g}{r}"


class AssStyleConfig(BaseModel):
    """Customizable typography and layout settings for .ass subtitle rendering (1080p canvas)."""

    zh_font: str = Field(default="Microsoft YaHei", description="Primary target language font family")
    en_font: str = Field(default="Arial", description="Secondary source language font family")
    zh_size: int = Field(default=54, ge=24, le=96, description="Primary font size on 1080p canvas")
    en_size: int = Field(default=32, ge=16, le=64, description="Secondary font size on 1080p canvas")
    zh_color_hex: str = Field(default="#FFFFFF", description="Primary text color in #RRGGBB")
    en_color_hex: str = Field(default="#DCE8F2", description="Secondary text color in #RRGGBB")
    outline_color_hex: str = Field(default="#101418", description="Outline/border color in #RRGGBB")
    bold_zh: bool = Field(default=True, description="Bold weight for primary text")
    outline_width: float = Field(default=2.6, ge=0.0, le=6.0, description="Outline thickness in px")
    shadow_depth: float = Field(default=1.2, ge=0.0, le=6.0, description="Drop shadow depth in px")
    opaque_box: bool = Field(
        default=False, description="Use Netflix-style translucent background box (BorderStyle=3)"
    )
    margin_v: int = Field(default=36, ge=10, le=150, description="Bottom vertical margin in px")
    bilingual_order: Literal["zh_top", "en_top"] = Field(
        default="zh_top", description="Whether translated Chinese appears above or below source text"
    )

    @classmethod
    def from_preset(cls, preset: AssStylePreset) -> "AssStyleConfig":
        """Create an AssStyleConfig from one of the built-in cinema presets."""
        if preset == AssStylePreset.CINEMA_GOLD:
            return cls(
                zh_font="Noto Sans CJK SC",
                en_font="Helvetica Neue",
                zh_size=56,
                en_size=32,
                zh_color_hex="#FFD838",
                en_color_hex="#CCCCCC",
                outline_color_hex="#000000",
                bold_zh=True,
                outline_width=2.8,
                shadow_depth=1.5,
            )
        if preset == AssStylePreset.ART_HOUSE_SERIF:
            return cls(
                zh_font="Source Han Serif SC",
                en_font="Georgia",
                zh_size=56,
                en_size=32,
                zh_color_hex="#F4F1EA",
                en_color_hex="#C8C2B8",
                outline_color_hex="#0A0A0A",
                bold_zh=False,
                outline_width=2.4,
                shadow_depth=1.4,
            )
        if preset == AssStylePreset.MINIMAL_BOX:
            return cls(
                zh_font="PingFang SC",
                en_font="Inter",
                zh_size=52,
                en_size=30,
                zh_color_hex="#FFFFFF",
                en_color_hex="#B8C2CC",
                outline_color_hex="#000000",
                bold_zh=False,
                outline_width=8.0,
                shadow_depth=0.0,
                opaque_box=True,
            )
        return cls()

    def to_ass_style_lines(self) -> tuple[str, str]:
        """Serialize into `[V4+ Styles]` MainZH and SubSRC definition lines."""
        zh_pri = hex_to_ass_color(self.zh_color_hex, alpha=0)
        en_pri = hex_to_ass_color(self.en_color_hex, alpha=15)
        out_col = hex_to_ass_color(
            self.outline_color_hex, alpha=96 if self.opaque_box else 0
        )
        back_col = hex_to_ass_color("#000000", alpha=128)
        border_style = 3 if self.opaque_box else 1
        bold_flag = -1 if self.bold_zh else 0
        en_outline = max(1.0, round(self.outline_width * 0.7, 1))
        en_shadow = max(0.0, round(self.shadow_depth * 0.7, 1))
        en_margin_v = max(10, self.margin_v // 2)

        main_line = (
            f"Style: MainZH,{self.zh_font},{self.zh_size},{zh_pri},&H000000FF,{out_col},{back_col},"
            f"{bold_flag},0,0,0,100,100,1,0,{border_style},{self.outline_width},{self.shadow_depth},"
            f"2,30,30,{self.margin_v},1"
        )
        sub_line = (
            f"Style: SubSRC,{self.en_font},{self.en_size},{en_pri},&H000000FF,{out_col},{back_col},"
            f"0,0,0,0,100,100,0,0,{border_style},{en_outline},{en_shadow},"
            f"2,30,30,{en_margin_v},1"
        )
        return main_line, sub_line


def render_srt(cues: list[SubtitleCue], bilingual: bool = True) -> str:
    """Export cues to standard SubRip (.srt) format."""
    blocks: list[str] = []
    for idx, c in enumerate(cues, start=1):
        start_ts = format_srt_timestamp(c.start_ms)
        end_ts = format_srt_timestamp(c.end_ms)
        zh_line = c.render_translated_with_tags()
        if bilingual and c.clean_text and c.clean_text != c.active_translation:
            body = f"{zh_line}\n{c.clean_text}"
        else:
            body = zh_line
        blocks.append(f"{idx}\n{start_ts} --> {end_ts}\n{body}")
    return "\n\n".join(blocks) + "\n"


def render_vtt(cues: list[SubtitleCue], bilingual: bool = True) -> str:
    """Export cues to standard WebVTT (.vtt) format."""
    blocks: list[str] = ["WEBVTT"]
    for idx, c in enumerate(cues, start=1):
        start_ts = format_vtt_timestamp(c.start_ms)
        end_ts = format_vtt_timestamp(c.end_ms)
        zh_line = c.render_translated_with_tags()
        if bilingual and c.clean_text and c.clean_text != c.active_translation:
            body = f"{zh_line}\n{c.clean_text}"
        else:
            body = zh_line
        blocks.append(f"{idx}\n{start_ts} --> {end_ts}\n{body}")
    return "\n\n".join(blocks) + "\n"


def _html_to_ass_inline(text: str) -> str:
    """Convert HTML italic tags <i>...</i> into ASS override tags {\\i1}...{\\i0} and \\n to \\N."""
    converted = text.replace("<i>", "{\\i1}").replace("</i>", "{\\i0}")
    converted = converted.replace("<I>", "{\\i1}").replace("</I>", "{\\i0}")
    return converted.replace("\n", "\\N")


def render_ass(
    cues: list[SubtitleCue],
    bilingual: bool = True,
    preset: AssStylePreset = AssStylePreset.FANSUB_CLASSIC,
    style_config: AssStyleConfig | None = None,
    title: str = "subllm Cinema Subtitle",
) -> str:
    """Export cues into a 1080p cinema-grade styled .ass file with custom fonts and colors."""
    cfg = style_config or AssStyleConfig.from_preset(preset)
    main_style, sub_style = cfg.to_ass_style_lines()
    header = f"""[Script Info]
; Generated by subllm (Time-Aware Cinema Subtitle Engine)
Title: {title}
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
{main_style}
{sub_style}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"""

    events: list[str] = [header]
    for c in cues:
        start_ts = format_ass_timestamp(c.start_ms)
        end_ts = format_ass_timestamp(c.end_ms)
        speaker = c.speaker.replace(",", " ") if c.speaker else ""

        zh_ass = _html_to_ass_inline(c.render_translated_with_tags())
        src_ass = _html_to_ass_inline(c.clean_text)

        if bilingual and src_ass and src_ass != c.active_translation:
            if cfg.bilingual_order == "en_top":
                combined = f"{{\\rSubSRC}}{src_ass}\\N{{\\rMainZH}}{zh_ass}"
            else:
                combined = f"{zh_ass}\\N{{\\rSubSRC}}{src_ass}"
        else:
            combined = zh_ass

        events.append(f"Dialogue: 0,{start_ts},{end_ts},MainZH,{speaker},0,0,0,,{combined}")

    return "\n".join(events) + "\n"
