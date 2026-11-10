"""Pydantic data models for subtitle cues, scene blocks, movie bible, and evaluation metrics."""

import re

from pydantic import BaseModel, Field, computed_field

# Regex matching inline HTML tags (<i>, </i>, <b>, <font...>) and ASS override tags ({\an8}, {\pos(...)})
INLINE_TAG_PATTERN = re.compile(r"(<[^>]+>|\{\\[^}]+\})")
# Regex matching leading positioning/alignment ASS tags like {\an8} or {\pos(x,y)}
LEADING_ASS_TAG_PATTERN = re.compile(r"^(\{\\[^}]+\})+")


class CharacterProfile(BaseModel):
    """Voice and register profile for a recurring character in the film/series."""

    name: str = Field(description="Original character name in source language")
    translated_name: str = Field(description="Canonical translated name in target language")
    role_and_tone: str = Field(
        description="Brief role description, personality, and speech register (e.g., terse intelligence officer, sarcastic teen)"
    )
    pronoun_and_address: str = Field(
        default="",
        description="Preferred pronouns or how they address key counterparts (e.g., uses '您' to Director Vance, '你' to Viktor)",
    )


class MovieBible(BaseModel):
    """Global context extracted in Pass 1 before block translation begins."""

    genre_and_tone: str = Field(
        default="General cinema dialogue",
        description="Genre, historical setting, and overall translation tone guide",
    )
    synopsis: str = Field(
        default="",
        description="2-3 sentence plot summary inferred from the script",
    )
    characters: list[CharacterProfile] = Field(
        default_factory=list,
        description="Key characters and their speech registers",
    )
    glossary: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of recurring proper nouns/terms to canonical translations",
    )


class SubtitleCue(BaseModel):
    """A single timed subtitle entry with timing physics and lossless tag metadata."""

    index: int = Field(description="1-based sequential cue index")
    start_ms: int = Field(ge=0, description="Start timestamp in milliseconds")
    end_ms: int = Field(ge=0, description="End timestamp in milliseconds")
    raw_text: str = Field(description="Original text including inline HTML/ASS tags")
    speaker: str = Field(
        default="", description="Optional speaker label (e.g. from ASS Name field)"
    )
    gap_to_next_sec: float | None = Field(
        default=None,
        description="Silence gap in seconds from this cue's end_ms to the next cue's start_ms",
    )
    linked_group_id: int | None = Field(
        default=None,
        description="If part of a sentence split across multiple cues, shared group ID",
    )
    translated_text: str = Field(
        default="",
        description="Translated clean text (without leading structural tags)",
    )
    condensed_text: str = Field(
        default="",
        description="Optional Pass-3 condensed translation when initial translation exceeds CPS/CPL",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def duration_sec(self) -> float:
        """Duration of the subtitle on screen in seconds (minimum 0.1s to avoid div-by-zero)."""
        return max((self.end_ms - self.start_ms) / 1000.0, 0.1)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def clean_text(self) -> str:
        """Source text with inline HTML and ASS override tags stripped out for LLM input."""
        # Strip <...> and {\...} tags, while keeping content inside
        cleaned = INLINE_TAG_PATTERN.sub("", self.raw_text)
        return cleaned.strip()

    @property
    def leading_ass_tags(self) -> str:
        """Extract leading ASS positioning/alignment tags (e.g. '{\\an8}') to re-attach after translation."""
        match = LEADING_ASS_TAG_PATTERN.match(self.raw_text.strip())
        return match.group(0) if match else ""

    @property
    def is_italic(self) -> bool:
        """Detect whether the entire cue is wrapped in <i>...</i> or {\\i1}...{\\i0}."""
        stripped = self.raw_text.strip()
        return ("<i>" in stripped.lower() and "</i>" in stripped.lower()) or ("\\i1" in stripped)

    @property
    def active_translation(self) -> str:
        """Return condensed translation if available, else standard translation, else clean_text."""
        if self.condensed_text.strip():
            return self.condensed_text.strip()
        if self.translated_text.strip():
            return self.translated_text.strip()
        return self.clean_text

    @property
    def final_text(self) -> str:
        """Alias for active_translation (Pass 3 condensed text if available, else Pass 2 translated text)."""
        return self.active_translation

    @staticmethod
    def _visible_char_count(text: str) -> int:
        """Count visible characters excluding whitespace, tags, and basic punctuation."""
        no_tags = INLINE_TAG_PATTERN.sub("", text)
        # Exclude whitespace and common punctuation from reading-speed budget
        visible = re.sub(r"[\s\.,!?;:'\"，。！？；：、“”‘’（）()\[\]—…♪\-]", "", no_tags)
        return len(visible)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cps(self) -> float:
        """Characters Per Second (CPS) of the active translation."""
        char_count = self._visible_char_count(self.active_translation)
        return round(char_count / self.duration_sec, 2)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cpl(self) -> int:
        """Maximum Characters Per Line (CPL) across lines in the active translation."""
        lines = self.active_translation.splitlines() or [self.active_translation]
        return max((self._visible_char_count(line) for line in lines), default=0)

    def max_budget_chars(self, target_cps: float = 4.5, max_cpl: int = 18) -> int:
        """Compute maximum recommended target characters given cue duration and CPS/CPL limits."""
        cps_budget = int(round(self.duration_sec * target_cps))
        # Allow up to 2 lines maximum per cue in standard cinema layout
        return max(4, min(cps_budget, max_cpl * 2))

    def render_translated_with_tags(self) -> str:
        """Re-apply preserved leading ASS tags (like {\\an8}) and italics to the active translation."""
        text = self.active_translation
        if self.is_italic and not text.startswith("<i>"):
            text = f"<i>{text}</i>"
        if self.leading_ass_tags and not text.startswith(self.leading_ass_tags):
            text = f"{self.leading_ass_tags}{text}"
        return text


class SceneBlock(BaseModel):
    """A scene-coherent batch of subtitle cues sent to the LLM in Pass 2."""

    block_id: int
    cues: list[SubtitleCue]
    context_before: list[str] = Field(
        default_factory=list,
        description="Preceding 2-3 lines from the previous block for narrative continuity",
    )
    context_after: list[str] = Field(
        default_factory=list,
        description="Following 2-3 lines from the next block to resolve forward references",
    )
