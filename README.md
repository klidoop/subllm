# 🎬 subllm: Time-Aware Cinema Subtitle Translation Studio

**“A subtitle translation engine that respects cinema physics.”**

Unlike generic book or document translators that strip timestamps and force rigid line-by-line literal translations, `subllm` treats **screen duration (`dur`)**, **inter-cue silence gaps (`gap`)**, **cross-cue continuation merging (`merge_continuation_cues`)**, and **Netflix-grade reading speed limits (`CPS` & `CPL`)** as first-class constraints.

---

## ✨ Key Capabilities

1. **Interactive Web Studio ([`index.html`](index.html)) — Zero-Install / GitHub Pages Ready**:
   - **Subtitle Upload & Auto-Detection**: Parses `.srt`, `.ass`, and `.vtt` directly in the browser, auto-detects source language, and supports **45+ languages** plus custom free-text language/dialect input (`Other`).
   - **8 LLM Providers & Live `/models` Sync**: Built-in catalog of **75+ models** across **DeepSeek**, **OpenRouter**, **Google Gemini**, **OpenAI**, **Anthropic Claude**, **Mistral AI**, **Groq**, and **Ollama ($0 local)** — plus automatic `/models` endpoint fetching when you paste an API key.
   - **Cross-Cue Split Sentence Merger**: Automatically merges fragmented continuation cues (`gap <= 0.8s`, no terminal punctuation) and extends timestamps so inverted German/English clauses translate into natural, complete target sentences.
   - **Sample-First Workflow (`3 Cues` vs. `All Cues`)**: Preview 3 representative interior cues (or compare 2 models side-by-side) before clicking **Start Full Translation** with a live GUI progress bar and one-click `.ASS` / `.SRT` file export.
   - **1080p `.ASS` Movie Screen Preview**: Step through cues on a virtual cinema screen, customize fonts (including uploading local `.ttf`/`.otf` fonts) and Criterion colors, and export styled bilingual `.ASS` files.

2. **3-Pass Python Pipeline (`src/subllm/`)**:
   - **Pass 1 — Whole-Film Script Supervisor (`MovieBible`)**: Samples the subtitle script to extract genre/tone, character speech registers, and a terminology glossary.
   - **Pass 2 — Time-Gap Scene Chunking & Cross-Cue Re-alignment**: Merges continuation fragments (`--merge-continuations`) and groups cues by scene boundaries (`gap >= 3.0s`).
   - **Pass 3 — Deterministic CPS/CPL Quality Gate & Auto-Condensation**: Measures **Characters Per Second (CPS)** and **Characters Per Line (CPL)** and condenses fast-paced lines that exceed reading-speed thresholds.

---

## 🚀 Quickstart

### 1. Open the Interactive Web Workbench
Open [`index.html`](index.html) directly in any browser (or deploy the repository to GitHub Pages), or launch the Streamlit backend workbench:

```bash
make install
make run-ui
```

### 2. CLI Usage (`uv`)

```bash
# Run linter and unit tests (7 tests)
make lint
make test

# Inspect subtitle timing physics & cross-cue split sentences
uv run subllm inspect examples/tanz_mit_dem_teufel_2001_de.srt

# Pre-flight sample test (3 cues) or full translation (--demo offline or live API)
uv run subllm translate examples/tanz_mit_dem_teufel_2001_de.srt --demo --merge-continuations -o movie.zh.ass
```
