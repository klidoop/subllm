"""Streamlit file-centric app for subllm: Drop a subtitle file, pick a language, get the result."""

import asyncio
from pathlib import Path

import streamlit as st

from subllm.config import ProfileConfig, SubtitleProfile, settings
from subllm.core.models import SubtitleCue
from subllm.core.parser import parse_ass, parse_srt_or_vtt
from subllm.eval.metrics import evaluate_subtitle_quality
from subllm.llm.client import CheckpointStore, DemoCinemaLLMClient, UnifiedLLMClient
from subllm.pipeline.translator import run_translation_pipeline
from subllm.render.ass_styler import AssStylePreset, render_ass, render_srt, render_vtt

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures"

st.set_page_config(
    page_title="subllm | 电影字幕一键精翻",
    page_icon="🎬",
    layout="centered",
)

st.title("🎬 subllm — 电影字幕一键精翻")
st.caption(
    "Drop a file, pick a language, get the result. "
    "后台全自动完成：**Pass 1 全片人物术语提取** → **Pass 2 跨时间轴断句对齐** → **Pass 3 CPS 超速自动压缩**"
)

uploaded_files = st.file_uploader(
    "📂 拖入字幕文件 (.srt / .ass / .vtt)，支持单部电影或整季批量上传",
    type=["srt", "ass", "ssa", "vtt"],
    accept_multiple_files=True,
)

col1, col2 = st.columns(2)
with col1:
    source_lang = st.selectbox("源语言 (Source)", ["English", "Japanese", "Korean", "French", "Spanish"])
    profile_choice = st.selectbox(
        "视频类型预设 (Mode)",
        options=[SubtitleProfile.CINEMA, SubtitleProfile.ASR_YOUTUBE, SubtitleProfile.LECTURE],
        format_func=lambda x: {
            SubtitleProfile.CINEMA: "🎬 电影 / 剧集 (Cinema)",
            SubtitleProfile.ASR_YOUTUBE: "📹 YouTube / ASR 自动字幕",
            SubtitleProfile.LECTURE: "📚 公开课 / 纪录片",
        }[x],
    )

with col2:
    target_lang = st.selectbox("目标语言 (Target)", ["Simplified Chinese", "Traditional Chinese", "English"])
    ass_preset = st.selectbox(
        "双语 .ass 视觉风格",
        options=list(AssStylePreset),
        format_func=lambda p: {
            AssStylePreset.FANSUB_CLASSIC: "💎 字幕组经典 (黑边雅黑 + 冰蓝英文)",
            AssStylePreset.CINEMA_GOLD: "🏆 影院暖金 (电影节黄字 + 银灰英文)",
            AssStylePreset.MINIMAL_BOX: "⬛ 极简半透明底框",
        }[p],
    )

with st.expander("⚙️ 模型与 API 设置 (支持 Gemini / DeepSeek / OpenAI / Claude / Ollama)"):
    from subllm.config import PROVIDER_MODELS

    provider = st.selectbox("Provider", list(PROVIDER_MODELS.keys()))
    preset_models = PROVIDER_MODELS.get(provider, ["gemini-2.5-flash"]) + ["自定义模型名 (Custom)..."]
    selected_model = st.selectbox("选择模型 (Model)", preset_models)
    if selected_model == "自定义模型名 (Custom)...":
        model_name = st.text_input("输入自定义 Model ID", value=settings.subllm_model)
    else:
        model_name = selected_model
    api_key = st.text_input(
        "API Key (留空则使用内置离线沙盒演示；支持逗号分隔多 Key 自动轮询)",
        value=settings.gemini_api_key,
        type="password",
    )

btn_col1, btn_col2 = st.columns([1, 1])
with btn_col1:
    test_sample_clicked = st.button(
        "🧪 先试译 5 句看质量 (Test Sample)",
        type="secondary",
        use_container_width=True,
    )
with btn_col2:
    full_run_clicked = st.button(
        "🚀 满意后全片翻译导出 (Full Movie)",
        type="primary",
        use_container_width=True,
    )

if test_sample_clicked or full_run_clicked:
    sample_n = 5 if test_sample_clicked else 0
    cfg = ProfileConfig.from_preset(profile_choice)
    keys = [k.strip() for k in api_key.split(",") if k.strip()]
    llm = (
        UnifiedLLMClient(provider=provider, model=model_name, api_keys=keys)
        if keys or provider == "ollama"
        else DemoCinemaLLMClient()
    )
    store = CheckpointStore(settings.cache_db_path)

    # Use uploaded files or fallback to built-in movie sample
    targets: list[tuple[str, str, str]] = []
    if uploaded_files:
        for uf in uploaded_files:
            p = Path(uf.name)
            targets.append((p.stem, p.suffix.lower(), uf.getvalue().decode("utf-8", errors="replace")))
    else:
        sample_p = FIXTURES_DIR / "sample_movie.srt"
        targets.append((sample_p.stem, ".srt", sample_p.read_text(encoding="utf-8")))

    results = []
    prog = st.progress(0.0, text="正在准备翻译...")

    for file_stem, ext, content in targets:
        cues: list[SubtitleCue] = (
            parse_ass(content) if ext in {".ass", ".ssa"} else parse_srt_or_vtt(content)
        )
        translated_cues, bible, _ = asyncio.run(
            run_translation_pipeline(
                cues=cues,
                source_lang=source_lang,
                target_lang=target_lang,
                profile=cfg,
                llm=llm,
                checkpoint_store=store if keys else None,
                progress_callback=lambda msg, frac, s=file_stem: prog.progress(
                    min(max(frac, 0.0), 1.0), text=f"[{s}] {msg}"
                ),
                sample_size=sample_n,
            )
        )
        report = evaluate_subtitle_quality(translated_cues, profile=cfg, bible=bible)
        results.append(
            {
                "stem": file_stem,
                "is_sample": sample_n > 0,
                "cues": translated_cues,
                "report": report,
                "bible": bible,
                "ass": render_ass(translated_cues, bilingual=True, preset=ass_preset, title=file_stem),
                "srt": render_srt(translated_cues, bilingual=True),
                "vtt": render_vtt(translated_cues, bilingual=True),
            }
        )

    st.session_state["batch_results"] = results

for item in st.session_state.get("batch_results", []):
    rep = item["report"]
    bible = item["bible"]
    with st.container(border=True):
        mode_badge = "🧪 试译样本验货 (5 Cues)" if item.get("is_sample") else "✅ 全片翻译完成"
        st.subheader(f"{mode_badge} — {item['stem']}")
        if item.get("is_sample"):
            st.caption("以下是自动抽取的代表性长难句（含跨时间轴断句对齐 + CPS 语速精简），满意后点击上方「🚀 满意后全片翻译导出」即可：")
            st.dataframe(
                [
                    {
                        "句号": f"#{c.index} ({c.duration_sec:.1f}s)",
                        "跨轴组": f"G{c.linked_group_id}" if c.linked_group_id else "-",
                        "原文字幕": c.clean_text,
                        "Pass 2 (跨轴重组初稿)": c.translated_text,
                        "Pass 3 终稿": f"{c.final_text} ({c.cps} CPS)",
                    }
                    for c in item["cues"]
                ],
                use_container_width=True,
                hide_index=True,
            )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("总字幕条数", f"{rep.total_cues} 条")
        c2.metric("自动重组跨轴断句", f"{rep.linked_cues_count} 条")
        c3.metric("自动压缩超速台词", f"{rep.condensed_cues_count} 条")
        c4.metric("时间轴对齐完整率", f"{rep.alignment_recall_pct}%")

        d1, d2, d3 = st.columns(3)
        d1.download_button(
            "💎 下载双语美化 .ass",
            data=item["ass"].encode("utf-8"),
            file_name=f"{item['stem']}.zh.ass",
            use_container_width=True,
        )
        d2.download_button(
            "📄 下载标准双语 .srt",
            data=item["srt"].encode("utf-8"),
            file_name=f"{item['stem']}.zh.srt",
            use_container_width=True,
        )
        d3.download_button(
            "🌐 下载双语 .vtt",
            data=item["vtt"].encode("utf-8"),
            file_name=f"{item['stem']}.zh.vtt",
            use_container_width=True,
        )

        with st.expander("▸ 查看自动提取的全片人物与术语表 (可选)"):
            st.write(f"**基调**：{bible.genre_and_tone}")
            st.write(f"**梗概**：{bible.synopsis}")
            if bible.glossary:
                st.write("**锁定术语**：" + " · ".join(f"`{k} → {v}`" for k, v in bible.glossary.items()))
