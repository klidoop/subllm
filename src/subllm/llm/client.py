"""Async LLM client supporting Gemini, OpenAI-compatible APIs, and SQLite checkpointing."""

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Protocol

import httpx

from subllm.config import Settings


class LLMProviderClient(Protocol):
    """Protocol for async LLM text generation."""

    async def complete(self, prompt: str, temperature: float = 0.2) -> str:
        """Generate completion for prompt."""


class CheckpointStore:
    """Lightweight SQLite store for block-level translation checkpointing and resume."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS block_checkpoints (
                    job_key TEXT NOT NULL,
                    block_id INTEGER NOT NULL,
                    translations_json TEXT NOT NULL,
                    PRIMARY KEY (job_key, block_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS bible_cache (
                    job_key TEXT PRIMARY KEY,
                    bible_json TEXT NOT NULL
                )
                """
            )

    def save_block(self, job_key: str, block_id: int, translations: dict[int, str]) -> None:
        payload = json.dumps({str(k): v for k, v in translations.items()}, ensure_ascii=False)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO block_checkpoints (job_key, block_id, translations_json)
                VALUES (?, ?, ?)
                """,
                (job_key, block_id, payload),
            )

    def load_blocks(self, job_key: str) -> dict[int, dict[int, str]]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT block_id, translations_json FROM block_checkpoints WHERE job_key = ?",
                (job_key,),
            ).fetchall()
        result: dict[int, dict[int, str]] = {}
        for block_id, raw_json in rows:
            parsed = json.loads(raw_json)
            result[int(block_id)] = {int(k): str(v) for k, v in parsed.items()}
        return result

    def save_bible(self, job_key: str, bible_json: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO bible_cache (job_key, bible_json) VALUES (?, ?)",
                (job_key, bible_json),
            )

    def load_bible(self, job_key: str) -> str | None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT bible_json FROM bible_cache WHERE job_key = ?",
                (job_key,),
            ).fetchone()
        return str(row[0]) if row else None


class UnifiedLLMClient:
    """Async HTTP client for Gemini, OpenAI, DeepSeek, OpenRouter, and Ollama with key rotation."""

    def __init__(
        self,
        provider: str = "gemini",
        model: str = "gemini-2.5-flash",
        api_keys: list[str] | None = None,
        api_endpoint: str = "",
        timeout_sec: float = 90.0,
    ) -> None:
        self.provider = provider.lower()
        self.model = model
        self.api_keys = api_keys or [""]
        self.api_endpoint = api_endpoint
        self.timeout_sec = timeout_sec
        self._key_idx = 0

    @classmethod
    def from_settings(cls, cfg: Settings) -> "UnifiedLLMClient":
        return cls(
            provider=cfg.subllm_provider,
            model=cfg.subllm_model,
            api_keys=cfg.active_api_keys,
            api_endpoint=cfg.subllm_api_endpoint,
        )

    def _next_api_key(self) -> str:
        if not self.api_keys:
            return ""
        key = self.api_keys[self._key_idx % len(self.api_keys)]
        self._key_idx += 1
        return key

    async def complete(self, prompt: str, temperature: float = 0.2, max_retries: int = 3) -> str:
        """Send completion request with automatic retry and API key rotation on rate limits."""
        last_err: Exception | None = None
        for attempt in range(max_retries):
            api_key = self._next_api_key()
            try:
                if self.provider == "gemini":
                    return await self._call_gemini(prompt, api_key, temperature)
                if self.provider == "anthropic":
                    return await self._call_anthropic(prompt, api_key, temperature)
                return await self._call_openai_compatible(prompt, api_key, temperature)
            except Exception as exc:
                last_err = exc
                if attempt < max_retries - 1:
                    await asyncio.sleep(1.5 * (attempt + 1))
        raise RuntimeError(
            f"LLM call failed after {max_retries} attempts: {last_err}"
        ) from last_err

    async def _call_gemini(self, prompt: str, api_key: str, temperature: float) -> str:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required when provider='gemini'.")
        base_url = (
            self.api_endpoint.rstrip("/")
            if self.api_endpoint
            else "https://generativelanguage.googleapis.com/v1beta"
        )
        url = f"{base_url}/models/{self.model}:generateContent"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature},
        }
        async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
            resp = await client.post(url, params={"key": api_key}, json=payload)
            resp.raise_for_status()
            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                raise ValueError(f"Empty candidates from Gemini API: {data}")
            parts = candidates[0].get("content", {}).get("parts", [])
            return "".join(p.get("text", "") for p in parts).strip()

    async def _call_anthropic(self, prompt: str, api_key: str, temperature: float) -> str:
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required when provider='anthropic'.")
        url = self.api_endpoint or "https://api.anthropic.com/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model": self.model,
            "max_tokens": 4096,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return str(data["content"][0]["text"]).strip()

    async def _call_openai_compatible(self, prompt: str, api_key: str, temperature: float) -> str:
        default_endpoints = {
            "openai": "https://api.openai.com/v1/chat/completions",
            "deepseek": "https://api.deepseek.com/v1/chat/completions",
            "openrouter": "https://openrouter.ai/api/v1/chat/completions",
            "mistral": "https://api.mistral.ai/v1/chat/completions",
            "groq": "https://api.groq.com/openai/v1/chat/completions",
            "ollama": "http://localhost:11434/v1/chat/completions",
        }
        url = self.api_endpoint or default_endpoints.get(
            self.provider, "https://api.openai.com/v1/chat/completions"
        )
        headers = {"Content-Type": "application/json"}
        if api_key and api_key != "ollama-local":
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return str(data["choices"][0]["message"]["content"]).strip()


class DemoCinemaLLMClient:
    """Deterministic offline simulator for testing & UI preview when no live API key is provided."""

    DEMO_TRANSLATIONS: dict[str, str] = {
        "BERLIN - WINTER 1989": "1989年冬 · 柏林",
        "Viktor, listen to me carefully.": "维克多，仔细听我说",
        "The reason why the Nightfall Protocol": "“夜幕协议”之所以迟迟未能启动",
        "was never activated by Director Vance": "一直被万斯局长强行压下",
        "is because the cipher key was compromised in Vienna.": "是因为密码本早在维也纳就已经泄密了",
        "You're lying to me, Elena!": "你在骗我，艾琳娜！",
        "If Stasi headquarters finds out we lost the microfiche, we're both dead before sunrise!": "要是让史塔西总部发现胶片丢了，天亮前我们都得死！",
        "[Radio static crackles] ♪ Wind of Change playing faintly ♪": "[无线电沙沙作响] ♪ 隐约响起《变迁之风》 ♪",
        "Welcome back to Lecture 4 on": "欢迎回到第四讲，今天我们探讨",
        "Point-in-Time Feature Engineering and": "时间点（Point-in-Time）特征工程",
        "why target leakage destroys production ML models.": "以及为何目标穿越（Target Leakage）会毁掉生产级模型",
        "Let's look at the Dixon-Coles bivariate Poisson matrix first.": "我们先来看 Dixon-Coles 双变量泊松矩阵",
        "FREISING BEI MÜNCHEN - 14. DEZEMBER 1976": "慕尼黑近郊弗赖辛 · 1976年12月14日",
        "Herr Oetker, hören Sie mir jetzt ganz genau zu.": "奥特克先生，现在请您仔细听我说",
        "Wenn Ihr Vater die einundzwanzig Millionen D-Mark": "要是您父亲拿不出两千一百万西德马克",
        "bis morgen Mittag in großen, unregistrierten Scheinen": "没在明天中午前备齐大面额无连号现钞",
        "nicht ohne Einschaltung der Kriminalpolizei bereitstellt,": "或是胆敢暗中惊动刑警介入",
        "wird die Frischluftzufuhr in dieser Holzkiste abgeschaltet.": "这口木箱里的通风供氧就会被彻底切断",
        "Wenn Ihr Vater die einundzwanzig Millionen D-Mark bis morgen Mittag in großen, unregistrierten Scheinen": "要是明午之前，你父亲没备齐两千一百万不连号的大面额马克，",
        "nicht ohne Einschaltung der Kriminalpolizei bereitstellt, wird die Frischluftzufuhr in dieser Holzkiste abgeschaltet.": "甚至胆敢惊动刑警，这口木箱里的通风供氧就会被彻底切断。",
        "Haben Sie eine Spur von diesem Ciliox, Kommissar?": "探长，查到这个齐利奥克斯的踪迹了吗？",
        "Der Kerl benutzt öffentliche Münzfernsprecher an der Autobahn und wechselt alle drei Minuten den Standort!": "这家伙专挑高速公路旁的投币公用电话打过来，而且每三分钟就换一个地方！",
        "[Tonbandgerät klickt] Ich melde mich um Punkt vierzehn Uhr wieder, Herr Kommissar.": "[录音机咔哒声] 探长先生，下午两点整我会再打过来",
    }

    DEMO_CONDENSED: dict[str, str] = {
        "维克多，仔细听我说": "维克多，听好",
        "“夜幕协议”之所以迟迟未能启动": "“夜幕协议”迟迟未启动",
        "一直被万斯局长强行压下": "被万斯局长强行压下",
        "是因为密码本早在维也纳就已经泄密了": "只因密码已在维也纳泄密",
        "你在骗我，艾琳娜！": "你骗我，艾琳娜！",
        "要是让史塔西总部发现胶片丢了，天亮前我们都得死！": "史塔西若发现丢了胶片，咱俩活不过天亮！",
        "[无线电沙沙作响] ♪ 隐约响起《变迁之风》 ♪": "[电流声] ♪《变迁之风》♪",
        "慕尼黑近郊弗赖辛 · 1976年12月14日": "慕尼黑弗赖辛 · 1976.12.14",
        "奥特克先生，现在请您仔细听我说": "奥特克先生，请听好",
        "要是您父亲拿不出两千一百万西德马克": "若令尊拿不出两千一百万西德马克",
        "没在明天中午前备齐大面额无连号现钞": "未在明午备齐无连号大钞",
        "或是胆敢暗中惊动刑警介入": "或是胆敢惊动刑警",
        "这口木箱里的通风供氧就会被彻底切断": "木箱就会立刻断氧",
        "要是明午之前，你父亲没备齐两千一百万不连号的大面额马克，": "明午之前若未备齐两千一百万不连号西德马克，",
        "甚至胆敢惊动刑警，这口木箱里的通风供氧就会被彻底切断。": "或是敢惊动刑警，木箱就会立刻断氧。",
        "探长，查到这个齐利奥克斯的踪迹了吗？": "探长，查到齐利奥克斯了吗？",
        "这家伙专挑高速公路旁的投币公用电话打过来，而且每三分钟就换一个地方！": "他专挑高速路投币电话，每三分钟换一处！",
        "[录音机咔哒声] 探长先生，下午两点整我会再打过来": "[咔哒声] 探长，两点整我再打来",
    }

    async def complete(self, prompt: str, temperature: float = 0.2) -> str:
        if '"genre_and_tone"' in prompt:
            if "FREISING" in prompt or "Oetker" in prompt:
                return json.dumps(
                    {
                        "genre_and_tone": "德国真实罪案惊悚片《与魔鬼共舞：理查德·奥特克绑架案》(Der Tanz mit dem Teufel, 2001)，Christoph Waltz 饰演冷血绑匪 Ciliox，语气冰冷礼貌却极具压迫感",
                        "synopsis": "1976年12月慕尼黑近郊，高智商绑匪迪特·齐利奥克斯（Christoph Waltz 饰）将富商之子理查德·奥特克锁在通风木箱中勒索2100万西德马克，与刑警展开心理博弈。",
                        "characters": [
                            {
                                "name": "Dieter Ciliox (Christoph Waltz)",
                                "translated_name": "齐利奥克斯",
                                "role_and_tone": "冷酷克制的绑匪，对人质与警方始终使用极具讽刺感的敬语“您 (Sie)”",
                                "pronoun_and_address": "称呼 Oetker 为“奥特克先生 / 您”，称呼警方为“探长先生”",
                            },
                            {
                                "name": "Richard Oetker (Sebastian Koch)",
                                "translated_name": "理查德·奥特克",
                                "role_and_tone": "被囚禁于狭窄木箱中的食品巨头继承人",
                                "pronoun_and_address": "",
                            },
                        ],
                        "glossary": {
                            "Oetker": "奥特克",
                            "Ciliox": "齐利奥克斯",
                            "D-Mark": "西德马克",
                            "Kriminalpolizei": "刑警",
                            "Freising": "弗赖辛",
                            "Kommissar": "探长",
                        },
                    },
                    ensure_ascii=False,
                )
            return json.dumps(
                {
                    "genre_and_tone": "冷战谍战惊悚片（Cold War Espionage Thriller），台词紧凑克制、充满紧迫感",
                    "synopsis": "1989年冬的柏林，情报特工艾琳娜向搭档维克多揭露“夜幕协议”因维也纳密码泄密而搁置的真相，两人面临史塔西总部的追杀危机。",
                    "characters": [
                        {
                            "name": "Elena",
                            "translated_name": "艾琳娜",
                            "role_and_tone": "冷静果断的资深特工，语气紧迫直接",
                            "pronoun_and_address": "称呼 Viktor 为“维克多 / 你”",
                        },
                        {
                            "name": "Viktor",
                            "translated_name": "维克多",
                            "role_and_tone": "高度警觉的特工搭档，情绪濒临爆发",
                            "pronoun_and_address": "称呼 Elena 为“艾琳娜 / 你”",
                        },
                    ],
                    "glossary": {
                        "Nightfall Protocol": "夜幕协议",
                        "Director Vance": "万斯局长",
                        "Stasi": "史塔西",
                        "Vienna": "维也纳",
                        "Elena": "艾琳娜",
                        "Viktor": "维克多",
                    },
                },
                ensure_ascii=False,
            )

        if "OVERFLOWING CUES:" in prompt:
            out_lines = []
            for line in prompt.splitlines():
                if line.startswith("[") and "|| Draft:" in line:
                    idx_part = line.split("|")[0].strip("[ ")
                    draft = line.split("|| Draft:")[1].strip()
                    condensed = self.DEMO_CONDENSED.get(draft, draft)
                    out_lines.append(f"[{idx_part}] {condensed}")
            return "\n".join(out_lines)

        # Standard Pass-2 Block translation
        out_lines = []
        in_cues = False
        for line in prompt.splitlines():
            if line.startswith("CUES TO TRANSLATE:"):
                in_cues = True
                continue
            if line.startswith("CONTEXT AFTER"):
                in_cues = False
                continue
            if in_cues and line.startswith("[") and "]" in line:
                idx_str = line.split("|")[0].strip("[ ")
                src_text = line.split("]", 1)[1].strip()
                zh_text = self.DEMO_TRANSLATIONS.get(src_text, f"[译] {src_text}")
                out_lines.append(f"[{idx_str}] {zh_text}")
        return "\n".join(out_lines)
