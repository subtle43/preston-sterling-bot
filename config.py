from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
DEFAULT_PROMPT_FILE = ROOT / "prompts" / "system.txt"


def _bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _think(name: str, default: bool | str = True) -> bool | str:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    value = raw.strip().lower()
    if value in {"0", "false", "no", "off"}:
        return False
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"low", "medium", "high"}:
        return value
    return default


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _id_set(name: str) -> frozenset[int]:
    raw = os.getenv(name, "")
    values: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if part:
            values.add(int(part))
    return frozenset(values)


def _name_set(name: str, default: str = "") -> frozenset[str]:
    raw = os.getenv(name)
    if raw is None:
        raw = default
    values: set[str] = set()
    for part in raw.split(","):
        cleaned = part.strip().lower().lstrip("@")
        if cleaned:
            values.add(cleaned)
    return frozenset(values)


def _load_system_prompt() -> str:
    override = os.getenv("SYSTEM_PROMPT", "").strip()
    if override:
        return override
    path_raw = os.getenv("SYSTEM_PROMPT_FILE", "").strip()
    path = Path(path_raw) if path_raw else DEFAULT_PROMPT_FILE
    if not path.is_absolute():
        path = ROOT / path
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return "You are a helpful Discord assistant running locally via Ollama."


@dataclass
class Settings:
    discord_token: str
    ollama_host: str
    ollama_model: str
    ollama_model_heavy: str
    num_ctx_heavy: int
    # Setting gemini_model switches the whole bot onto the Gemini API. Ollama stays
    # configured either way so swapping back is one commented line.
    gemini_api_key: str
    gemini_model: str
    gemini_thinking_level: str
    gemini_num_ctx: int
    chat_index_dir: str
    chat_index_exclude: frozenset[int]
    system_prompt: str
    guild_id: int | None
    history_limit: int
    history_char_budget: int
    history_limit_gemini: int
    history_char_budget_gemini: int
    summary_words: int
    summary_words_gemini: int
    summary_enabled: bool
    temperature: float
    num_ctx: int
    num_predict: int | None
    keep_alive: str
    think: bool | str
    show_thinking: bool
    request_timeout: float
    listen_channels: frozenset[int]
    allowed_guilds: frozenset[int]
    respond_to_dms: bool
    message_content_intent: bool
    command_prefix: str
    roast_usernames: frozenset[str]
    roast_user_ids: frozenset[int]
    roast_cooldown: int
    roast_bot_usernames: frozenset[str]
    roast_bot_ids: frozenset[int]
    roast_bot_cooldown: int
    reply_to_bots: bool
    bot_reply_cooldown: float
    dm_cooldown: int
    drunk_chance: float
    reply_max_words: int
    reply_max_words_local: int
    summary_max_words: int
    reply_hard_max_words: int
    lore_enabled: bool
    straight_chance: float
    moods_enabled: bool
    daily_rap_enabled: bool
    daily_rap_channel: str
    daily_rap_time: str
    mood_min_minutes: int
    mood_max_minutes: int
    vocab_register: str
    callback_chance: float
    callback_cooldown: int
    persist_memory: bool
    vision_enabled: bool
    reaction_chance: float
    interject_chance: float
    interject_cooldown: int
    ambient_channels: frozenset[int]
    context_messages: int
    context_messages_gemini: int
    context_chars: int
    context_chars_gemini: int
    image_gen_enabled: bool
    image_cooldown: int
    image_model: str
    image_local_enabled: bool
    image_local_model: str
    song_gen_enabled: bool
    song_model: str
    song_duration: int
    song_cooldown: int
    song_steps: int
    daily_rap_audio: bool
    weekly_awards_enabled: bool
    weekly_awards_day: int
    weekly_awards_time: str
    intent_classifier_enabled: bool
    feedback_enabled: bool
    reply_channel: int
    reply_in_place_channels: frozenset[int]
    voice_mode: bool
    harness: str
    lite_prompt: str
    lite_context: int
    gemini_fallback_model: str
    tavily_api_key: str
    search_enabled: bool
    search_max_per_day: int
    search_cooldown: int
    fr_enabled: bool
    chat_index_enabled: bool
    chat_router_enabled: bool
    fr_index_dir: str
    fr_top_k: int
    fr_min_score: float
    owner_ids: frozenset[int] = field(default_factory=frozenset)

    @classmethod
    def load(cls) -> Settings:
        load_dotenv(ROOT / ".env")
        guild_raw = os.getenv("DISCORD_GUILD_ID", "").strip()
        predict_raw = os.getenv("OLLAMA_NUM_PREDICT", "").strip()
        return cls(
            discord_token=os.getenv("DISCORD_TOKEN", "").strip(),
            ollama_host=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").strip(),
            ollama_model=os.getenv("OLLAMA_MODEL", "gemma4:e4b").strip(),
            ollama_model_heavy=os.getenv("OLLAMA_MODEL_HEAVY", "").strip(),
            num_ctx_heavy=max(512, _int("OLLAMA_NUM_CTX_HEAVY", 131072)),
            gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
            gemini_model=os.getenv("GEMINI_MODEL", "").strip(),
            gemini_thinking_level=os.getenv("GEMINI_THINKING_LEVEL", "").strip(),
            # gemini-3.5-flash-lite reports inputTokenLimit 1,048,576. This is what
            # log budgets are sized against when Gemini is the backend; lower it to
            # cap how much a single log review can cost.
            gemini_num_ctx=max(512, _int("GEMINI_NUM_CTX", 1_048_576)),
            chat_index_dir=os.getenv("CHAT_INDEX_DIR", "data/chat_index").strip(),
            # Applied while SCRAPING, not while querying: an excluded channel's
            # messages never reach the index file at all.
            chat_index_exclude=_id_set("CHAT_INDEX_EXCLUDE_CHANNELS"),
            system_prompt=_load_system_prompt(),
            guild_id=int(guild_raw) if guild_raw else None,
            history_limit=max(2, _int("HISTORY_LIMIT", 20)),
            history_char_budget=max(1000, _int("HISTORY_CHAR_BUDGET", 14000)),
            history_limit_gemini=max(2, _int("HISTORY_LIMIT_GEMINI", 400)),
            history_char_budget_gemini=max(1000, _int("HISTORY_CHAR_BUDGET_GEMINI", 300000)),
            summary_words=max(50, _int("SUMMARY_WORDS", 300)),
            summary_words_gemini=max(50, _int("SUMMARY_WORDS_GEMINI", 1200)),
            summary_enabled=_bool("SUMMARY_ENABLED", True),
            temperature=_float("OLLAMA_TEMPERATURE", 0.7),
            num_ctx=max(512, _int("OLLAMA_NUM_CTX", 8192)),
            num_predict=int(predict_raw) if predict_raw else None,
            keep_alive=os.getenv("OLLAMA_KEEP_ALIVE", "30m").strip() or "30m",
            think=_think("OLLAMA_THINK", True),
            show_thinking=_bool("SHOW_THINKING", False),
            request_timeout=_float("OLLAMA_TIMEOUT", 300.0),
            listen_channels=_id_set("LISTEN_CHANNELS"),
            allowed_guilds=_id_set("ALLOWED_GUILD_IDS"),
            respond_to_dms=_bool("RESPOND_TO_DMS", True),
            message_content_intent=_bool("MESSAGE_CONTENT_INTENT", True),
            command_prefix=os.getenv("COMMAND_PREFIX", "!ai").strip() or "!ai",
            roast_usernames=_name_set("ROAST_USERNAMES", "member_b"),
            roast_user_ids=_id_set("ROAST_USER_IDS"),
            roast_cooldown=max(0, _int("ROAST_COOLDOWN", 900)),
            roast_bot_usernames=_name_set("ROAST_BOT_USERNAMES", ""),
            roast_bot_ids=_id_set("ROAST_BOT_IDS"),
            roast_bot_cooldown=max(0, _int("ROAST_BOT_COOLDOWN", 300)),
            reply_to_bots=_bool("REPLY_TO_BOTS", True),
            bot_reply_cooldown=max(3.0, _float("BOT_REPLY_COOLDOWN", 8.0)),
            dm_cooldown=max(0, _int("DM_COOLDOWN", 60)),
            drunk_chance=min(1.0, max(0.0, _float("DRUNK_CHANCE", 0.2))),
            reply_max_words=max(0, _int("REPLY_MAX_WORDS", 80)),
            # Small local models fill whatever cap they are given; Gemini follows
            # the persona's own length rules and keeps REPLY_MAX_WORDS.
            reply_max_words_local=max(0, _int("REPLY_MAX_WORDS_LOCAL", 40)),
            summary_max_words=max(0, _int("SUMMARY_MAX_WORDS", 500)),
            reply_hard_max_words=max(0, _int("REPLY_HARD_MAX_WORDS", 1200)),
            lore_enabled=_bool("LORE_ENABLED", True),
            straight_chance=min(1.0, max(0.0, _float("STRAIGHT_CHANCE", 0.65))),
            moods_enabled=_bool("MOODS_ENABLED", True),
            daily_rap_enabled=_bool("DAILY_RAP_ENABLED", True),
            daily_rap_channel=os.getenv("DAILY_RAP_CHANNEL", "random-pops-and-bangs").strip(),
            daily_rap_time=os.getenv("DAILY_RAP_TIME", "07:00").strip() or "07:00",
            mood_min_minutes=max(5, _int("MOOD_MIN_MINUTES", 30)),
            mood_max_minutes=max(5, _int("MOOD_MAX_MINUTES", 120)),
            vocab_register=os.getenv("VOCAB_REGISTER", "auto").strip() or "auto",
            callback_chance=min(1.0, max(0.0, _float("CALLBACK_CHANCE", 0.25))),
            callback_cooldown=max(0, _int("CALLBACK_COOLDOWN", 900)),
            persist_memory=_bool("PERSIST_MEMORY", True),
            vision_enabled=_bool("VISION_ENABLED", True),
            reaction_chance=min(1.0, max(0.0, _float("REACTION_CHANCE", 0.05))),
            interject_chance=min(1.0, max(0.0, _float("INTERJECT_CHANCE", 0.02))),
            interject_cooldown=max(0, _int("INTERJECT_COOLDOWN", 600)),
            ambient_channels=_id_set("AMBIENT_CHANNELS"),
            context_messages=max(0, _int("CONTEXT_MESSAGES", 25)),
            context_messages_gemini=max(0, _int("CONTEXT_MESSAGES_GEMINI", 150)),
            context_chars=max(500, _int("CONTEXT_CHARS", 4000)),
            context_chars_gemini=max(500, _int("CONTEXT_CHARS_GEMINI", 40000)),
            image_gen_enabled=_bool("IMAGE_GEN_ENABLED", True),
            image_cooldown=max(0, _int("IMAGE_COOLDOWN", 30)),
            image_model=os.getenv("IMAGE_MODEL", "").strip(),
            image_local_enabled=_bool("IMAGE_LOCAL_ENABLED", True),
            image_local_model=os.getenv("IMAGE_LOCAL_MODEL", "Lykon/dreamshaper-xl-lightning").strip(),
            song_gen_enabled=_bool("SONG_GEN_ENABLED", True),
            song_model=os.getenv("SONG_MODEL", "ACE-Step/acestep-v15-xl-turbo-diffusers").strip(),
            song_duration=min(180, max(10, _int("SONG_DURATION", 60))),
            song_cooldown=max(0, _int("SONG_COOLDOWN", 120)),
            song_steps=max(1, _int("SONG_STEPS", 8)),
            daily_rap_audio=_bool("DAILY_RAP_AUDIO", False),
            # The Preston Awards: posted in the DAILY_RAP_CHANNEL once a week.
            # Day is 0=Monday .. 6=Sunday.
            weekly_awards_enabled=_bool("WEEKLY_AWARDS_ENABLED", True),
            weekly_awards_day=min(6, max(0, _int("WEEKLY_AWARDS_DAY", 6))),
            weekly_awards_time=os.getenv("WEEKLY_AWARDS_TIME", "18:00").strip() or "18:00",
            intent_classifier_enabled=_bool("INTENT_CLASSIFIER", True),
            feedback_enabled=_bool("FEEDBACK_ENABLED", True),
            reply_channel=_int("REPLY_CHANNEL", 0),
            # Channels exempt from REPLY_CHANNEL: pinged there, answered there.
            reply_in_place_channels=_id_set("REPLY_IN_PLACE_CHANNELS"),
            voice_mode=_bool("VOICE_MODE", False),
            # full = the whole persona harness (default); lite = a short persona
            # plus recent chat; voice = the train/ format, for fine-tuned models.
            # VOICE_MODE=true is the older spelling of HARNESS=voice.
            harness=(os.getenv("HARNESS", "").strip().lower()
                     or ("voice" if _bool("VOICE_MODE", False) else "full")),
            # The short prompt HARNESS=lite uses; one file per persona/model.
            lite_prompt=os.getenv("LITE_PROMPT", "prompts/lite.txt").strip() or "prompts/lite.txt",
            # How many recent human messages lite/voice show (its own replies:
            # up to half that, min 3). Small on purpose - see voice_reply.
            lite_context=min(40, max(1, _int("LITE_CONTEXT", 5))),
            # Which Gemini model "/model gemini" switches to when GEMINI_MODEL is blank.
            gemini_fallback_model=os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite").strip(),
            tavily_api_key=os.getenv("TAVILY_API_KEY", "").strip(),
            search_enabled=_bool("SEARCH_ENABLED", False),
            search_max_per_day=max(0, _int("SEARCH_MAX_PER_DAY", 30)),
            search_cooldown=max(0, _int("SEARCH_COOLDOWN", 20)),
            fr_enabled=_bool("FR_ENABLED", False),
            chat_index_enabled=_bool("CHAT_INDEX_ENABLED", False),
            chat_router_enabled=_bool("CHAT_INDEX_ROUTER", True),
            fr_index_dir=os.getenv("FR_INDEX_DIR", "data/fr_index").strip() or "data/fr_index",
            fr_top_k=max(1, _int("FR_TOP_K", 24)),
            fr_min_score=min(1.0, max(0.0, _float("FR_MIN_SCORE", 0.50))),
            owner_ids=_id_set("OWNER_IDS"),
        )
