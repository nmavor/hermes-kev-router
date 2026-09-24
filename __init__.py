"""Hermes Kev Router plugin.

Classifies a user turn through a local System One endpoint, then narrows only
recognized Hermes built-in tool schemas in ``llm_request`` middleware. Unknown
plugin and MCP tools are preserved. Routing is disclosure optimization, not an
authorization boundary.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

PLUGIN_NAME = "hermes-kev-router"
PLUGIN_VERSION = "0.1.1"

logger = logging.getLogger(__name__)

DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "endpoint": "http://127.0.0.1:8009/v1/systemone",
    "kev_model": "kev-latest",
    "timeout": 1.25,
    "min_probability": 0.45,
    "min_confidence": 0.30,
    "min_margin": 0.15,
    "max_state_chars": 12000,
    "preserve_unknown_tools": True,
}

_PROFILES = ("chat", "research", "coding", "browser", "personal", "automation", "full")
_PROFILE_DESCRIPTIONS = {
    "chat": "Conversation, explanation, or greeting needing no external action",
    "research": (
        "Information lookup such as weather, news, facts, web search, reading a page, "
        "or prior-session search"
    ),
    "coding": "Software development, repository inspection, editing, testing, or debugging",
    "browser": "Interactive browser navigation or page interaction; not ordinary web lookup",
    "personal": "Personal memory, Home Assistant, media, or connected-account assistance",
    "automation": "Scheduling, repeated jobs, delegation, or coordinated system actions",
    "full": (
        "Last resort only when one request clearly needs at least three different capability "
        "families; never choose for one lookup or one task"
    ),
}

_SKILL_TOOLS = frozenset({"skills_list", "skill_view", "skill_manage"})
_BROWSER_TOOLS = frozenset(
    {
        "browser_navigate",
        "browser_snapshot",
        "browser_click",
        "browser_type",
        "browser_scroll",
        "browser_back",
        "browser_press",
        "browser_get_images",
        "browser_vision",
        "browser_console",
        "browser_cdp",
        "browser_dialog",
        "browser_vault_list",
        "browser_vault_unlock",
        "browser_vault_fill",
        "browser_vault_save_login",
        "browser_vault_enter_code",
        "browser_exec",
    }
)
_KANBAN_TOOLS = frozenset(
    {
        "kanban_show",
        "kanban_list",
        "kanban_complete",
        "kanban_block",
        "kanban_request_review",
        "kanban_request_changes",
        "kanban_heartbeat",
        "kanban_comment",
        "kanban_create",
        "kanban_link",
        "kanban_unblock",
        "kanban_attach",
        "kanban_attach_url",
        "kanban_attachments",
    }
)
_HOME_ASSISTANT_TOOLS = frozenset(
    {
        "ha_list_entities",
        "ha_get_state",
        "ha_list_services",
        "ha_call_service",
    }
)
_KNOWN_BUILTINS = (
    frozenset(
        {
            "web_search",
            "web_extract",
            "terminal",
            "process_manage",
            "read_file",
            "write_file",
            "patch",
            "search_files",
            "vision_analyze",
            "image_generate",
            "text_to_speech",
            "todo_list",
            "memory",
            "session_search",
            "clarify",
            "execute_code",
            "delegate_task",
            "cronjob_manage",
            "computer_use",
            "manage_connections",
        }
    )
    | _SKILL_TOOLS
    | _BROWSER_TOOLS
    | _KANBAN_TOOLS
    | _HOME_ASSISTANT_TOOLS
)

_PROFILE_TOOLS = {
    "chat": frozenset({"clarify"}) | _SKILL_TOOLS,
    "research": frozenset(
        {
            "web_search",
            "web_extract",
            "session_search",
            "vision_analyze",
            "read_file",
            "clarify",
        }
    )
    | _SKILL_TOOLS,
    "coding": frozenset(
        {
            "terminal",
            "process_manage",
            "read_file",
            "write_file",
            "patch",
            "search_files",
            "web_search",
            "web_extract",
            "vision_analyze",
            "todo_list",
            "execute_code",
            "delegate_task",
            "clarify",
        }
    )
    | _SKILL_TOOLS,
    "browser": _BROWSER_TOOLS
    | frozenset(
        {
            "web_search",
            "web_extract",
            "vision_analyze",
            "clarify",
        }
    )
    | _SKILL_TOOLS,
    "personal": _HOME_ASSISTANT_TOOLS
    | frozenset(
        {
            "memory",
            "session_search",
            "text_to_speech",
            "image_generate",
            "vision_analyze",
            "manage_connections",
            "clarify",
        }
    )
    | _SKILL_TOOLS,
}


@dataclass(frozen=True)
class Decision:
    profile: str
    created_at: float


_decisions: dict[tuple[str, str], Decision] = {}
_decisions_lock = threading.RLock()
_DECISION_TTL_SECONDS = 3600.0
_MAX_DECISIONS = 2048


def _setting(ctx: Any, key: str) -> Any:
    try:
        return ctx.get_config(key, DEFAULTS[key])
    except Exception:  # noqa: BLE001 - older/probe contexts fail open to defaults
        return DEFAULTS[key]


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.strip().lower() in {"true", "yes", "on", "1"}:
            return True
        if value.strip().lower() in {"false", "no", "off", "0"}:
            return False
    return default


def _settings(ctx: Any) -> dict[str, Any]:
    try:
        timeout = float(_setting(ctx, "timeout"))
        min_probability = float(_setting(ctx, "min_probability"))
        min_confidence = float(_setting(ctx, "min_confidence"))
        min_margin = float(_setting(ctx, "min_margin"))
        max_state_chars = int(_setting(ctx, "max_state_chars"))
    except (TypeError, ValueError):
        raise ValueError("invalid numeric plugin setting") from None
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive and finite")
    thresholds = (min_probability, min_confidence, min_margin)
    if any(not math.isfinite(value) or value < 0 or value > 1 for value in thresholds):
        raise ValueError("routing thresholds must be between 0 and 1")
    return {
        "enabled": _as_bool(_setting(ctx, "enabled"), DEFAULTS["enabled"]),
        "endpoint": str(_setting(ctx, "endpoint") or DEFAULTS["endpoint"]),
        "kev_model": str(_setting(ctx, "kev_model") or DEFAULTS["kev_model"]),
        "timeout": timeout,
        "min_probability": min_probability,
        "min_confidence": min_confidence,
        "min_margin": min_margin,
        "max_state_chars": max(256, min(max_state_chars, 100_000)),
        "preserve_unknown_tools": _as_bool(
            _setting(ctx, "preserve_unknown_tools"), DEFAULTS["preserve_unknown_tools"]
        ),
    }


def _flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, Sequence) or isinstance(content, (bytes, bytearray)):
        return str(content or "")
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, Mapping):
            text = part.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts)


def _recent_state(
    user_message: Any,
    history: Sequence[Mapping[str, Any]] | None,
    max_chars: int,
    platform: str,
) -> dict[str, Any]:
    selected: list[tuple[int, str, str]] = []
    prior_users = 0
    assistant_found = False
    rows = history or ()
    for index in range(len(rows) - 1, -1, -1):
        message = rows[index]
        if not isinstance(message, Mapping):
            continue
        role = message.get("role")
        text = _flatten_content(message.get("content"))
        # Hermes includes this same user turn as the final assembled history row.
        if index == len(rows) - 1 and role == "user":
            continue
        if role == "user" and prior_users < 2:
            selected.append((index, "user", text))
            prior_users += 1
        elif (
            role == "assistant"
            and not assistant_found
            and not message.get("tool_calls")
            and text.strip()
        ):
            selected.append((index, "assistant", text))
            assistant_found = True
        if prior_users == 2 and assistant_found:
            break

    current = _flatten_content(user_message)[:max_chars]
    remaining = max(0, max_chars - len(current))
    bounded: list[tuple[int, str, str]] = []
    for index, role, text in sorted(selected, reverse=True):
        if remaining <= 0:
            break
        text = text[:remaining]
        remaining -= len(text)
        if text:
            bounded.append((index, role, text))
    return {
        "recent_context": [
            {"role": role, "content": text} for _index, role, text in sorted(bounded)
        ],
        "user_message": current,
        "platform": platform,
    }


def _system_one(
    endpoint: str,
    model: str,
    state: Mapping[str, Any],
    questions: Mapping[str, Any],
    timeout: float,
) -> Any:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps({"state": state, "model": model, "questions": questions}).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read(1_000_001)
    if len(raw) > 1_000_000:
        raise ValueError("System One response exceeded 1 MB")
    payload = json.loads(raw)
    answers = payload.get("answers") if isinstance(payload, Mapping) else None
    if not isinstance(answers, Mapping):
        raise TypeError("System One response has no answers")
    return answers


def _validated_choice(answer: Any, settings: Mapping[str, Any]) -> str | None:
    if not isinstance(answer, Mapping) or answer.get("type") != "choice":
        return None
    selected = answer.get("choice")
    probabilities = answer.get("probabilities")
    confidence = answer.get("confidence")
    if selected not in _PROFILES or not isinstance(probabilities, Mapping):
        return None
    if set(probabilities) != set(_PROFILES):
        return None
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return None
    if not math.isfinite(confidence):
        return None
    values = list(probabilities.values())
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or value > 1
        for value in values
    ):
        return None
    if abs(sum(values) - 1.0) > 0.02:
        return None
    ranked = sorted(
        ((float(probability), name) for name, probability in probabilities.items()), reverse=True
    )
    top_probability, top_choice = ranked[0]
    margin = top_probability - ranked[1][0]
    if top_choice != selected:
        return None
    if top_probability < settings["min_probability"] or confidence < settings["min_confidence"]:
        return None
    if margin < settings["min_margin"]:
        return None
    return str(selected)


def _key(session_id: Any, turn_id: Any) -> tuple[str, str] | None:
    session = str(session_id or "")
    turn = str(turn_id or "")
    return (session, turn) if session and turn else None


def _prune_decisions(now: float) -> None:
    expired = [
        key for key, value in _decisions.items() if now - value.created_at > _DECISION_TTL_SECONDS
    ]
    for key in expired:
        _decisions.pop(key, None)
    if len(_decisions) > _MAX_DECISIONS:
        oldest = sorted(_decisions, key=lambda key: _decisions[key].created_at)
        for key in oldest[: len(_decisions) - _MAX_DECISIONS]:
            _decisions.pop(key, None)


def _remember(session_id: Any, turn_id: Any, profile: str) -> None:
    key = _key(session_id, turn_id)
    if key is None:
        return
    now = time.monotonic()
    with _decisions_lock:
        _prune_decisions(now)
        _decisions[key] = Decision(profile=profile, created_at=now)


def _forget_turn(session_id: Any, turn_id: Any) -> None:
    key = _key(session_id, turn_id)
    if key is not None:
        with _decisions_lock:
            _decisions.pop(key, None)


def _forget_session(session_id: Any) -> None:
    session = str(session_id or "")
    if not session:
        return
    with _decisions_lock:
        for key in [key for key in _decisions if key[0] == session]:
            _decisions.pop(key, None)


def _tool_name(schema: Any) -> str:
    if not isinstance(schema, Mapping):
        return ""
    function = schema.get("function")
    if isinstance(function, Mapping):
        return str(function.get("name") or "")
    return str(schema.get("name") or "")


def _filter_tools(tools: list[Any], profile: str, preserve_unknown: bool) -> list[Any]:
    if profile in {"automation", "full"}:
        return list(tools)
    allowed = _PROFILE_TOOLS.get(profile)
    if allowed is None:
        return list(tools)
    filtered: list[Any] = []
    for schema in tools:
        name = _tool_name(schema)
        if name in allowed or (preserve_unknown and name not in _KNOWN_BUILTINS):
            filtered.append(schema)
    return filtered


def _pre_llm_call(ctx: Any, **event: Any) -> None:
    """Classify once per turn. Returning no context leaves the user message untouched."""
    try:
        settings = _settings(ctx)
        if not settings["enabled"]:
            return
        key = _key(event.get("session_id"), event.get("turn_id"))
        if key is None:
            return
        with _decisions_lock:
            if key in _decisions:
                return
        state = _recent_state(
            event.get("user_message"),
            event.get("conversation_history"),
            settings["max_state_chars"],
            str(event.get("platform") or ""),
        )
        answers = _system_one(
            settings["endpoint"],
            settings["kev_model"],
            state,
            {
                "profile": {
                    "type": "choice",
                    "instructions": (
                        "Choose the narrowest sufficient profile. Use full only for requests "
                        "requiring at least three distinct profile families."
                    ),
                    "criteria": _PROFILE_DESCRIPTIONS,
                }
            },
            settings["timeout"],
        )
        profile = _validated_choice(answers.get("profile"), settings)
        if profile is not None:
            _remember(event.get("session_id"), event.get("turn_id"), profile)
            logger.info("Kev routed turn %s to %s", event.get("turn_id"), profile)
        else:
            logger.info(
                "Kev routing was uncertain for turn %s; preserving all tools",
                event.get("turn_id"),
            )
    except Exception:
        logger.debug("Kev profile routing failed open", exc_info=True)


def _llm_request(ctx: Any, **event: Any) -> dict[str, Any] | None:
    """Filter only the effective request's tool list; all execution policy stays in Hermes."""
    try:
        settings = _settings(ctx)
        if not settings["enabled"]:
            return None
        key = _key(event.get("session_id"), event.get("turn_id"))
        if key is None:
            return None
        with _decisions_lock:
            decision = _decisions.get(key)
        if decision is None:
            return None
        request = event.get("request")
        if not isinstance(request, dict) or not isinstance(request.get("tools"), list):
            return None
        tools = request["tools"]
        filtered = _filter_tools(tools, decision.profile, settings["preserve_unknown_tools"])
        if len(filtered) == len(tools):
            return None
        updated = dict(request)
        updated["tools"] = filtered
        return {
            "request": updated,
            "source": PLUGIN_NAME,
            "reason": f"Kev profile {decision.profile}",
        }
    except Exception:
        logger.debug("Kev request filtering failed open", exc_info=True)
        return None


def register(ctx: Any) -> None:
    """Register hooks and middleware without network or filesystem side effects."""
    ctx.register_hook("pre_llm_call", lambda **event: _pre_llm_call(ctx, **event))
    ctx.register_hook(
        "post_llm_call",
        lambda **event: _forget_turn(event.get("session_id"), event.get("turn_id")),
    )
    ctx.register_hook("on_session_end", lambda **event: _forget_session(event.get("session_id")))
    ctx.register_middleware("llm_request", lambda **event: _llm_request(ctx, **event))


__all__ = ["register"]
