from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest import TestCase, mock


def _load_plugin():
    path = Path(__file__).parents[1] / "__init__.py"
    spec = importlib.util.spec_from_file_location("hermes_kev_router", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


plugin = _load_plugin()


def _schema(name):
    return {"type": "function", "function": {"name": name, "description": name, "parameters": {}}}


def _choice(selected, probability=0.8, confidence=0.8):
    remainder = (1.0 - probability) / (len(plugin._PROFILES) - 1)
    probabilities = {name: remainder for name in plugin._PROFILES}
    probabilities[selected] = probability
    return {
        "type": "choice",
        "choice": selected,
        "probabilities": probabilities,
        "confidence": confidence,
    }


class Context:
    def __init__(self, settings=None):
        self.settings = settings or {}
        self.config_reads = []
        self.hooks = {}
        self.middleware = {}

    def get_config(self, key, default=None):
        self.config_reads.append(key)
        return self.settings.get(key, default)

    def register_hook(self, name, callback):
        self.hooks[name] = callback

    def register_middleware(self, name, callback):
        self.middleware[name] = callback


class RouterTests(TestCase):
    def setUp(self):
        plugin._decisions.clear()

    def test_validated_choice_rejects_uncertain_answer(self):
        settings = dict(plugin.DEFAULTS)
        answer = _choice("coding", probability=0.3, confidence=0.8)
        self.assertIsNone(plugin._validated_choice(answer, settings))

    def test_settings_do_not_read_reserved_model_path(self):
        ctx = Context()
        settings = plugin._settings(ctx)
        self.assertEqual(settings["kev_model"], "kev-latest")
        self.assertIn("kev_model", ctx.config_reads)
        self.assertNotIn("model", ctx.config_reads)

    def test_weather_criteria_prefer_narrow_research_profile(self):
        ctx = Context()

        def classify(_endpoint, _model, _state, questions, _timeout):
            profile = questions["profile"]
            self.assertIn("weather", profile["criteria"]["research"])
            self.assertIn("three distinct", profile["instructions"])
            return {"profile": _choice("research")}

        with mock.patch.object(plugin, "_system_one", side_effect=classify):
            plugin._pre_llm_call(
                ctx,
                session_id="session",
                turn_id="weather",
                user_message="weather in Ridgewood 10 days",
                conversation_history=[],
                platform="cli",
            )
        self.assertEqual(plugin._decisions[("session", "weather")].profile, "research")

    def test_chat_keeps_skill_and_unknown_plugin_tools(self):
        tools = [
            _schema("terminal"),
            _schema("clarify"),
            _schema("skill_view"),
            _schema("weather_plugin"),
        ]
        filtered = plugin._filter_tools(tools, "chat", preserve_unknown=True)
        self.assertEqual(
            [plugin._tool_name(schema) for schema in filtered],
            ["clarify", "skill_view", "weather_plugin"],
        )

    def test_unknown_tools_can_be_removed_explicitly(self):
        filtered = plugin._filter_tools([_schema("clarify"), _schema("plugin_tool")], "chat", False)
        self.assertEqual([plugin._tool_name(schema) for schema in filtered], ["clarify"])

    def test_pre_hook_and_middleware_share_one_turn_decision(self):
        ctx = Context()
        plugin.register(ctx)
        answer = {"profile": _choice("coding")}
        with mock.patch.object(plugin, "_system_one", return_value=answer) as classify:
            ctx.hooks["pre_llm_call"](
                session_id="session",
                turn_id="turn",
                user_message="fix the test",
                conversation_history=[],
                platform="cli",
            )
            ctx.hooks["pre_llm_call"](
                session_id="session",
                turn_id="turn",
                user_message="fix the test",
                conversation_history=[],
                platform="cli",
            )
        self.assertEqual(classify.call_count, 1)

        request = {"model": "test", "tools": [_schema("terminal"), _schema("memory")]}
        result = ctx.middleware["llm_request"](
            request=request,
            session_id="session",
            turn_id="turn",
        )
        self.assertEqual(
            [plugin._tool_name(schema) for schema in result["request"]["tools"]],
            ["terminal"],
        )

    def test_errors_fail_open(self):
        ctx = Context()
        plugin.register(ctx)
        with mock.patch.object(plugin, "_system_one", side_effect=OSError("offline")):
            self.assertIsNone(
                ctx.hooks["pre_llm_call"](
                    session_id="session",
                    turn_id="turn",
                    user_message="hello",
                    conversation_history=[],
                    platform="cli",
                )
            )
        request = {"tools": [_schema("terminal")]}
        self.assertIsNone(
            ctx.middleware["llm_request"](
                request=request,
                session_id="session",
                turn_id="turn",
            )
        )

    def test_post_turn_forgets_decision(self):
        ctx = Context()
        plugin.register(ctx)
        plugin._remember("session", "turn", "chat")
        ctx.hooks["post_llm_call"](session_id="session", turn_id="turn")
        self.assertNotIn(("session", "turn"), plugin._decisions)

    def test_response_style_tool_name_is_supported(self):
        self.assertEqual(plugin._tool_name({"type": "function", "name": "terminal"}), "terminal")

    def test_recent_state_does_not_duplicate_current_user_row(self):
        state = plugin._recent_state(
            "current",
            [
                {"role": "user", "content": "prior"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "current"},
            ],
            1000,
            "cli",
        )
        self.assertEqual(state["user_message"], "current")
        self.assertEqual(
            state["recent_context"],
            [
                {"role": "user", "content": "prior"},
                {"role": "assistant", "content": "answer"},
            ],
        )
