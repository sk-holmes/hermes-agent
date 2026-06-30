"""Usage/context accounting for the TUI gateway payload."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def server():
    with patch.dict(
        "sys.modules",
        {
            "hermes_constants": MagicMock(
                get_hermes_home=MagicMock(return_value="/tmp/hermes_test_usage")
            ),
            "hermes_cli.env_loader": MagicMock(),
            "hermes_cli.banner": MagicMock(),
            "hermes_state": MagicMock(),
        },
    ):
        yield importlib.import_module("tui_gateway.server")


def _agent_with_usage(*, last_prompt_tokens: int, total_tokens: int = 22_400_000):
    return SimpleNamespace(
        model="gpt-5.5",
        session_input_tokens=total_tokens,
        session_output_tokens=0,
        session_reasoning_tokens=0,
        session_prompt_tokens=total_tokens,
        session_completion_tokens=0,
        session_total_tokens=total_tokens,
        session_api_calls=3,
        context_compressor=SimpleNamespace(
            last_prompt_tokens=last_prompt_tokens,
            context_length=272_000,
            compression_count=1,
        ),
    )


def test_get_usage_does_not_use_cumulative_total_as_context_used(server):
    usage = server._get_usage(_agent_with_usage(last_prompt_tokens=0))

    assert usage["total"] == 22_400_000
    assert usage["context_max"] == 272_000
    assert usage["context_used"] == 0
    assert usage["context_percent"] == 0


def test_get_usage_clamps_post_compression_prompt_sentinel(server):
    usage = server._get_usage(_agent_with_usage(last_prompt_tokens=-1))

    assert usage["context_used"] == 0
    assert usage["context_percent"] == 0


def test_get_usage_reports_current_prompt_tokens_when_available(server):
    usage = server._get_usage(_agent_with_usage(last_prompt_tokens=136_000))

    assert usage["context_used"] == 136_000
    assert usage["context_percent"] == 50
