from types import SimpleNamespace

import pytest

from core.llm_backend import LLMBackend


def _backend(content: str, finish_reason: str):
    backend = object.__new__(LLMBackend)
    backend.model = "test-model"
    backend.temperature = 0.2
    backend.max_tokens = 1800
    backend.extra_body = {"thinking": {"type": "disabled"}}
    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(
            finish_reason=finish_reason, message=SimpleNamespace(content=content))])

    backend.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return backend, requests


def test_complete_llm_response_preserves_text_and_backend_options() -> None:
    backend, requests = _backend("| 方案 | 参数 |\n", "stop")
    assert backend.chat("system", "user") == "| 方案 | 参数 |\n"
    assert requests[0]["extra_body"] == {"thinking": {"type": "disabled"}}


@pytest.mark.parametrize("content,reason", [("", "length"), ("| 方案 |", "length"), ("", "stop"), ("", "content_filter")])
def test_incomplete_or_empty_llm_response_is_rejected(content, reason) -> None:
    backend, _ = _backend(content, reason)
    with pytest.raises(ValueError):
        backend.chat("system", "user")
