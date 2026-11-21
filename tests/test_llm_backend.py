from types import SimpleNamespace

import pytest

from core.llm_backend import LLMBackend, LLMResponseError


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
    backend, requests = _backend(content, reason)
    with pytest.raises(LLMResponseError):
        backend.chat("system", "user")
    assert len(requests) == 1
    assert backend.last_response["finish_reason"] == reason
    assert backend.last_response["content"] == content


def test_qwen_request_parameters_replace_provider_defaults(monkeypatch):
    import sys

    captured = {}
    def constructor(**kwargs):
        captured.update(kwargs)
        return object()
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=constructor))
    monkeypatch.setenv("PANEL_TEST_EXTRA_BODY", '{"chat_template_kwargs":{"enable_thinking":false}}')
    backend = LLMBackend({"backend": {"base_url": "http://localhost:8000/v1", "api_key": "test-only",
                                    "model": "qwen-test", "extra_body_env": "PANEL_TEST_EXTRA_BODY",
                                    "extra_body": {"thinking": {"type": "disabled"}}}})
    assert backend.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
    assert captured["max_retries"] == 0
    sender, requests = _backend("{}", "stop")
    sender.extra_body = backend.extra_body
    sender.chat("system", "user", max_tokens_override=4096)
    assert requests[0]["max_tokens"] == 4096
    assert requests[0]["extra_body"] == backend.extra_body
    monkeypatch.setenv("PANEL_TEST_EXTRA_BODY", "[]")
    with pytest.raises(ValueError):
        LLMBackend({"backend": {"extra_body_env": "PANEL_TEST_EXTRA_BODY"}})


def test_empty_content_retains_usage_for_budget_diagnosis():
    usage = {"completion_tokens": 4096, "completion_tokens_details": {"reasoning_tokens": 4096}}
    response = SimpleNamespace(choices=[SimpleNamespace(finish_reason="length", message=SimpleNamespace(content=""))],
                               model="qwen-test", usage=SimpleNamespace(model_dump=lambda: usage))
    backend, _ = _backend("", "length")
    backend.client.chat.completions.create = lambda **kwargs: response
    with pytest.raises(LLMResponseError):
        backend.chat("system", "user")
    assert backend.last_response["usage"] == usage
    assert backend.last_response["model"] == "qwen-test"
