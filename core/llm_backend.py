"""单一 OpenAI 兼容 LLM 后端。"""

from __future__ import annotations

import os
import json
from typing import Any, Dict

from core.config_loader import load_llm_config


def resolve_config_value(env_name: str | None, fallback_value: str = "") -> str:
    """从环境变量或配置默认值读取 LLM 参数。"""
    if env_name:
        env_value = os.getenv(env_name, "").strip()
        if env_value:
            return env_value
    return str(fallback_value or "").strip()


def resolve_backend_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """返回当前唯一可用的 OpenAI 兼容后端配置。"""
    backend = config.get("backend", {})
    if not isinstance(backend, dict):
        raise ValueError("llm_config.yaml 缺少 backend 配置")
    provider = str(backend.get("provider") or "openai_compatible")
    if provider != "openai_compatible":
        raise ValueError("当前项目只支持 openai_compatible LLM 后端")
    return backend


def auto_llm_enabled() -> bool:
    """控制业务对象是否自动初始化 LLM，测试环境可关闭外部调用。"""
    return os.getenv("CSDM_panel_DISABLE_LLM_AUTO", "0").strip() != "1"


class LLMResponseError(ValueError):
    """模型未返回完整、非空的正文。"""


class LLMBackend:
    """面向当前项目唯一 LLM 接口的最小封装。"""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or load_llm_config()
        backend = resolve_backend_config(self.config)

        self.base_url = resolve_config_value(backend.get("base_url_env"), backend.get("base_url", ""))
        self.api_key = resolve_config_value(backend.get("api_key_env"), backend.get("api_key", ""))
        self.model = resolve_config_value(backend.get("model_env"), backend.get("model", ""))
        self.temperature = float(backend.get("temperature", 0.2))
        self.max_tokens = int(backend.get("max_tokens", 1800))
        self.timeout_seconds = int(backend.get("timeout_seconds", 180))
        extra_body_text = resolve_config_value(backend.get("extra_body_env"))
        self.extra_body = json.loads(extra_body_text) if extra_body_text else backend.get("extra_body", {})
        if not isinstance(self.extra_body, dict):
            raise ValueError("LLM backend.extra_body 必须是对象")
        self.last_response = None
        missing = []
        if not self.base_url:
            missing.append(str(backend.get("base_url_env") or "base_url"))
        if not self.api_key:
            missing.append(str(backend.get("api_key_env") or "api_key"))
        if not self.model:
            missing.append(str(backend.get("model_env") or "model"))
        if missing:
            raise ValueError(f"LLM 配置不完整，请设置：{', '.join(missing)}")

        try:
            from openai import OpenAI
        except Exception as exc:
            raise ValueError("当前环境未安装 openai 依赖") from exc

        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout_seconds, max_retries=0)

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens_override: int | None = None,
        json_mode: bool = False,
    ) -> str:
        """调用当前模型并返回完整响应正文。"""
        request_payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": int(max_tokens_override or self.max_tokens),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "extra_body": self.extra_body,
        }
        self.last_response = None
        if json_mode:
            request_payload["response_format"] = {"type": "json_object"}
        response = self.client.chat.completions.create(**request_payload)
        usage = getattr(response, "usage", None)
        self.last_response = {"model": getattr(response, "model", self.model),
                              "usage": usage.model_dump() if usage is not None else None,
                              "finish_reason": None, "content": None}
        if not response.choices:
            raise LLMResponseError("LLM 未返回任何应答")
        choice = response.choices[0]
        content = choice.message.content or ""
        self.last_response.update(finish_reason=choice.finish_reason, content=content)
        if choice.finish_reason != "stop":
            raise LLMResponseError(f"LLM 回答未完整结束（finish_reason={choice.finish_reason}）")
        if not content.strip():
            raise LLMResponseError("LLM 返回空正文")
        return content
