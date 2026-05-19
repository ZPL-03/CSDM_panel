"""单一 OpenAI 兼容 LLM 后端。"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

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
        self.json_output_tokens = int(backend.get("json_output_tokens", max(self.max_tokens, 4096)))

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

        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout_seconds)

    def _json_output_budget(self) -> int:
        return max(int(self.json_output_tokens), int(self.max_tokens), 4096)

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens_override: int | None = None,
        json_mode: bool = False,
    ) -> str:
        """调用当前 LLM 生成普通文本。"""
        request_payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": int(max_tokens_override or self.max_tokens),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if json_mode:
            request_payload["response_format"] = {"type": "json_object"}
        try:
            response = self.client.chat.completions.create(**request_payload)
        except Exception as exc:
            if not json_mode or "response_format" not in str(exc):
                raise
            request_payload.pop("response_format", None)
            response = self.client.chat.completions.create(**request_payload)
        return response.choices[0].message.content or ""

    def _extract_json_text(self, text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.replace("json\n", "", 1).replace("JSON\n", "", 1)
            cleaned = cleaned.strip()

        candidates = [cleaned]
        array_start = cleaned.find("[")
        array_end = cleaned.rfind("]")
        if array_start != -1 and array_end != -1 and array_end > array_start:
            candidates.append(cleaned[array_start : array_end + 1])

        object_start = cleaned.find("{")
        object_end = cleaned.rfind("}")
        if object_start != -1 and object_end != -1 and object_end > object_start:
            candidates.append(cleaned[object_start : object_end + 1])

        for candidate in candidates:
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                continue
        return cleaned

    def _repair_json(self, broken_text: str) -> str:
        repair_system = "你是 JSON 修复器。请把用户提供的内容修复成合法 JSON，除 JSON 外不要输出任何说明。"
        repair_user = f"请修复为合法 JSON：\n{broken_text}"
        return self.chat(repair_system, repair_user, max_tokens_override=self._json_output_budget())

    def _parse_json_robust(self, text: str) -> Dict[str, Any] | List[Any]:
        """多级解析 LLM 输出，确保候选生成链路得到结构化 JSON。"""
        extracted = self._extract_json_text(text)
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            pass

        try:
            repaired = self._extract_json_text(self._repair_json(extracted))
            return json.loads(repaired)
        except Exception:
            pass

        for wrapper_start, wrapper_end in [("{", "}"), ("[", "]")]:
            depth = 0
            start = -1
            end = -1
            for index, char in enumerate(text):
                if char == wrapper_start:
                    if depth == 0:
                        start = index
                    depth += 1
                elif char == wrapper_end:
                    depth -= 1
                    if depth == 0:
                        end = index + 1
                        break
            if start != -1 and end != -1 and end > start:
                candidate = text[start:end]
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, (dict, list)):
                    return parsed
        return {}

    def generate_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any] | List[Any]:
        """调用当前 LLM 并解析 JSON 输出。"""
        text = self.chat(
            system_prompt,
            user_prompt,
            max_tokens_override=self._json_output_budget(),
            json_mode=True,
        )
        return self._parse_json_robust(text)
