"""统一的 LLM 后端适配层。"""

from __future__ import annotations

import json
import os
from typing import Dict, List
from urllib import request as urllib_request
from urllib.parse import urlparse

from core.config_loader import load_llm_config


def resolve_api_key(env_name: str | None, fallback_value: str = "") -> str:
    if fallback_value:
        return fallback_value
    if not env_name:
        return ""

    env_value = os.getenv(env_name, "")
    if env_value:
        return env_value

    if os.name == "nt":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                return str(winreg.QueryValueEx(key, env_name)[0])
        except Exception:
            return ""
    return ""


def resolve_backend_config(config: Dict) -> Dict:
    provider_name = os.getenv("CSDM_LLM_PROVIDER", config.get("active_provider", "ollama_cloud"))
    provider_config = config.get(provider_name)
    if isinstance(provider_config, dict):
        return provider_config
    return config.get("backend", {})


class LLMBackend:
    """面向 Ollama 云端或本地 OpenAI 兼容接口的最小封装。"""

    def __init__(self, config: Dict | None = None) -> None:
        self.config = config or load_llm_config()
        backend = resolve_backend_config(self.config)
        self.base_url = backend["base_url"]
        self.model = backend["model"]
        self.temperature = backend["temperature"]
        self.max_tokens = backend["max_tokens"]
        self.provider = backend.get("provider", "")

        api_key_env = backend.get("api_key_env")
        api_key = resolve_api_key(api_key_env, backend.get("api_key", ""))
        if not api_key:
            raise ValueError(f"未设置 LLM API Key，请配置环境变量 {api_key_env}")

        self.client = None
        if self.provider != "local_ollama":
            try:
                from openai import OpenAI
            except Exception as exc:
                raise ValueError("当前环境未安装 openai 依赖") from exc
            self.client = OpenAI(base_url=self.base_url, api_key=api_key)

    def _json_output_budget(self) -> int:
        return max(int(self.max_tokens), 4096)

    def _chat_local_ollama(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
        max_tokens_override: int | None = None,
    ) -> str:
        native_base_url = self.base_url.rstrip("/")
        if native_base_url.endswith("/v1"):
            native_base_url = native_base_url[:-3].rstrip("/")
        parsed_base_url = urlparse(native_base_url)
        use_direct_connection = parsed_base_url.hostname in {"127.0.0.1", "localhost", "::1"}
        num_predict = int(max_tokens_override or (self._json_output_budget() if json_mode else self.max_tokens))

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": num_predict,
            },
        }
        if json_mode:
            payload["format"] = "json"

        req = urllib_request.Request(
            url=f"{native_base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        opener = (
            urllib_request.build_opener(urllib_request.ProxyHandler({}))
            if use_direct_connection
            else urllib_request.build_opener()
        )
        with opener.open(req, timeout=300) as response:
            body = json.loads(response.read().decode("utf-8"))
        return body.get("message", {}).get("content", "")

    def chat(self, system_prompt: str, user_prompt: str, max_tokens_override: int | None = None) -> str:
        if self.provider == "local_ollama":
            return self._chat_local_ollama(
                system_prompt,
                user_prompt,
                json_mode=False,
                max_tokens_override=max_tokens_override,
            )

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=int(max_tokens_override or self.max_tokens),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
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

    def generate_json(self, system_prompt: str, user_prompt: str) -> Dict | List:
        if self.provider == "local_ollama":
            text = self._chat_local_ollama(
                system_prompt,
                user_prompt,
                json_mode=True,
                max_tokens_override=self._json_output_budget(),
            )
            extracted = self._extract_json_text(text)
            try:
                return json.loads(extracted)
            except json.JSONDecodeError:
                repaired = self._extract_json_text(self._repair_json(extracted))
                return json.loads(repaired)

        text = self.chat(system_prompt, user_prompt)
        extracted = self._extract_json_text(text)
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            repaired = self._extract_json_text(self._repair_json(extracted))
            return json.loads(repaired)
