from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Mapping

from .catalog import DETERMINISTIC_PROVIDER, ResolvedLLMSelection


PostJson = Callable[[str, dict, Mapping[str, str], float], dict]


@dataclass(frozen=True)
class LLMResponse:
    text: str
    provider: str
    model: str


class LLMClientError(RuntimeError):
    pass


class SimpleLLMClient:
    """Small HTTP client for the three supported providers.

    API keys are supplied at runtime and are never stored in selection metadata.
    Tests can inject ``post_json`` to avoid real network calls.
    """

    def __init__(
        self,
        selection: ResolvedLLMSelection,
        *,
        api_key: str | None = None,
        timeout: float = 30.0,
        post_json: PostJson | None = None,
    ) -> None:
        self.selection = selection
        self.api_key = api_key or ""
        self.timeout = timeout
        self._post_json_impl = post_json or _urllib_post_json

    def generate_text(self, prompt: str, *, system: str = "You are a financial research assistant.") -> LLMResponse:
        provider = self.selection.resolved_provider
        if provider == DETERMINISTIC_PROVIDER:
            return LLMResponse(text="", provider=provider, model=self.selection.resolved_model)
        if not self.api_key:
            raise LLMClientError(f"No API key available for provider '{provider}'.")
        if provider == "openai":
            text = self._openai_chat(prompt, system)
        elif provider == "anthropic":
            text = self._anthropic_messages(prompt, system)
        elif provider == "google":
            text = self._google_generate(prompt, system)
        else:
            raise LLMClientError(f"Unsupported resolved provider '{provider}'.")
        return LLMResponse(text=text, provider=provider, model=self.selection.resolved_model)

    def _openai_chat(self, prompt: str, system: str) -> str:
        payload = {
            "model": self.selection.resolved_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        response = self._post_json_impl("https://api.openai.com/v1/chat/completions", payload, headers, self.timeout)
        return response["choices"][0]["message"]["content"]

    def _anthropic_messages(self, prompt: str, system: str) -> str:
        payload = {
            "model": self.selection.resolved_model,
            "max_tokens": 1200,
            "temperature": 0.2,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        response = self._post_json_impl("https://api.anthropic.com/v1/messages", payload, headers, self.timeout)
        content = response.get("content", [])
        if content and isinstance(content[0], dict):
            return str(content[0].get("text", ""))
        return json.dumps(response)

    def _google_generate(self, prompt: str, system: str) -> str:
        model = self.selection.resolved_model
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
        }
        response = self._post_json_impl(
            url,
            payload,
            {"Content-Type": "application/json", "x-goog-api-key": self.api_key},
            self.timeout,
        )
        return response["candidates"][0]["content"]["parts"][0]["text"]


def _urllib_post_json(url: str, payload: dict, headers: Mapping[str, str], timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=dict(headers),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, IndexError) as exc:
        raise LLMClientError(f"LLM provider call failed: {exc}") from exc
