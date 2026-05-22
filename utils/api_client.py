"""Unified API client for OpenAI-compatible embedding and chat completion APIs."""
from __future__ import annotations

import json
import os
import time
from typing import Any

import requests


class APIClient:
    def __init__(
        self,
        api_key_env: str = "CLOSEAI_API_KEY",
        api_url: str = "https://api.openai-proxy.org/v1",
        model: str = "gpt-4o",
        max_retries: int = 3,
        request_timeout: int = 300,
    ):
        self.api_key = os.environ.get(api_key_env, "").strip()
        if not self.api_key:
            raise ValueError(f"Missing API key in environment variable: {api_key_env}")
        self.api_url = api_url.rstrip("/")
        self.model = model
        self.max_retries = max_retries
        self.request_timeout = request_timeout

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def embed_batch(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        url = f"{self.api_url}/embeddings"
        payload = {"model": model or self.model, "input": texts}
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(
                    url,
                    headers=self._headers(),
                    json=payload,
                    timeout=self.request_timeout,
                )
                resp.raise_for_status()
                data = resp.json()["data"]
                ordered = sorted(data, key=lambda x: x["index"])
                return [item["embedding"] for item in ordered]
            except Exception as e:
                if attempt == self.max_retries:
                    raise RuntimeError(f"Embedding API failed after {self.max_retries} retries: {e}")
                time.sleep(min(60, 2 ** (attempt - 1)))
        return []

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.0,
        response_format: dict | None = None,
    ) -> str:
        url = f"{self.api_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format:
            payload["response_format"] = response_format
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(
                    url,
                    headers=self._headers(),
                    json=payload,
                    timeout=self.request_timeout,
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except Exception as e:
                if attempt == self.max_retries:
                    raise RuntimeError(f"Chat API failed after {self.max_retries} retries: {e}")
                time.sleep(min(60, 2 ** (attempt - 1)))
        return ""

    def chat_completion_json(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        content = self.chat_completion(
            messages,
            model=model,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        return json.loads(content)
