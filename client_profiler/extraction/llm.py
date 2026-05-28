from __future__ import annotations

import json
from abc import ABC, abstractmethod

import requests


class LLMClient(ABC):
    @abstractmethod
    def extract_structured(self, prompt: str) -> dict:
        raise NotImplementedError


class OllamaClient(LLMClient):
    def __init__(self, base_url: str, model: str, timeout: int = 600) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._configured_model = model
        self._fallback_model_checked = False
        self.timeout = timeout
        self.last_error: str | None = None

    def extract_structured(self, prompt: str) -> dict:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0,
                "top_p": 1,
                "seed": 42,
            },
        }
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
            raw = payload.get("response", "{}")
            if not raw:
                self.last_error = "empty_model_response"
                return {}
            if isinstance(raw, dict):
                self.last_error = None
                return raw
            parsed = json.loads(raw)
            self.last_error = None
            return parsed
        except requests.Timeout:
            self.last_error = "request_timed_out"
            return {}
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "unknown"
            if status == 404 and self._looks_like_model_not_found(exc.response) and not self._fallback_model_checked:
                self._fallback_model_checked = True
                fallback = self._pick_fallback_model()
                if fallback:
                    self.model = fallback
                    return self.extract_structured(prompt)
            self.last_error = f"http_error_{status}"
            return {}
        except requests.RequestException:
            self.last_error = "request_error"
            return {}
        except json.JSONDecodeError:
            self.last_error = "invalid_json_response"
            return {}

    def _looks_like_model_not_found(self, response: requests.Response | None) -> bool:
        if response is None:
            return False
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        message = str(payload.get("error") or response.text or "").lower()
        return "model" in message and "not found" in message

    def _pick_fallback_model(self) -> str | None:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError):
            return None

        models = payload.get("models") if isinstance(payload, dict) else []
        model_rows = models if isinstance(models, list) else []
        names = [str(row.get("name") or "").strip() for row in model_rows if isinstance(row, dict)]
        names = [name for name in names if name]
        if not names:
            return None

        preferred_prefix = self._configured_model.split(":", 1)[0].strip().lower()
        for name in names:
            if name.lower().split(":", 1)[0] == preferred_prefix:
                return name
        return names[0]
