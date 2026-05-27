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
        self.timeout = timeout
        self.last_error: str | None = None

    def extract_structured(self, prompt: str) -> dict:
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                },
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
            self.last_error = f"http_error_{status}"
            return {}
        except requests.RequestException:
            self.last_error = "request_error"
            return {}
        except json.JSONDecodeError:
            self.last_error = "invalid_json_response"
            return {}
