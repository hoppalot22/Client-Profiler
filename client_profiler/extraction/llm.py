from __future__ import annotations

import json
import os
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
        self.last_error_detail: str | None = None
        self._base_options = self._build_base_options()
        self._low_mem_retry_enabled = self._env_bool("CP_OLLAMA_LOW_MEM_RETRY", default=True)
        self._low_mem_force_cpu = self._env_bool("CP_OLLAMA_LOW_MEM_FORCE_CPU", default=True)
        self._low_mem_num_ctx = self._env_int("CP_OLLAMA_LOW_MEM_NUM_CTX") or 1024

    def extract_structured(self, prompt: str) -> dict:
        urls = self._candidate_generate_urls()
        last_http_status: int | str | None = None
        attempted_low_mem_retry = False

        for use_low_mem in (False, True):
            if use_low_mem and (not self._low_mem_retry_enabled or attempted_low_mem_retry):
                continue
            if use_low_mem:
                attempted_low_mem_retry = True

            payload = self._build_generate_payload(prompt, json_mode=True, low_mem=use_low_mem)
            payload_no_json_mode = self._build_generate_payload(prompt, json_mode=False, low_mem=use_low_mem)

            for payload_mode in (payload, payload_no_json_mode):
                for index, url in enumerate(urls):
                    is_last_url = index == (len(urls) - 1)
                    try:
                        response = requests.post(
                            url,
                            json=payload_mode,
                            timeout=self.timeout,
                        )
                        response.raise_for_status()
                        response_payload = response.json()
                        raw = response_payload.get("response", "{}")
                        if not raw:
                            self.last_error = "empty_model_response"
                            continue
                        if isinstance(raw, dict):
                            self.last_error = None
                            self.last_error_detail = None
                            return raw
                        parsed = self._parse_structured_json(raw)
                        if parsed:
                            self.last_error = None
                            self.last_error_detail = None
                            return parsed
                        self.last_error = "invalid_json_response"
                        self.last_error_detail = None
                        continue
                    except requests.Timeout:
                        self.last_error = "request_timed_out"
                        self.last_error_detail = None
                        return {}
                    except requests.HTTPError as exc:
                        status = exc.response.status_code if exc.response is not None else "unknown"
                        detail = self._extract_error_detail(exc.response)
                        last_http_status = status
                        if status == 404 and self._looks_like_model_not_found(exc.response) and not self._fallback_model_checked:
                            self._fallback_model_checked = True
                            fallback = self._pick_fallback_model()
                            if fallback:
                                self.model = fallback
                                return self.extract_structured(prompt)
                        if status == 404 and not is_last_url:
                            continue
                        # Retry in non-JSON mode on 500s that can happen with strict format on small models.
                        if status == 500 and payload_mode is payload:
                            continue
                        if status == 500 and not use_low_mem and self._is_memory_allocation_error(detail):
                            # Try a low-memory configuration once before surfacing failure.
                            break
                        self.last_error = f"http_error_{status}"
                        self.last_error_detail = detail or None
                        return {}
                    except requests.RequestException:
                        self.last_error = "request_error"
                        self.last_error_detail = None
                        return {}
                else:
                    continue
                # Break from payload-mode loop to attempt low-memory retry.
                break
            else:
                continue

            if use_low_mem:
                break

        if last_http_status is not None:
            self.last_error = f"http_error_{last_http_status}"
            return {}
        self.last_error = "request_error"
        self.last_error_detail = None
        return {}

    def _parse_structured_json(self, raw: str) -> dict:
        text = str(raw or "").strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            pass

        # Handle markdown code fences and extra prose around JSON.
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        first = text.find("{")
        last = text.rfind("}")
        if first >= 0 and last > first:
            candidate = text[first : last + 1]
            try:
                parsed = json.loads(candidate)
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                return {}
        return {}

    def list_models(self) -> list[str]:
        for index, url in enumerate(self._candidate_tags_urls()):
            is_last_url = index == (len(self._candidate_tags_urls()) - 1)
            try:
                response = requests.get(url, timeout=self.timeout)
                response.raise_for_status()
                payload = response.json()
            except requests.Timeout:
                self.last_error = "request_timed_out"
                self.last_error_detail = None
                return []
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else "unknown"
                if status == 404 and not is_last_url:
                    continue
                self.last_error = f"http_error_{status}"
                self.last_error_detail = self._extract_error_detail(exc.response) or None
                return []
            except (requests.RequestException, ValueError):
                self.last_error = "request_error"
                self.last_error_detail = None
                return []

            rows = payload.get("models") if isinstance(payload, dict) else []
            names = [str(row.get("name") or "").strip() for row in rows if isinstance(row, dict)]
            names = [name for name in names if name]
            self.last_error = None
            self.last_error_detail = None
            return names

        self.last_error = "request_error"
        self.last_error_detail = None
        return []

    def pull_model(self, model_name: str) -> tuple[bool, str]:
        model = str(model_name or "").strip()
        if not model:
            return False, "Model name is required."

        urls = self._candidate_pull_urls()
        for index, url in enumerate(urls):
            is_last_url = index == (len(urls) - 1)
            try:
                response = requests.post(
                    url,
                    json={"model": model, "stream": False},
                    timeout=max(self.timeout, 1800),
                )
                response.raise_for_status()
            except requests.Timeout:
                self.last_error = "request_timed_out"
                self.last_error_detail = None
                return False, "Download timed out. Try running 'ollama pull <model>' in a terminal."
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else "unknown"
                if status == 404 and not is_last_url:
                    continue
                self.last_error = f"http_error_{status}"
                detail = self._extract_error_detail(exc.response)
                self.last_error_detail = detail or None
                if detail:
                    return False, f"Local model service returned HTTP error ({status}): {detail}"
                return False, f"Local model service returned HTTP error ({status})."
            except requests.RequestException:
                self.last_error = "request_error"
                self.last_error_detail = None
                return False, "Could not reach local model service."

            self.last_error = None
            self.last_error_detail = None
            return True, "Model downloaded successfully."

        self.last_error = "request_error"
        self.last_error_detail = None
        return False, "Could not reach local model service."

    def _candidate_generate_urls(self) -> list[str]:
        base = self.base_url.rstrip("/")
        candidates = [f"{base}/api/generate"]

        lowered = base.lower()
        if lowered.endswith("/v1"):
            candidates.append(f"{base[:-3].rstrip('/')}/api/generate")
        if lowered.endswith("/api"):
            candidates.append(f"{base[:-4].rstrip('/')}/api/generate")

        deduped: list[str] = []
        seen: set[str] = set()
        for item in candidates:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        return deduped

    def _candidate_tags_urls(self) -> list[str]:
        base = self.base_url.rstrip("/")
        candidates = [f"{base}/api/tags"]

        lowered = base.lower()
        if lowered.endswith("/v1"):
            candidates.append(f"{base[:-3].rstrip('/')}/api/tags")
        if lowered.endswith("/api"):
            candidates.append(f"{base[:-4].rstrip('/')}/api/tags")

        deduped: list[str] = []
        seen: set[str] = set()
        for item in candidates:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        return deduped

    def _candidate_pull_urls(self) -> list[str]:
        base = self.base_url.rstrip("/")
        candidates = [f"{base}/api/pull"]

        lowered = base.lower()
        if lowered.endswith("/v1"):
            candidates.append(f"{base[:-3].rstrip('/')}/api/pull")
        if lowered.endswith("/api"):
            candidates.append(f"{base[:-4].rstrip('/')}/api/pull")

        deduped: list[str] = []
        seen: set[str] = set()
        for item in candidates:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        return deduped

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
        names = self.list_models()
        if not names:
            return None

        preferred_prefix = self._configured_model.split(":", 1)[0].strip().lower()
        for name in names:
            if name.lower().split(":", 1)[0] == preferred_prefix:
                return name
        return names[0]

    def _extract_error_detail(self, response: requests.Response | None) -> str:
        if response is None:
            return ""
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        if isinstance(payload, dict):
            text = str(payload.get("error") or payload.get("message") or "").strip()
            if text:
                return text[:400]
        return str(response.text or "").strip()[:400]

    def _build_generate_payload(self, prompt: str, json_mode: bool, low_mem: bool) -> dict:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": self._options_for_request(low_mem=low_mem),
        }
        if json_mode:
            payload["format"] = "json"
        return payload

    def _options_for_request(self, low_mem: bool) -> dict:
        options = dict(self._base_options)
        if low_mem:
            options["num_ctx"] = max(128, self._low_mem_num_ctx)
            if self._low_mem_force_cpu:
                options["num_gpu"] = 0
        return options

    def _build_base_options(self) -> dict:
        options: dict[str, int | float] = {
            "temperature": 0,
            "top_p": 1,
            "seed": 42,
        }
        num_ctx = self._env_int("CP_OLLAMA_NUM_CTX")
        if num_ctx:
            options["num_ctx"] = num_ctx
        num_gpu = self._env_int("CP_OLLAMA_NUM_GPU")
        if num_gpu is not None:
            options["num_gpu"] = num_gpu
        return options

    def _is_memory_allocation_error(self, detail: str) -> bool:
        text = str(detail or "").lower()
        return "unable to allocate" in text or "out of memory" in text

    def _env_int(self, name: str) -> int | None:
        raw = str(os.getenv(name, "")).strip()
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError:
            return None

    def _env_bool(self, name: str, default: bool) -> bool:
        raw = str(os.getenv(name, "")).strip().lower()
        if not raw:
            return default
        return raw in {"1", "true", "yes", "on"}
