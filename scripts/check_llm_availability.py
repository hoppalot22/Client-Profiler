from __future__ import annotations

import sys

import requests

MODEL = "qwen2.5:1.5b"
BASE_URL = "http://localhost:11434"


def main() -> int:
    url = f"{BASE_URL.rstrip('/')}/api/tags"
    try:
        response = requests.get(url, timeout=3)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"[WARN] LLM check failed ({exc}). Web app will still start; AI features may use fallback paths.")
        return 0

    try:
        payload = response.json() if response.content else {}
    except ValueError:
        payload = {}

    model_rows = payload.get("models") if isinstance(payload, dict) else []
    names = [str(row.get("name") or "").strip() for row in model_rows if isinstance(row, dict)]
    names = [name for name in names if name]

    if not names:
        print("[WARN] Ollama is reachable but no models are installed. LLM-backed features may fall back.")
        return 0

    if any(name.lower() == MODEL.lower() for name in names):
        print(f"[OK] LLM check passed: found configured model {MODEL}")
        return 0

    shown = ", ".join(names[:5])
    print(f"[WARN] Ollama is reachable but configured model {MODEL} was not found. Available: {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
