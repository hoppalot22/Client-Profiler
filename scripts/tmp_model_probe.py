from __future__ import annotations

import json
import time

from client_profiler.extraction.llm import OllamaClient

MODELS = ["qwen2.5:0.5b", "qwen3:0.6b", "qwen2.5:1.5b"]
PROMPT = (
    'Return STRICT JSON only: {"is_present":true,"value":"ok","evidence":"simple"}. '
    'No extra keys.'
)

rows = []
for model in MODELS:
    client = OllamaClient("http://127.0.0.1:11434", model, timeout=120)
    start = time.perf_counter()
    result = client.extract_structured(PROMPT)
    elapsed = time.perf_counter() - start
    rows.append(
        {
            "model": model,
            "elapsed_seconds": round(elapsed, 3),
            "has_result": bool(result),
            "result": result,
            "last_error": client.last_error,
            "last_error_detail": getattr(client, "last_error_detail", None),
        }
    )

print(json.dumps(rows, indent=2))
