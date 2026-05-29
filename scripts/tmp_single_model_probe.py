from __future__ import annotations

import json
import sys
import time

from client_profiler.extraction.llm import OllamaClient


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:1.5b"
    timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 45
    client = OllamaClient("http://127.0.0.1:11434", model, timeout=timeout)
    prompt = 'Return STRICT JSON only: {"is_present":true,"value":"ok","evidence":"simple"}. No extra keys.'
    start = time.perf_counter()
    result = client.extract_structured(prompt)
    elapsed = time.perf_counter() - start
    print(
        json.dumps(
            {
                "model": model,
                "timeout_seconds": timeout,
                "elapsed_seconds": round(elapsed, 3),
                "has_result": bool(result),
                "result": result,
                "last_error": client.last_error,
                "last_error_detail": getattr(client, "last_error_detail", None),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
