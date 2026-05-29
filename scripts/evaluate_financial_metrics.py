from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from client_profiler import ClientProfiler, ProfilerConfig


def _load_expected(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Expected metrics file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Expected metrics file must be a JSON object keyed by client")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate client financial metrics against expected totals")
    parser.add_argument("--db", type=Path, default=Path("./data/profiler_financial_variants.db"))
    parser.add_argument("--expected", type=Path, default=Path("./data/financial_variants_expected.json"))
    parser.add_argument("--tolerance", type=float, default=1.0)
    args = parser.parse_args()

    expected = _load_expected(args.expected)
    config = ProfilerConfig(db_path=args.db, data_dir=args.db.parent)
    profiler = ClientProfiler(config)

    evaluations = []
    failures = []

    for client_name, row in expected.items():
        expected_revenue = float(row.get("revenue") or 0.0)
        expected_cost = float(row.get("cost") or 0.0)
        expected_profit = float(row.get("profit") or (expected_revenue - expected_cost))

        actual = profiler.storage.get_client_metrics(client_name)
        actual_revenue = float(actual.get("revenue_total") or 0.0)
        actual_cost = float(actual.get("cost_total") or 0.0)
        actual_profit = float(actual.get("gross_profit_total") or (actual_revenue - actual_cost))

        revenue_delta = round(actual_revenue - expected_revenue, 2)
        cost_delta = round(actual_cost - expected_cost, 2)
        profit_delta = round(actual_profit - expected_profit, 2)

        ok = all(abs(delta) <= float(args.tolerance) for delta in [revenue_delta, cost_delta, profit_delta])
        record = {
            "client_name": client_name,
            "ok": ok,
            "expected": {
                "revenue": expected_revenue,
                "cost": expected_cost,
                "profit": expected_profit,
            },
            "actual": {
                "revenue": actual_revenue,
                "cost": actual_cost,
                "profit": actual_profit,
                "financial_documents": int(actual.get("financial_documents") or 0),
                "documents_with_financials": int(actual.get("documents_with_financials") or 0),
                "gross_margin_pct": float(actual.get("gross_margin_pct") or 0.0),
            },
            "delta": {
                "revenue": revenue_delta,
                "cost": cost_delta,
                "profit": profit_delta,
            },
        }
        evaluations.append(record)
        if not ok:
            failures.append(record)

    summary = {
        "db": str(args.db),
        "expected": str(args.expected),
        "tolerance": args.tolerance,
        "clients": len(expected),
        "passed": len(expected) - len(failures),
        "failed": len(failures),
        "results": evaluations,
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
