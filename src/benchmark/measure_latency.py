"""Measure response time for verified questions on the FALCON orchestrator."""
from __future__ import annotations

import argparse
import csv
import json

from .adapters import run_orchestrator
from .cost_common import add_common_arguments, default_output, load_selected
from .evaluation import compare


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--max-attempts", type=int, default=3,
                        help="Maximum SQL generation attempts (default: 3)")
    args = parser.parse_args(argv)
    if args.max_attempts < 1:
        parser.error("--max-attempts must be at least 1")

    try:
        cases, database, _ = load_selected(args)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    # Exclude Hugging Face model initialization from the measured query times.
    import src.agents.request_agent  # noqa: F401
    import src.agents.query_agent  # noqa: F401
    import src.agents.response_agent  # noqa: F401

    rows = []
    for index, case in enumerate(cases, start=1):
        print(f"[FALCON {index}/{len(cases)}] {case.category}: {case.query_id}", flush=True)
        result = run_orchestrator(case, database, args.max_attempts)
        category = "boolean" if case.category == "boolean" else case.category
        correct = compare(case.expected, result.get("result"), category) if result.get("status") == "success" else None
        rows.append({
            "system": "orchestrator",
            "category": case.category,
            "query_id": case.query_id,
            "source_file": case.source_file,
            "source_row": case.source_row,
            "question": case.question,
            "expected_result": case.expected,
            "previous_pipeline_result": case.metadata.get("Answer Pipeline"),
            "previously_correct": True,
            "result": result.get("result"),
            "correct": correct,
            "status": result.get("status"),
            "seconds": result.get("seconds"),
            "attempts": result.get("attempts"),
            "note": result.get("note"),
            "error": result.get("error"),
        })
        print(f"  status={result.get('status')} correct={correct} seconds={result.get('seconds')}", flush=True)

    output = default_output(args, "response_times_falcon.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, default=str)
                             if isinstance(value, (dict, list, tuple)) else value
                             for key, value in row.items()})
    print(f"Saved {len(rows)} FALCON query measurements to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
