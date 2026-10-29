"""Measure response time for four verified questions across the three systems.

Run with: python -m benchmark.measure_latency
"""
from __future__ import annotations

import argparse
import csv
import json

from .adapters import PMAgentAdapter, PM4PyLLMAdapter, run_orchestrator
from .cost_common import add_common_arguments, default_output, load_selected
from .evaluation import compare

VALID_SYSTEMS = {"orchestrator", "PMAgent", "pm4py"}


def _run(system, case, database, pma, pm4, max_attempts):
    if system == "orchestrator":
        return run_orchestrator(case, database, max_attempts)
    if system == "PMAgent":
        return pma.run(case)
    return pm4.run(case)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--systems", default="orchestrator,PMAgent,pm4py",
                        help="Comma-separated systems: orchestrator,PMAgent,pm4py")
    parser.add_argument("--max-attempts", type=int, default=3,
                        help="Maximum SQL generation attempts (default: 3)")
    parser.add_argument("--timeout", type=float, default=300,
                        help="PMAgent SQLite query timeout in seconds")
    args = parser.parse_args(argv)
    if args.max_attempts < 1:
        parser.error("--max-attempts must be at least 1")
    systems = [name.strip() for name in args.systems.split(",") if name.strip()]
    unknown = set(systems) - VALID_SYSTEMS
    if unknown or not systems:
        parser.error(f"Unknown/empty systems {sorted(unknown)}; choose from {sorted(VALID_SYSTEMS)}")

    try:
        cases, database, _ = load_selected(args)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    # Exclude Hugging Face model initialization from the measured query times.
    import agenti.request_agent  # noqa: F401
    import agenti.query_agent  # noqa: F401
    import agenti.response_agent  # noqa: F401
    import orchestratore.orchestratore  # noqa: F401

    pma = PMAgentAdapter(database, timeout=args.timeout, max_attempts=args.max_attempts) if "PMAgent" in systems else None
    pm4 = PM4PyLLMAdapter(database, max_attempts=args.max_attempts) if "pm4py" in systems else None
    rows = []
    try:
        for system in systems:
            for index, case in enumerate(cases, start=1):
                print(f"[{system} {index}/{len(cases)}] {case.category}: {case.query_id}", flush=True)
                result = _run(system, case, database, pma, pm4, args.max_attempts)
                category = "boolean" if case.category == "boolean" else case.category
                correct = compare(case.expected, result.get("result"), category) if result.get("status") == "success" else None
                rows.append({
                    "system": system,
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
    finally:
        if pma:
            pma.close()

    output = default_output(args, "response_times_all_systems.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, default=str)
                             if isinstance(value, (dict, list, tuple)) else value
                             for key, value in row.items()})
    print(f"Saved {len(rows)} system-query measurements to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
