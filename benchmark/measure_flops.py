"""Track model compute for four verified queries across the three systems.

Run with: python -m benchmark.measure_flops
Requires ``pip install floppy-tracker``. Models use the normal 4-bit loading.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time

from .adapters import PMAgentAdapter, PM4PyLLMAdapter, run_orchestrator
from .cost_common import add_common_arguments, default_output, load_selected
from .evaluation import compare

MODEL_SPECS = {
    "coder": ("Qwen/Qwen3-Coder-30B-A3B-Instruct", "agenti.query_agent"),
    "judge": ("Qwen/Qwen2.5-14B-Instruct", "agenti.response_agent"),
}
VALID_SYSTEMS = {"orchestrator", "PMAgent", "pm4py"}


def _get_hf_parts(agent_module):
    """Unwrap the LangChain ChatHuggingFace adapter to its HF pipeline."""
    hf_wrapper = getattr(agent_module.LLM, "llm", None)
    hf_pipeline = getattr(hf_wrapper, "pipeline", None)
    model = getattr(hf_pipeline, "model", None)
    tokenizer = getattr(hf_pipeline, "tokenizer", None)
    if model is None:
        raise RuntimeError(f"Could not access the Transformers model from {agent_module.__name__}.LLM")
    return model, tokenizer


def _number(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return int(number) if number.is_integer() else number


def _report_metrics(report) -> dict:
    return {
        "flops": _number(getattr(report, "overall_flop", None)),
        "bops": _number(getattr(report, "overall_bop", None)),
        "arithmetic_intensity": _number(getattr(report, "arithmetic_intensity", None)),
        "model_forward_flops": _number(getattr(report, "model_forward_flop", None)),
        "model_forward_bops": _number(getattr(report, "model_forward_bop", None)),
        "forward_memory_bytes": _number(getattr(report, "forward_memory_bytes", None)),
        "backend": getattr(report, "backend", None),
    }


def _run(system, case, database, pma, pm4, max_attempts):
    if system == "orchestrator":
        return run_orchestrator(case, database, max_attempts)
    if system == "PMAgent":
        return pma.run(case)
    return pm4.run(case)


def _track_query(system, case, database, pma, pm4, max_attempts, models, tracker_class):
    trackers = []
    started = time.perf_counter()
    try:
        for model_name, model, tokenizer in models:
            tracker = tracker_class(run_name=f"{system}-{case.category}-{model_name.rsplit('/', 1)[-1]}")
            tracker.start(model=model, tokenizer=tokenizer)
            trackers.append((model_name, tracker))
        result = _run(system, case, database, pma, pm4, max_attempts)
    finally:
        for _, tracker in reversed(trackers):
            tracker.stop()

    elapsed = time.perf_counter() - started
    model_metrics = {name: _report_metrics(tracker.report()) for name, tracker in trackers}
    return result, elapsed, model_metrics


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
    try:
        from floppy import FLOPpyTracker
    except ImportError as exc:
        parser.error(f"FLOPpy is unavailable; install it with 'pip install floppy-tracker': {exc}")

    # The benchmark adapters use the coder for SQL generation. The orchestrator
    # also uses the 14B model to judge relational SQL; competitors do not.
    import importlib
    loaded = {}
    for key in ("coder", "judge"):
        model_name, module_name = MODEL_SPECS[key]
        module = importlib.import_module(module_name)
        model, tokenizer = _get_hf_parts(module)
        loaded[key] = (model_name, model, tokenizer)

    pma = PMAgentAdapter(database, timeout=args.timeout, max_attempts=args.max_attempts) if "PMAgent" in systems else None
    pm4 = PM4PyLLMAdapter(database, max_attempts=args.max_attempts) if "pm4py" in systems else None
    records = []
    try:
        for system in systems:
            # These four selected cases reach query_agent in every system.
            # Only the orchestrator additionally invokes the 14B response judge.
            models = [loaded["coder"]]
            if system == "orchestrator":
                models.append(loaded["judge"])
            for index, case in enumerate(cases, start=1):
                print(f"[{system} {index}/{len(cases)}] FLOPpy {case.category}: {case.query_id}", flush=True)
                result, elapsed, costs = _track_query(
                    system, case, database, pma, pm4, args.max_attempts, models, FLOPpyTracker,
                )
                category = "boolean" if case.category == "boolean" else case.category
                correct = compare(case.expected, result.get("result"), category) if result.get("status") == "success" else None
                row = {
                    "system": system,
                    "category": case.category,
                    "query_id": case.query_id,
                    "source_file": case.source_file,
                    "source_row": case.source_row,
                    "expected_result": case.expected,
                    "previous_pipeline_result": case.metadata.get("Answer Pipeline"),
                    "previously_correct": True,
                    "result": result.get("result"),
                    "correct": correct,
                    "status": result.get("status"),
                    "response_seconds": result.get("seconds"),
                    "tracking_seconds": elapsed,
                    "attempts": result.get("attempts"),
                    "note": result.get("note"),
                    "error": result.get("error"),
                    "model_costs": costs,
                }
                records.append(row)
                print(json.dumps({"system": system, "category": case.category, "correct": correct,
                                  "status": row["status"], "model_costs": costs}, ensure_ascii=False), flush=True)
    finally:
        if pma:
            pma.close()

    output = default_output(args, "model_computational_cost_all_systems.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["system", "category", "query_id", "source_file", "source_row", "expected_result",
               "previous_pipeline_result", "previously_correct", "result", "correct", "status",
               "response_seconds", "tracking_seconds", "attempts", "note", "error", "model_costs"]
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in records:
            serialized = dict(row)
            for key in ("expected_result", "previous_pipeline_result", "result", "model_costs"):
                if isinstance(serialized[key], (dict, list, tuple)):
                    serialized[key] = json.dumps(serialized[key], ensure_ascii=False, default=str)
            writer.writerow(serialized)
    json_output = output.with_suffix(".json")
    json_output.write_text(json.dumps(records, ensure_ascii=False, indent=2, allow_nan=False, default=str), encoding="utf-8")
    print(f"Saved {len(records)} system-query cost measurements to {output} and {json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
