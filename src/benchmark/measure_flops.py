"""Track model compute for verified queries on the FALCON orchestrator."""
from __future__ import annotations

import argparse
import csv
import json
import math
import time

from .adapters import run_orchestrator
from .cost_common import add_common_arguments, default_output, load_selected
from .evaluation import compare

MODEL_SPECS = {
    "coder": ("Qwen/Qwen3-Coder-30B-A3B-Instruct", "src.agents.query_agent"),
    "judge": ("Qwen/Qwen2.5-14B-Instruct", "src.agents.response_agent"),
}


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


def _track_query(case, database, max_attempts, models, tracker_class):
    trackers = []
    started = time.perf_counter()
    try:
        for model_name, model, tokenizer in models:
            tracker = tracker_class(run_name=f"falcon-{case.category}-{model_name.rsplit('/', 1)[-1]}")
            tracker.start(model=model, tokenizer=tokenizer)
            trackers.append((model_name, tracker))
        result = run_orchestrator(case, database, max_attempts)
    finally:
        for _, tracker in reversed(trackers):
            tracker.stop()

    elapsed = time.perf_counter() - started
    model_metrics = {name: _report_metrics(tracker.report()) for name, tracker in trackers}
    return result, elapsed, model_metrics


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
    try:
        from floppy import FLOPpyTracker
    except ImportError as exc:
        parser.error(f"FLOPpy is unavailable; install it with 'pip install floppy-tracker': {exc}")

    import importlib
    loaded = {}
    for key in ("coder", "judge"):
        model_name, module_name = MODEL_SPECS[key]
        module = importlib.import_module(module_name)
        model, tokenizer = _get_hf_parts(module)
        loaded[key] = (model_name, model, tokenizer)

    models = [loaded["coder"], loaded["judge"]]
    records = []
    for index, case in enumerate(cases, start=1):
        print(f"[FALCON {index}/{len(cases)}] FLOPpy {case.category}: {case.query_id}", flush=True)
        result, elapsed, costs = _track_query(
            case, database, args.max_attempts, models, FLOPpyTracker,
        )
        category = "boolean" if case.category == "boolean" else case.category
        correct = compare(case.expected, result.get("result"), category) if result.get("status") == "success" else None
        row = {
            "system": "orchestrator",
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
        print(json.dumps({"system": "orchestrator", "category": case.category, "correct": correct,
                          "status": row["status"], "model_costs": costs}, ensure_ascii=False), flush=True)

    output = default_output(args, "flops_falcon.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(records[0])
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, default=str)
                             if isinstance(value, (dict, list, tuple)) else value
                             for key, value in row.items()})
    print(f"Saved {len(records)} FLOPpy measurements to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
