"""Single entry point for comparative OCEL query experiments."""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

from .datasets import load_workbooks
from .adapters import PMAgentAdapter, PM4PyLLMAdapter, run_orchestrator
from .evaluation import compare, summaries
from .ocel import schema

ROOT = Path(__file__).resolve().parents[1]


def _json_default(value):
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run selected comparative experiments over OCEL query workbooks.")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "dataset")
    parser.add_argument("--database", type=Path, default=ROOT / "databases" / "container_logistics.sqlite")
    parser.add_argument("--files", help="Comma-separated .xlsx filenames; default: all dataset/*.xlsx")
    parser.add_argument("--systems", default="orchestrator,PMAgent,pm4py", help="Comma-separated: orchestrator,PMAgent,pm4py")
    parser.add_argument("--max-queries", type=int, help="Limit the first N loaded cases (smoke subset)")
    parser.add_argument("--query-id", action="append", help="Run only this stable source-file:sheet:row id; repeatable")
    parser.add_argument("--timeout", type=float, default=300, help="PMAgent SQLite query timeout in seconds")
    parser.add_argument("--max-attempts", type=int, default=3, help="Maximum query generation/execution attempts for SQL adapters (default: 3)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "risultati" / "benchmark")
    args = parser.parse_args(argv)

    dataset_dir = args.dataset_dir if args.dataset_dir.is_absolute() else ROOT / args.dataset_dir
    database = args.database if args.database.is_absolute() else ROOT / args.database
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if not database.is_file():
        parser.error(f"OCEL database not found: {database}")
    files = [x.strip() for x in args.files.split(",") if x.strip()] if args.files else None
    cases = load_workbooks(dataset_dir, files, ROOT / "risultati")
    if args.query_id:
        wanted = set(args.query_id)
        cases = [c for c in cases if c.query_id in wanted]
    if args.max_queries:
        cases = cases[:args.max_queries]
    if not cases:
        parser.error("No query rows found after filtering")
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    valid = {"orchestrator", "PMAgent", "pm4py"}
    if set(systems) - valid:
        parser.error(f"Unknown systems {set(systems)-valid}; choose from {sorted(valid)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=output_dir / "diagnostics.log", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", encoding="utf-8")
    logging.info("Loaded %d query rows from %s", len(cases), dataset_dir)
    # Save the real source schema used by the experiment for reproducibility.
    (output_dir / "ocel_schema.json").write_text(json.dumps(schema(database), indent=2), encoding="utf-8")
    records = []
    if args.max_attempts < 1:
        parser.error("--max-attempts must be at least 1")
    pma = PMAgentAdapter(database, args.timeout, args.max_attempts) if "PMAgent" in systems else None
    pm4 = PM4PyLLMAdapter(database, args.max_attempts) if "pm4py" in systems else None
    try:
        for system in systems:
            for i, case in enumerate(cases, 1):
                logging.info("START %s %s (%d/%d)", system, case.query_id, i, len(cases))
                print(f"[{system} {i}/{len(cases)}] {case.query_id}", flush=True)
                if system == "orchestrator":
                    row = run_orchestrator(case, database, args.max_attempts)
                elif system == "PMAgent":
                    row = pma.run(case)
                else:
                    row = pm4.run(case)
                if row["status"] == "success":
                    row["correct"] = compare(row["expected"], row["result"], row["category"])
                    if row["correct"] is None:
                        row["note"] = (row["note"] + " " if row["note"] else "") + "Expected result is empty; correctness not evaluated."
                records.append(row)
                logging.info("END %s %s status=%s seconds=%s", system, case.query_id, row["status"], row["seconds"])
                print(f"  status={row['status']} seconds={row['seconds']}", flush=True)
                if row["error"]:
                    logging.error("%s %s %s", system, case.query_id, row["error"])
    finally:
        if pma:
            pma.close()

    metrics = summaries(records)
    (output_dir / "results.json").write_text(json.dumps(records, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False, default=_json_default) if isinstance(v, (dict, list)) else v for k, v in row.items()} for row in records)
    lines = ["# Comparative OCEL experiment report", "", f"Query rows loaded: {len(cases)}", "",
             "Metrics are separated by system and dataset category. Accuracy and error rate use only completed executions with a known expected value. Applicability is the share of rows within a system's declared scope; execution rate excludes unsupported and out-of-scope rows; coverage is supported rows divided by all loaded rows. Mean response time uses successful rows with a measured duration.", ""]
    for label, values in metrics.items():
        lines += [f"## {label}", "", f"- Accuracy: {values['accuracy']['value']} (n={values['accuracy']['numerator']}, denominator={values['accuracy']['denominator']})",
                  f"- Error rate: {values['error_rate']['value']} (n={values['error_rate']['numerator']}, denominator={values['error_rate']['denominator']})",
                  f"- Execution success rate: {values['execution_success_rate']['value']} (n={values['execution_success_rate']['numerator']}, denominator={values['execution_success_rate']['denominator']})",
                  f"- Applicability: {values['applicability']['value']} (n={values['applicability']['numerator']}, denominator={values['applicability']['denominator']})",
                  f"- Coverage: {values['coverage']['value']} (n={values['coverage']['numerator']}, denominator={values['coverage']['denominator']})",
                  f"- Mean response time: {values['mean_seconds']['value']} seconds (n={values['mean_seconds']['denominator']})",
                  f"- Out of scope: {values['out_of_scope']}; unsupported within scope: {values['unsupported']}; out of context: {values['out_of_context']}; pipeline rejected after execution: {values['pipeline_rejected_after_execution']}; successful but not evaluable: {values['not_evaluable']}", ""]
    lines += ["## Per-query comparison", "", "| Query ID | Category | System | Status | Correct | Attempts | Expected source | Seconds |", "|---|---|---|---|---:|---:|---|---:|"]
    for row in records:
        lines.append(f"| {row['query_id']} | {row['category']} | {row['system']} | {row['status']} | {row['correct']} | {row['attempts']} | {row['expected_source']} | {row['seconds']} |")
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(records)} system results to {output_dir}")


if __name__ == "__main__":
    main()
