"""Shared query selection and CLI helpers for orchestrator cost experiments."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from src.dataset.datasets import QueryCase, load_workbooks
from .evaluation import compare

ROOT = Path(__file__).resolve().parents[2]
CATEGORY_ORDER = ("boolean", "scalar", "set", "next_step_probability",
                  "next_step_case", "next_step_history", "next_step_what_if")
RESULTS_WORKBOOK = ROOT / "benchmark" / "results" / "results.xlsx"


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--database", type=Path, default=ROOT / "benchmark" / "databases" / "container_logistics.sqlite",
                        help="SQLite OCEL database")
    parser.add_argument("--output", type=Path, help="Output CSV path")


def _metadata_value(case: QueryCase, key: str):
    normalize = lambda value: "".join(ch for ch in str(value or "").casefold() if ch.isalnum())
    wanted = normalize(key)
    return next((value for name, value in case.metadata.items()
                 if normalize(name) == wanted and value is not None), None)


def select_representative_cases(cases: Iterable[QueryCase]) -> list[QueryCase]:
    """Choose one row per requested type, but only if pipeline output matches expected."""
    from .adapters import _nsp_specification, _relational_specification

    groups: dict[str, list[QueryCase]] = {key: [] for key in ("boolean", "scalar", "set", "next_step")}
    for case in cases:
        raw_type = str(_metadata_value(case, "Query Type") or "").strip().casefold()
        actual = _metadata_value(case, "Answer Pipeline")
        if raw_type == "bool":
            case.category = "boolean"
            group, comparison_category = "boolean", "boolean"
        elif raw_type in {"scalar", "set"}:
            group, comparison_category = raw_type, raw_type
        elif raw_type == "nextstep" and case.category.startswith("next_step_"):
            group, comparison_category = "next_step", case.category
        else:
            continue

        # Reuse the benchmark's semantic comparison (including numeric
        # precision, boolean parsing, set ordering, and activity normalization).
        if compare(case.expected, actual, comparison_category) is not True:
            continue
        if group == "next_step":
            # Select a verified NSP query that actually invokes model inference:
            # the structured trace SQL path would make FLOPpy report all zeros.
            if (_nsp_specification(case) is not None
                    or _relational_specification(case) is None):
                continue
        groups[group].append(case)

    selected: list[QueryCase] = []
    for group in ("set", "boolean", "scalar", "next_step"):
        candidates = groups[group]
        if not candidates:
            extra = " and invokes the orchestrator models" if group == "next_step" else ""
            raise ValueError(f"No correct {group} query{extra} found in {RESULTS_WORKBOOK.name}.")
        candidates.sort(key=lambda c: (
            CATEGORY_ORDER.index(c.category.casefold()) if c.category.casefold() in CATEGORY_ORDER else len(CATEGORY_ORDER),
            c.source_row,
        ))
        selected.append(candidates[0])
    return selected


def load_selected(args: argparse.Namespace) -> tuple[list[QueryCase], Path, Path]:
    database = args.database if args.database.is_absolute() else ROOT / args.database
    if not database.is_file():
        raise FileNotFoundError(f"Database does not exist: {database}")
    if not RESULTS_WORKBOOK.is_file():
        raise FileNotFoundError(f"Results workbook does not exist: {RESULTS_WORKBOOK}")
    cases = load_workbooks(RESULTS_WORKBOOK.parent, [RESULTS_WORKBOOK.name], ROOT / "benchmark" / "results")
    return select_representative_cases(cases), database.resolve(), RESULTS_WORKBOOK.parent


def default_output(args: argparse.Namespace, filename: str) -> Path:
    if args.output:
        return args.output if args.output.is_absolute() else ROOT / args.output
    return ROOT / "benchmark" / "results" / filename
