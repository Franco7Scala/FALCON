"""Result normalization and denominator-explicit metrics."""
from __future__ import annotations

from collections import defaultdict
from typing import Any
import ast
import json


def parse_cell(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        try:
            return json.loads(text)
        except (ValueError, TypeError):
            try:
                return ast.literal_eval(text)
            except (ValueError, SyntaxError):
                if "\n" in text:
                    parsed_rows = []
                    try:
                        for line in text.splitlines():
                            line = line.strip()
                            if line:
                                parsed_rows.append(ast.literal_eval(line))
                        if parsed_rows:
                            return parsed_rows
                    except (ValueError, SyntaxError):
                        pass
                if text.casefold() == "null":
                    return None
                if text.casefold() in {"true", "false"}:
                    return text.casefold() == "true"
                try:
                    return float(text) if "." in text else int(text)
                except ValueError:
                    return text
    return value


def _canonical(value: Any) -> Any:
    value = parse_cell(value)
    if isinstance(value, dict):
        return tuple(sorted((str(k), _canonical(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        vals = [_canonical(v) for v in value]
        # Lists of rows are compared as multisets, while order inside each row is retained.
        if vals and all(isinstance(v, tuple) for v in vals):
            return tuple(sorted(vals, key=repr))
        return tuple(vals)
    if isinstance(value, str):
        return value.strip()
    return value


def _as_set_rows(value: Any) -> tuple[Any, ...]:
    value = parse_cell(value)
    if value is None:
        return ()
    if isinstance(value, str) and value.strip().casefold() in {
        "nessun risultato", "[nessun risultato]", "no result", "[no result]",
    }:
        return ()
    if not isinstance(value, (list, tuple)):
        rows = [(value,)]
    elif not value:
        rows = []
    elif all(isinstance(row, dict) for row in value):
        rows = [tuple(row.values()) for row in value]
    elif all(isinstance(row, (list, tuple)) for row in value):
        rows = [tuple(row) for row in value]
    elif isinstance(value, tuple):
        # A literal tuple in the workbook represents one SQL row, not a set
        # of unrelated one-cell rows.
        rows = [tuple(value)]
    else:
        rows = [(item,) for item in value]
    canonical_rows = [tuple(_canonical(cell) for cell in row) for row in rows]
    return tuple(sorted(canonical_rows, key=repr))


def _boolean_value(value: Any) -> bool | None:
    """Read a single boolean value, including the common one-cell SQL shapes."""
    value = parse_cell(value)
    if isinstance(value, dict):
        if len(value) != 1:
            return None
        value = next(iter(value.values()))
    if isinstance(value, (list, tuple)):
        # A SQL result must contain exactly one row with exactly one column.
        if len(value) != 1:
            return None
        value = value[0]
        if isinstance(value, dict):
            if len(value) != 1:
                return None
            value = next(iter(value.values()))
        elif isinstance(value, (list, tuple)):
            if len(value) != 1:
                return None
            value = value[0]
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
        return None
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"si", "sì", "yes", "true", "1"}:
            return True
        if normalized in {"no", "false", "0"}:
            return False
    return None


def compare(expected: Any, actual: Any, category: str = "") -> bool | None:
    if expected is None or (isinstance(expected, str) and not expected.strip()):
        return None
    if category.strip().casefold() == "set":
        return _as_set_rows(expected) == _as_set_rows(actual)
    if category.strip().casefold() == "boolean":
        expected_boolean = _boolean_value(expected)
        if expected_boolean is not None:
            # A malformed or non-boolean SQL result is incorrect, not an
            # unevaluable match against a yes/no expectation.
            return _boolean_value(actual) == expected_boolean
    actual = parse_cell(actual)
    if isinstance(actual, list) and len(actual) == 1:
        only = actual[0]
        if isinstance(only, dict) and len(only) == 1:
            actual = next(iter(only.values()))
        elif isinstance(only, (list, tuple)) and len(only) == 1:
            actual = only[0]
        elif not isinstance(only, (dict, list, tuple)):
            actual = only
    expected, actual = _canonical(expected), _canonical(actual)
    # Excel scalar expectations often correspond to one-cell SQL result sets.
    if isinstance(actual, tuple) and len(actual) == 1 and isinstance(actual[0], tuple) and len(actual[0]) == 1:
        actual = actual[0][0]
    if isinstance(actual, tuple) and len(actual) == 1 and not isinstance(actual[0], tuple):
        actual = actual[0]
    return expected == actual


def score(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    successes = [r for r in records if r["status"] == "success"]
    assessed = [r for r in records if r["correct"] is not None]
    applicable = [r for r in records if r["status"] != "out_of_scope"]
    supported = [r for r in applicable if r["status"] != "unsupported"]
    return {
        "queries_total": total,
        "accuracy": {"value": sum(r["correct"] is True for r in assessed) / len(assessed) if assessed else None,
                     "numerator": sum(r["correct"] is True for r in assessed), "denominator": len(assessed)},
        "error_rate": {"value": sum(r["correct"] is False for r in assessed) / len(assessed) if assessed else None,
                       "numerator": sum(r["correct"] is False for r in assessed), "denominator": len(assessed)},
        "execution_success_rate": {"value": len(successes) / len(supported) if supported else None,
                                   "numerator": len(successes), "denominator": len(supported)},
        "coverage": {"value": len(supported) / total if total else None,
                     "numerator": len(supported), "denominator": total},
        "applicability": {"value": len(applicable) / total if total else None,
                           "numerator": len(applicable), "denominator": total},
        "mean_seconds": {"value": sum(r["seconds"] for r in successes if r["seconds"] is not None) /
                                  sum(r["seconds"] is not None for r in successes) if any(r["seconds"] is not None for r in successes) else None,
                         "denominator": sum(r["seconds"] is not None for r in successes)},
        "unsupported": sum(r["status"] == "unsupported" for r in records),
        "out_of_scope": sum(r["status"] == "out_of_scope" for r in records),
        "out_of_context": sum(r["status"] == "out_of_context" for r in records),
        "pipeline_rejected_after_execution": sum(r.get("agent_decision") not in (None, "SODDISFATTA") and r["status"] == "success" for r in records),
        "not_evaluable": sum(r["correct"] is None and r["status"] == "success" for r in records),
    }


def summaries(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped = defaultdict(list)
    for row in records:
        grouped[(row["system"], row["category"])].append(row)
    return {f"{system}/{category}": score(rows) for (system, category), rows in sorted(grouped.items())}
