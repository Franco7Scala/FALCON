"""Read and validate query rows from the repository's Excel workbooks."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re
import json
import csv
import openpyxl


@dataclass
class QueryCase:
    query_id: str
    source_file: str
    sheet: str
    source_row: int
    category: str
    question: str
    expected: Any
    metadata: dict[str, Any]
    expected_source: str | None = None


ALIASES = {
    "question": {"query in natural language", "domanda", "question", "query in linguaggio naturale"},
    "expected": {"expected result", "risultato atteso"},
    "category": {"type query", "query type", "nsp", "category", "tipo query"},
    "object_type": {"object type"},
    "case_id": {"case id / target", "case id", "target"},
    "trace": {"traccia / prefisso (case-based)", "trace / prefix", "trace prefix"},
    "sql": {"query", "sql", "query sql"},
}


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().casefold())


def _infer_category(raw_category: Any, object_type: Any, target: Any,
                    trace: Any, question: str, sql: Any = None) -> str:
    """Infer benchmark task type from explicit labels and NSP workbook metadata."""
    raw = _norm(raw_category)
    if raw in {"boolean", "scalar", "set"}:
        return raw
    if raw in {"nsp-prob", "nsp_prob", "prob", "probability"}:
        return "next_step_probability"

    target_text = str(target or "").strip()
    target_norm = _norm(target_text)
    question_norm = _norm(question)
    sql_text = str(sql or "")

    # Some generated workbooks put several NSP subtypes under `nextstep`.
    # Keep activity prediction separate; only duration questions remain scalar.
    if raw in {"nextstep", "next-step", "next step"}:
        if ("what-if nsp" in target_norm or "what-if nsp" in question_norm
                or "[what-if nsp]" in question_norm):
            return "next_step_what_if"
        # Distinguish probability questions about the *immediate next activity*
        # from lifecycle probabilities that merely mention event names.
        if (re.search(
                r"probability distribution.{0,120}(?:next|following|immediately following).{0,80}activity",
                question_norm,
            )
                or re.search(r"probability.{0,80}next event.{0,120}\bversus\b", question_norm)):
            return "next_step_probability"
        if (re.search(r"\b(?:hours?|minutes?|days?)\b", question_norm)
                and re.search(r"julianday|timestamp|next_ts", sql_text, re.IGNORECASE)):
            return "scalar"
        if re.search(r"\b(?:object_id|ocel_object_id)\s*=\s*['\"]", sql_text, re.IGNORECASE):
            return "next_step_case"
        return "next_step_history"

    if object_type is None and raw not in {"nsp", "next_step"}:
        return raw or "generic"

    trace_text = str(trace or "")

    # NSP probabilities and transition matrices have their own benchmark
    # category even though their answer contains a set of rows.
    if ("global top-1" in target_norm or "matrix" in target_norm
            or "prob" in target_norm or "probabilit" in question_norm):
        return "next_step_probability"
    if "what-if nsp" in target_norm or "what-if nsp" in question_norm:
        return "next_step_what_if"
    if target_norm.startswith("multi-obj"):
        return "scalar"

    explicit_case = re.search(
        r"\bCase\s+['\"]([^'\"]+)['\"]\s*\(\s*Object\s+Type\s*:",
        trace_text, re.IGNORECASE,
    )
    if explicit_case and target_text.casefold() == explicit_case.group(1).casefold():
        return "next_step_case"
    return "next_step_history"


def _set_reference(source_row: int, results_dir: Path) -> tuple[Any, str] | None:
    """Load a set reference from the full JSON first, then its per-query CSV."""
    index = source_row - 1  # workbook header is row 1; q001 corresponds to data row 2
    complete = results_dir / "risultati_completi.json"
    if complete.is_file():
        try:
            payload = json.loads(complete.read_text(encoding="utf-8"))
            entry = payload.get(str(index))
            if isinstance(entry, dict) and entry.get("tipo") == "set" and "righe" in entry:
                return entry["righe"], str(complete)
        except (OSError, ValueError, TypeError):
            pass
    csv_path = results_dir / f"q{index:03d}_set.csv"
    if csv_path.is_file():
        with csv_path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.reader(stream))
        return (rows[1:] if rows else []), str(csv_path)
    return None


def load_workbooks(dataset_dir: Path, files: list[str] | None = None,
                   results_dir: Path | None = None) -> list[QueryCase]:
    """Load all xlsx files or an explicit list; ambiguous/missing question headers fail closed."""
    paths = [dataset_dir / f for f in files] if files else sorted(dataset_dir.glob("*.xlsx"))
    cases: list[QueryCase] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        book = openpyxl.load_workbook(path, read_only=True, data_only=True)
        try:
            for sheet in book.worksheets:
                rows = sheet.iter_rows(values_only=True)
                header = next(rows, None)
                if not header or all(v is None or not str(v).strip() for v in header):
                    continue
                names = [_norm(v) for v in header]
                found: dict[str, int] = {}
                for key, aliases in ALIASES.items():
                    hits = [i for i, name in enumerate(names) if name in aliases]
                    if len(hits) > 1:
                        raise ValueError(f"{path.name}/{sheet.title}: header {key!r} is ambiguous at columns {hits}")
                    if hits:
                        found[key] = hits[0]
                # Sheets used only to provide SQL view definitions are not query sheets.
                if "question" not in found:
                    if "expected" in found and "sql" in found:
                        raise ValueError(f"{path.name}/{sheet.title}: query-like sheet is missing a natural-language question header")
                    continue
                if "expected" not in found:
                    raise ValueError(f"{path.name}/{sheet.title}: missing 'Expected result' header")
                for row_number, row in enumerate(rows, start=2):
                    if not row or all(v is None or (isinstance(v, str) and not v.strip()) for v in row):
                        continue
                    get = lambda key: row[found[key]] if key in found and found[key] < len(row) else None
                    question = get("question")
                    if question is None or not str(question).strip():
                        continue
                    metadata = {str(header[i]): row[i] for i in range(min(len(header), len(row))) if header[i] is not None}
                    raw_category = get("category")
                    category = _infer_category(
                        raw_category, get("object_type"), get("case_id"),
                        get("trace"), str(question).strip(), get("sql"),
                    )
                    expected = get("expected")
                    expected_source = f"{path.name}/{sheet.title}!{row_number}"
                    if (path.stem.casefold() == "queries_200_cleaned"
                            and category.strip().casefold() in {"set", "next_step_probability"}
                            and results_dir is not None):
                        reference = _set_reference(row_number, results_dir)
                        if reference is not None:
                            expected, expected_source = reference
                    query_id = f"{path.stem}:{sheet.title}:{row_number}"
                    cases.append(QueryCase(query_id, path.name, sheet.title, row_number, category,
                                           str(question).strip(), expected, metadata, expected_source))
        finally:
            book.close()
    return cases
