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
    "category": {"type query", "nsp", "category", "tipo query"},
    "object_type": {"object type"},
    "case_id": {"case id / target", "case id", "target"},
    "trace": {"traccia / prefisso (case-based)", "trace / prefix", "trace prefix"},
    "sql": {"query", "sql", "query sql"},
}


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().casefold())


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
                    category = str(get("category") or ("next_step" if get("object_type") else "generic"))
                    expected = get("expected")
                    expected_source = f"{path.name}/{sheet.title}!{row_number}"
                    if (path.stem.casefold() == "queries_200_cleaned"
                            and category.strip().casefold() == "set" and results_dir is not None):
                        reference = _set_reference(row_number, results_dir)
                        if reference is not None:
                            expected, expected_source = reference
                    if get("object_type") is not None and "object_type" in found:
                        category = "next_step"
                    elif category.strip().casefold() == "nsp":
                        category = "next_step"
                    if category == "next_step":
                        target_id = str(get("case_id") or "").strip()
                        trace_text = str(get("trace") or "")
                        explicit_case = re.search(
                            r"\bCase\s+['\"]([^'\"]+)['\"]\s*\(\s*Object\s+Type\s*:",
                            trace_text, re.IGNORECASE,
                        )
                        if explicit_case and target_id.casefold() == explicit_case.group(1).casefold():
                            category = "next_step_case"
                        else:
                            category = "next_step_history"
                    query_id = f"{path.stem}:{sheet.title}:{row_number}"
                    cases.append(QueryCase(query_id, path.name, sheet.title, row_number, category,
                                           str(question).strip(), expected, metadata, expected_source))
        finally:
            book.close()
    return cases
