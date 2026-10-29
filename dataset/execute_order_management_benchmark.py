"""Esegue le query di order_management_benchmark.xlsx e popola Expected result.

Uso dalla cartella principale del progetto:
    python dataset/execute_order_management_benchmark.py

Per usare altri percorsi:
    python dataset/execute_order_management_benchmark.py --db path/database.sqlite --xlsx path/file.xlsx

Per aggiornare il file di input direttamente:
    python dataset/execute_order_management_benchmark.py --in-place

Il database viene aperto in sola lettura. Le viste NSP definite nel foglio
"Viste" vengono create come viste TEMPORARY nella connessione, senza cambiare
il database.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

from openpyxl import load_workbook


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE = PROJECT_ROOT / "databases" / "order_management.sqlite"
DEFAULT_WORKBOOK = Path(__file__).resolve().parent / "order_management_benchmark.xlsx"
MAX_CELL_LENGTH = 32_000


def _normalize(value: Any) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def _column_index(headers: list[Any], name: str, sheet_name: str) -> int | None:
    matches = [i for i, value in enumerate(headers, start=1) if _normalize(value) == _normalize(name)]
    if len(matches) > 1:
        raise ValueError(f"Nel foglio {sheet_name!r} la colonna {name!r} è duplicata.")
    return matches[0] if matches else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.hex()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _expected_value(query_type: str, rows: list[tuple[Any, ...]]) -> Any:
    safe_rows = [tuple(_json_safe(value) for value in row) for row in rows]
    if query_type in {"set", "nsp-prob", "prob", "probability"}:
        return repr(safe_rows)
    if len(safe_rows) == 1 and len(safe_rows[0]) == 1:
        value = safe_rows[0][0]
        return "NULL" if value is None else value
    return repr(safe_rows)


def _readonly_authorizer(action: int, _arg1: str | None, _arg2: str | None,
                         _database: str | None, _trigger: str | None) -> int:
    allowed = {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        getattr(sqlite3, "SQLITE_RECURSIVE", -1),
    }
    return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY


def _open_read_only(database: Path) -> sqlite3.Connection:
    if not database.is_file():
        raise FileNotFoundError(f"Database SQLite non trovato: {database}")
    uri = f"file:{database.resolve().as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def _create_temporary_views(connection: sqlite3.Connection, workbook) -> int:
    if "Viste" not in workbook.sheetnames:
        return 0
    sheet = workbook["Viste"]
    views_created = 0
    for row in sheet.iter_rows(min_row=2, values_only=True):
        if len(row) < 2 or not row[1] or not str(row[1]).strip():
            continue
        definition = str(row[1]).strip().rstrip(";")
        if not re.match(r"CREATE\s+(?:TEMP(?:ORARY)?\s+)?VIEW\b", definition, re.IGNORECASE):
            continue
        # A TEMP VIEW is stored in the connection's temporary schema, so this
        # also works while the source SQLite database is opened read-only.
        definition = re.sub(
            r"^CREATE\s+(?:TEMP(?:ORARY)?\s+)?VIEW\b",
            "CREATE TEMP VIEW",
            definition,
            count=1,
            flags=re.IGNORECASE,
        )
        connection.execute(definition)
        views_created += 1
    return views_created


def execute_workbook(database: Path, workbook_path: Path, output_path: Path,
                     timeout_seconds: int = 60) -> tuple[int, int, dict[str, int], Path | None]:
    if not workbook_path.is_file():
        raise FileNotFoundError(f"Workbook non trovato: {workbook_path}")

    workbook = load_workbook(workbook_path)
    connection = _open_read_only(database)
    total_ok = total_errors = 0
    sheet_counts: dict[str, int] = {}
    extended_results: dict[str, Any] = {}

    try:
        views_created = _create_temporary_views(connection, workbook)
        connection.set_authorizer(_readonly_authorizer)
        print(f"Viste temporanee create: {views_created}")

        for sheet in workbook.worksheets:
            headers = [cell.value for cell in sheet[1]]
            query_col = _column_index(headers, "Query", sheet.title)
            expected_col = _column_index(headers, "Expected result", sheet.title)
            if query_col is None or expected_col is None:
                continue
            type_col = _column_index(headers, "Query type", sheet.title)
            if type_col is None:
                type_col = _column_index(headers, "NSP", sheet.title)

            sheet_ok = sheet_errors = 0
            for row_number in range(2, sheet.max_row + 1):
                query = sheet.cell(row_number, query_col).value
                if query is None or not str(query).strip():
                    continue
                query_type = _normalize(sheet.cell(row_number, type_col).value) if type_col else ""

                try:
                    deadline = time.monotonic() + timeout_seconds
                    connection.set_progress_handler(
                        lambda: 1 if time.monotonic() > deadline else 0,
                        10_000,
                    )
                    cursor = connection.execute(str(query))
                    rows = cursor.fetchall()
                    columns = [column[0] for column in cursor.description or []]
                    connection.set_progress_handler(None, 0)

                    value = _expected_value(query_type, rows)
                    if isinstance(value, str) and len(value) > MAX_CELL_LENGTH:
                        key = f"{sheet.title}!{row_number}"
                        extended_results[key] = {
                            "query_type": query_type,
                            "query": str(query),
                            "columns": columns,
                            "rows": [[_json_safe(item) for item in row] for row in rows],
                        }
                        value = (f"RISULTATO DI {len(rows)} RIGHE TROPPO LUNGO PER EXCEL; "
                                 "vedi il file JSON dei risultati estesi")

                    sheet.cell(row_number, expected_col).value = value
                    sheet_ok += 1
                    total_ok += 1
                except sqlite3.Error as exc:
                    connection.set_progress_handler(None, 0)
                    sheet.cell(row_number, expected_col).value = f"ERRORE SQL: {exc}"
                    sheet_errors += 1
                    total_errors += 1

            sheet_counts[sheet.title] = sheet_ok
            if sheet_ok or sheet_errors:
                print(f"{sheet.title}: {sheet_ok} query riuscite, {sheet_errors} errori")
    finally:
        connection.close()

    if total_ok + total_errors == 0:
        workbook.close()
        raise ValueError("Non ho trovato fogli con colonne Query ed Expected result.")

    try:
        workbook.save(output_path)
    finally:
        workbook.close()

    json_path = None
    if extended_results:
        json_path = output_path.with_name(f"{output_path.stem}_risultati_estesi.json")
        json_path.write_text(json.dumps(extended_results, ensure_ascii=False, indent=2), encoding="utf-8")
    return total_ok, total_errors, sheet_counts, json_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DATABASE,
                        help=f"Database SQLite (default: {DEFAULT_DATABASE})")
    parser.add_argument("--xlsx", type=Path, default=DEFAULT_WORKBOOK,
                        help=f"Workbook di input (default: {DEFAULT_WORKBOOK})")
    parser.add_argument("--out", type=Path, default=None,
                        help="Workbook di output (default: <nome>_with_results.xlsx)")
    parser.add_argument("--in-place", action="store_true",
                        help="Aggiorna il workbook originale invece di crearne una copia")
    parser.add_argument("--timeout", type=int, default=60,
                        help="Timeout massimo per query in secondi (default: 60)")
    args = parser.parse_args()

    if args.timeout <= 0:
        parser.error("--timeout deve essere maggiore di zero")
    if args.in_place and args.out is not None:
        parser.error("Usa --in-place oppure --out, non entrambi")

    output_path = args.xlsx if args.in_place else (
        args.out or args.xlsx.with_name(f"{args.xlsx.stem}_with_results.xlsx")
    )
    ok, errors, sheet_counts, json_path = execute_workbook(
        args.db, args.xlsx, output_path, args.timeout
    )
    print(f"Totale: {ok} query riuscite, {errors} query con errore")
    print(f"Workbook salvato in: {output_path.resolve()}")
    if json_path:
        print(f"Risultati estesi salvati in: {json_path.resolve()}")


if __name__ == "__main__":
    main()
