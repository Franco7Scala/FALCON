"""Read-only generic SQL extension used by PMAgent and PM4Py adapters."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import re
import sqlite3
import time

from .ocel import connect_readonly


def _schema_text(database: Path) -> str:
    with connect_readonly(database) as con:
        rows = con.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
    return "\n\n".join(sql for _, sql in rows if sql)


def _clean_sql(output: Any) -> str:
    text = getattr(output, "content", output)
    if isinstance(text, list):
        text = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in text)
    text = str(text).strip().lstrip("\ufeff")
    fence = re.search(r"```(?:sql)?\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    if not re.match(r"^(SELECT|WITH)\b", text, re.IGNORECASE):
        match = re.search(r"\b(SELECT|WITH)\b", text, re.IGNORECASE)
        if not match:
            raise ValueError("The model did not return a SELECT/WITH query")
        text = text[match.start():].strip()
    return text


def _authorize_read_only(action, arg1, arg2, database_name, trigger_name):
    allowed = {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        getattr(sqlite3, "SQLITE_RECURSIVE", -1),
    }
    return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY


def _execute_read_only(database: Path, sql: str, timeout_seconds: float = 60) -> tuple[list[tuple[Any, ...]], list[str]]:
    con = sqlite3.connect(f"file:{database.resolve().as_posix()}?mode=ro", uri=True, timeout=30)
    try:
        con.set_authorizer(_authorize_read_only)
        deadline = time.monotonic() + timeout_seconds
        con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10000)
        cursor = con.execute(sql)
        names = [column[0] for column in cursor.description or []]
        return cursor.fetchall(), names
    finally:
        con.close()


def run_generic_sql(case, database: Path, system: str, max_attempts: int = 3,
                   timeout_seconds: float = 60) -> dict[str, Any]:
    """Generate, execute, and return one generic scalar/set/boolean SQL answer."""
    started = time.perf_counter()
    out = {
        "query_id": case.query_id, "system": system, "category": case.category,
        "question": case.question, "object_type": None, "expected": case.expected,
        "result": None, "sql": None, "seconds": None, "status": "execution_error",
        "correct": None, "error": None,
        "note": f"Generic relational SQL extension; results executed read-only against {database.name}.",
        "expected_source": case.expected_source, "pipeline_status": None,
        "agent_decision": None, "agent_error": None, "attempts": 0,
        "source_file": case.source_file, "sheet": case.sheet, "source_row": case.source_row,
    }
    try:
        from agenti.query_agent import LLM

        schema = _schema_text(database)
        task_guidance = {
            "scalar": "Return one row and one value answering the numeric/text scalar question.",
            "boolean": "Return exactly one row and one column with 'SI' when the complete proposition is true, otherwise 'NO'. Use CASE WHEN/EXISTS or compare scalar subquery counts; never return a raw count, row list, or multiple columns.",
            "set": "Return all requested rows and columns. Preserve requested filters and ordering; do not aggregate unless the question asks for it.",
        }.get(case.category, "Answer the request using the relevant rows and columns.")
        base_prompt = f'''You are a SQLite text-to-SQL system for an OCEL logistics database.
Generate exactly one read-only SQL query using only the schema below.
Return SQL only, without prose or Markdown.
The query must be a SELECT or WITH ... SELECT. Never modify the database.
Task category: {case.category}.
Category requirement: {task_guidance}

OCEL schema facts:
- Event IDs and object IDs are in event.ocel_id and object.ocel_id.
- Event-object links are in event_object(ocel_event_id, ocel_object_id).
- Object-object links are in object_object(ocel_source_id, ocel_target_id, ocel_qualifier).
- Event time is in the event_<type> tables in ocel_time, joined to event by ocel_id.
- Mapped labels are available through event_map_type and object_map_type.
- Do not assume event.ocel_time exists.
- For Italian quantities/counts, a dot followed by groups of three digits is a thousands separator (2.000 means 2000); decimal fractions such as 0.5 remain decimals.
- For temporal relations, compare event timestamps for the same object/case using event_<type>.ocel_time; do not join unrelated events only by event ID.

Schema:
{schema}

Question:
{case.question}
'''
        previous_sql = None
        feedback = ""
        for attempt in range(1, max_attempts + 1):
            prompt = base_prompt
            if previous_sql:
                prompt += f"\nPrevious SQL:\n{previous_sql}\nExecution feedback:\n{feedback}\nRepair the query addressing this error."
            generated = LLM.invoke(prompt)
            out["attempts"] = attempt
            try:
                sql = _clean_sql(generated)
                out["sql"] = sql
                rows, columns = _execute_read_only(database, sql, timeout_seconds)
                out.update(result=rows, status="success", error=None)
                out["note"] += f" Returned columns: {columns}."
                break
            except Exception as exc:
                previous_sql = out["sql"] or str(getattr(generated, "content", generated))
                feedback = f"{type(exc).__name__}: {exc}"
                out["error"] = feedback
        else:
            out["note"] += f" Failed after {max_attempts} read-only execution attempts."
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    out["seconds"] = time.perf_counter() - started
    return out
