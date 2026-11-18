"""Read-only generic SQL extension used by PMAgent and PM4Py adapters."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import re
import sqlite3
import time
import ast

from src.dataset.ocel import connect_readonly


def _schema_text(database: Path) -> str:
    with connect_readonly(database) as con:
        rows = con.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        event_mappings = con.execute(
            'SELECT ocel_type, ocel_type_map FROM event_map_type ORDER BY ocel_type_map'
        ).fetchall()
        object_mappings = con.execute(
            'SELECT ocel_type, ocel_type_map FROM object_map_type ORDER BY ocel_type_map'
        ).fetchall()
    return "\n\n".join(sql for _, sql in rows if sql) + (
        "\n\nACTIVITY TYPE MAPPINGS (raw ocel_type -> mapped label):\n"
        + "\n".join(f"{raw!r} -> {mapped!r}" for raw, mapped in event_mappings)
        + "\n\nOBJECT TYPE MAPPINGS (raw ocel_type -> mapped label):\n"
        + "\n".join(f"{raw!r} -> {mapped!r}" for raw, mapped in object_mappings)
    )


def _clean_sql(output: Any) -> str:
    text = getattr(output, "content", output)
    if isinstance(text, list):
        text = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in text)
    text = str(text).strip().lstrip("\ufeff")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    fence = re.search(r"```(?:sql)?\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    else:
        # Recover a truncated/open code fence instead of passing backticks to SQLite.
        text = re.sub(r"^```(?:sql)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"```\s*$", "", text).strip()
    text = text.replace("```", "").strip()
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
        from src.agents.query_agent import LLM

        schema = _schema_text(database)
        task_guidance = {
            "scalar": "Return one row and one value answering the numeric/text scalar question.",
            "boolean": "Return exactly one row and one column with 'SI' when the complete proposition is true, otherwise 'NO'. Use CASE WHEN/EXISTS or compare scalar subquery counts; never return a raw count, row list, or multiple columns.",
            "set": "Return all requested rows and columns. Preserve requested filters and ordering; do not aggregate unless the question asks for it.",
            "next_step_case": "This is a case-specific next-activity prediction. Filter the exact object/case named in the question, follow its ordered trace, and return only the activity immediately after the described prefix as one row and one value. Do not aggregate across other objects.",
            "next_step_history": "This is a historical next-activity prediction. Match the complete ordered prefix on the specified object type, then return the most frequent immediate next activity as one row and one value.",
            "next_step_what_if": "This is a what-if next-activity prediction. Apply the hypothetical condition in the question, then return the requested alternative next activity as one row and one value.",
        }.get(case.category, "Answer the request using the relevant rows and columns.")
        if case.category == "next_step_probability":
            expected = case.expected
            if isinstance(expected, str):
                try:
                    expected = ast.literal_eval(expected)
                except (ValueError, SyntaxError):
                    expected = None
            rows = expected if isinstance(expected, (list, tuple)) else []
            if rows and isinstance(rows[0], (list, tuple)) and len(rows[0]) >= 3:
                first = rows[0][0]
                if isinstance(first, str):
                    task_guidance = (
                        "Return a categorical probability distribution with the requested label, event, or person, its count, and its percentage. "
                        "Use the exact denominator implied by the question, calculate percentages to two decimal places, and preserve all requested categories."
                    )
                else:
                    task_guidance = (
                        "Return exactly one row with three numeric columns: numerator (matching objects/events), denominator (eligible objects/events), "
                        "and probability_percentage rounded to two decimals. Compute the probability as numerator * 100.0 / denominator. "
                        "Apply all lifecycle, temporal, attribute, relationship, and What-If conditions in the question to the correct entity."
                    )
            else:
                task_guidance = (
                    "Answer the probability question with a numerator, denominator, and percentage when the expected result has that form. "
                    "Use the exact population and event conditions stated in the question; do not substitute a next-activity distribution unless asked for one."
                )
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
- The schema appendix lists the exact raw-to-mapped labels. Do not compare a mapped activity label directly to event.ocel_type if the raw value is different; use event_map_type.ocel_type_map or the matching raw value.
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
