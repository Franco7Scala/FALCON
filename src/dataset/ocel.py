"""Read-only OCEL SQLite schema inspection and prompt guidance."""
from __future__ import annotations

from pathlib import Path
import sqlite3


def connect_readonly(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)


def schema(path: Path) -> dict[str, list[dict[str, str | None]]]:
    with connect_readonly(path) as con:
        names = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        return {name: [{"name": r[1], "type": r[2]} for r in con.execute(f'PRAGMA table_info("{name}")')]
                for name in names}


def object_types(path: Path) -> set[str]:
    with connect_readonly(path) as con:
        return {value for row in con.execute("SELECT ocel_type, ocel_type_map FROM object_map_type") for value in row}


def orchestrator_schema_guidance(path: Path) -> str:
    """Describe the real OCEL event-time layout to the SQL-generation prompt."""
    with connect_readonly(path) as con:
        event_types = con.execute(
            "SELECT ocel_type, ocel_type_map FROM event_map_type ORDER BY ocel_type_map"
        ).fetchall()
        obj_types = con.execute(
            "SELECT ocel_type, ocel_type_map FROM object_map_type ORDER BY ocel_type_map"
        ).fetchall()
        event_cols = {r[1] for r in con.execute('PRAGMA table_info("event")')}
    tables = ", ".join(f"event_{mapped} (mapped type {raw!r}; columns include ocel_id, ocel_time)"
                        for raw, mapped in event_types)
    return (
        "\n\nOCEL SCHEMA DETAILS (from the live SQLite database):\n"
        f"- event columns: {', '.join(sorted(event_cols))}. Do not refer to event.ocel_time unless listed above.\n"
        "- Timestamps are stored in the per-event-type tables, joined by ocel_id to event.ocel_id.\n"
        f"- Per-event-type timestamp tables: {tables}.\n"
        "- Raw-to-mapped object type labels: "
        + ", ".join(f"{raw!r} -> {mapped!r}" for raw, mapped in obj_types)
        + ".\n"
        "- Event-object links are in event_object (ocel_event_id, ocel_object_id); object type is in object.ocel_type, "
        "with labels in object_map_type. For object traces, the case key is ocel_object_id, not event.ocel_id.\n"
        "- For temporal event queries, construct an event_times CTE by UNION ALL-ing "
        "SELECT ocel_id AS event_id, ocel_time AS timestamp from each listed event_<mapped type> table, "
        "then join event_times to event on event_id=event.ocel_id. Join event_object and object to reconstruct "
        "object-centric sequences, and order ties deterministically by timestamp, event id.\n"
    )
