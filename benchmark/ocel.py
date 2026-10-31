"""Read-only access and PMAgent-only object-type flattening for OCEL SQLite."""
from __future__ import annotations

from pathlib import Path
import shutil
import sqlite3
import tempfile


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
        object_types = con.execute(
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
        + ", ".join(f"{raw!r} -> {mapped!r}" for raw, mapped in object_types)
        + ".\n"
        "- Event-object links are in event_object (ocel_event_id, ocel_object_id); object type is in object.ocel_type, "
        "with labels in object_map_type. For object traces, the case key is ocel_object_id, not event.ocel_id.\n"
        "- For temporal event queries, construct an event_times CTE by UNION ALL-ing "
        "SELECT ocel_id AS event_id, ocel_time AS timestamp from each listed event_<mapped type> table, "
        "then join event_times to event on event_id=event.ocel_id. Join event_object and object to reconstruct "
        "object-centric sequences, and order ties deterministically by timestamp, event id.\n"
    )


def raw_object_type(path: Path, label: str) -> str | None:
    """Resolve a workbook's raw or mapped OCEL type label to next_step_data's raw label."""
    with connect_readonly(path) as con:
        row = con.execute("SELECT ocel_type FROM object_map_type WHERE ocel_type=? OR ocel_type_map=? LIMIT 1",
                          (label, label)).fetchone()
    return row[0] if row else None


def create_type_flattening(source: Path, object_type: str, directory: Path) -> Path:
    """Copy source DB, then build next_step_data's container_trace for one type only."""
    from benchmark.next_step_data import build_trace_table
    directory.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in object_type)
    target = directory / f"pmagent_{safe}.sqlite"
    shutil.copy2(source, target)
    raw_type = raw_object_type(source, object_type)
    if raw_type is None:
        raise ValueError(f"No OCEL object type maps to {object_type!r}")
    with sqlite3.connect(target) as con:
        build_trace_table(con, simplified=False, case_type=raw_type)
        # Workbook prefixes/expected activities use the mapped compact activity labels.
        con.execute("""UPDATE container_trace SET activity=(
            SELECT ocel_type_map FROM event_map_type WHERE event_map_type.ocel_type=container_trace.activity
        ) WHERE activity IN (SELECT ocel_type FROM event_map_type)""")
        con.commit()
    return target


class TypeFlattenings:
    def __init__(self, database: Path):
        self.database = database
        self.temp = tempfile.TemporaryDirectory(prefix="ocel_pma_")
        self.cache: dict[str, Path] = {}

    def for_type(self, object_type: str) -> Path:
        if object_type not in self.cache:
            self.cache[object_type] = create_type_flattening(self.database, object_type, Path(self.temp.name))
        return self.cache[object_type]

    def close(self) -> None:
        self.temp.cleanup()
