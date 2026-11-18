# Data side of the Next-Step Prediction experiment
import ast
import random
import shutil
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

CASE_TYPE = "Container"
PREFIX_SEPARATOR = " > "
TRACE_TABLE = "container_trace"
UNKNOWN_ACTIVITY = "UNKNOWN"


# 1. Traces of the case objects
def _event_tables(conn: sqlite3.Connection) -> Dict[str, str]:
    """Event type name -> name of its per-type table (e.g. 'Reschedule Container' -> 'event_RescheduleContainer')."""
    rows = conn.execute("SELECT ocel_type, ocel_type_map FROM event_map_type").fetchall()
    return {object_kind: f"event_{mapping}" for object_kind, mapping in rows}


def _object_table(conn: sqlite3.Connection, object_type: str) -> str:
    row = conn.execute(
        "SELECT ocel_type_map FROM object_map_type WHERE ocel_type = ?", (object_type,)
    ).fetchone()
    if row is None:
        raise ValueError(f"Tipo di oggetto '{object_type}' assente da object_map_type")
    return f"object_{row[0]}"


def load_traces_with_timestamps(
    conn: sqlite3.Connection, case_type: str = CASE_TYPE
) -> Dict[str, List[Tuple[str, str]]]:
    """
    {case_id: [(timestamp, activity), ...]} ordered by (timestamp, event id).
    The UNION ALL over the per-type tables is built from event_map_type, so
    no table name is hardcoded.
    """
    tables = _event_tables(conn)
    unions = " UNION ALL ".join(f'SELECT ocel_id, ocel_time FROM "{t}"' for t in tables.values())
    sql = f"""
        WITH tempi AS ({unions})
        SELECT DISTINCT eo.ocel_object_id, e.ocel_type, t.ocel_time, e.ocel_id
        FROM event_object eo
        JOIN object o ON o.ocel_id = eo.ocel_object_id
        JOIN event e ON e.ocel_id = eo.ocel_event_id
        JOIN tempi t ON t.ocel_id = e.ocel_id
        WHERE o.ocel_type = ?
    """
    by_case: Dict[str, list] = defaultdict(list)
    for case, activity, time, event_id in conn.execute(sql, (case_type,)):
        by_case[case].append((time or "", event_id, activity))
    return {
        case: [(time, activity) for time, _, activity in sorted(events)]
        for case, events in by_case.items()
    }


def load_traces(conn: sqlite3.Connection, case_type: str = CASE_TYPE) -> Dict[str, List[str]]:
    """{case_id: [activity, ...]} in temporal order."""
    return {
        case: [activity for _, activity in events]
        for case, events in load_traces_with_timestamps(conn, case_type).items()
    }


# 2. Split: dev / test prefixes + the training cases
def create_split(
    traces: Dict[str, List[str]], n_dev: int, n_test: int, seed: int = 42
) -> Tuple[List[Tuple[str, List[str], str]], List[Tuple[str, List[str], str]], List[str]]:
    """
    Returns (dev_cases, test_cases, training_ids). The first two are lists of
    (case_id, prefix, target): for each held-out case a random cut point
    1..len-1 is drawn, prefix = activities before it, target = the next one
    (following standard trace-splitting procedure). Only traces with at
    least 2 events are usable. Deterministic for a given seed.
    """
    usable_cases = sorted(case for case, t in traces.items() if len(t) > 1)
    if len(usable_cases) < n_dev + n_test + 10:
        raise ValueError(
            f"Only {len(usable_cases)} usable traces (>= 2 events): not enough for "
            f"{n_dev} dev + {n_test} test + training. Reduce --n-dev / --n-test."
        )

    rng = random.Random(seed)
    rng.shuffle(usable_cases)
    dev_ids, test_ids, training_ids = (
        usable_cases[:n_dev], usable_cases[n_dev:n_dev + n_test], usable_cases[n_dev + n_test:]
    )

    rng_tagli = random.Random(seed + 1)

    def with_prefix(case_ids):
        result = []
        for case in case_ids:
            activity = traces[case]
            cut_point = rng_tagli.randint(1, len(activity) - 1)
            result.append((case, activity[:cut_point], activity[cut_point]))
        return result

    return with_prefix(dev_ids), with_prefix(test_ids), training_ids


# 3. Training database: the original dump minus the held-out containers
def create_training_database(
    source_database: Path, training_database: Path, excluded_cases: Sequence[str],
    simplified: bool, case_type: str = CASE_TYPE,
) -> None:
    """
    Copies source_database to training_database and removes everything that belongs to
    casi_esclusi, then adds the flattened `container_trace` table built from
    what is left. Rebuilt from scratch on every call (deterministic).
    """
    training_database = Path(training_database)
    if training_database.exists():
        training_database.unlink()
    shutil.copyfile(source_database, training_database)

    conn = sqlite3.connect(training_database)
    try:
        event_tables = _event_tables(conn)
        case_table = _object_table(conn, case_type)

        conn.execute("CREATE TEMP TABLE _esclusi (ocel_id TEXT PRIMARY KEY)")
        conn.executemany("INSERT INTO _esclusi VALUES (?)", [(c,) for c in excluded_cases])
        conn.execute(
            "CREATE TEMP TABLE _eventi_esclusi AS "
            "SELECT DISTINCT ocel_event_id AS ocel_id FROM event_object "
            "WHERE ocel_object_id IN (SELECT ocel_id FROM _esclusi)"
        )

        conn.execute("DELETE FROM event_object WHERE ocel_object_id IN (SELECT ocel_id FROM _esclusi)")
        conn.execute(
            "DELETE FROM object_object WHERE ocel_source_id IN (SELECT ocel_id FROM _esclusi) "
            "OR ocel_target_id IN (SELECT ocel_id FROM _esclusi)"
        )
        conn.execute(f'DELETE FROM "{case_table}" WHERE ocel_id IN (SELECT ocel_id FROM _esclusi)')
        conn.execute("DELETE FROM object WHERE ocel_id IN (SELECT ocel_id FROM _esclusi)")

        orphans = (
            "SELECT ocel_id FROM _eventi_esclusi "
            "WHERE ocel_id NOT IN (SELECT ocel_event_id FROM event_object)"
        )
        for table in event_tables.values():
            conn.execute(f'DELETE FROM "{table}" WHERE ocel_id IN ({orphans})')
        conn.execute(f"DELETE FROM event WHERE ocel_id IN ({orphans})")
        conn.commit()

        build_trace_table(conn, simplified, case_type)
        conn.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_event_type ON event (ocel_type);
            CREATE INDEX IF NOT EXISTS idx_object_type ON object (ocel_type);
            CREATE INDEX IF NOT EXISTS idx_event_object_event ON event_object (ocel_event_id);
            CREATE INDEX IF NOT EXISTS idx_event_object_object ON event_object (ocel_object_id);
            """
        )
        conn.commit()
    finally:
        conn.close()


def build_trace_table(conn: sqlite3.Connection, simplified: bool, case_type: str = CASE_TYPE) -> None:
    """Builds `container_trace` from the traces currently in the database."""
    traces = load_traces_with_timestamps(conn, case_type)

    conn.execute(f"DROP TABLE IF EXISTS {TRACE_TABLE}")
    columns = "case_id TEXT, step INTEGER, activity TEXT, ts TEXT"
    if simplified:
        columns += ", prefix TEXT, next_activity TEXT"
    conn.execute(f"CREATE TABLE {TRACE_TABLE} ({columns}, PRIMARY KEY (case_id, step))")

    rows = []
    for case in sorted(traces):
        events = traces[case]
        activity = [a for _, a in events]
        for i, (time, name) in enumerate(events):
            row = [case, i + 1, name, time]
            if simplified:
                row += [
                    PREFIX_SEPARATOR.join(activity[: i + 1]),
                    activity[i + 1] if i + 1 < len(activity) else None,
                ]
            rows.append(tuple(row))

    placeholders = ", ".join("?" * (6 if simplified else 4))
    conn.executemany(f"INSERT INTO {TRACE_TABLE} VALUES ({placeholders})", rows)
    if simplified:
        conn.execute(f"CREATE INDEX idx_trace_prefix ON {TRACE_TABLE} (prefix)")
    conn.commit()


# 4. Oracle: the hand-written query a perfect text-to-SQL system would write
def oracle_prediction(conn: sqlite3.Connection, prefix: Sequence[str], simplified: bool) -> Optional[str]:
    """
    Activity that most often followed this exact prefix in the training
    traces, or None if the prefix never occurs. Ties are broken
    alphabetically so the result is deterministic. It separates "is the
    information in the database enough?" from "can the model write the query?".
    """
    key = PREFIX_SEPARATOR.join(prefix)
    if simplified:
        sql = (
            f"SELECT next_activity FROM {TRACE_TABLE} "
            "WHERE prefix = ? AND next_activity IS NOT NULL "
            "GROUP BY next_activity ORDER BY COUNT(*) DESC, next_activity ASC LIMIT 1"
        )
    else:
        sql = (
            "WITH passi AS ("
            " SELECT group_concat(activity, ?) OVER (PARTITION BY case_id ORDER BY step "
            "           ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS prefix,"
            "        LEAD(activity) OVER (PARTITION BY case_id ORDER BY step) AS next_activity"
            f" FROM {TRACE_TABLE}) "
            "SELECT next_activity FROM passi WHERE prefix = ? AND next_activity IS NOT NULL "
            "GROUP BY next_activity ORDER BY COUNT(*) DESC, next_activity ASC LIMIT 1"
        )
    parameters = (key,) if simplified else (PREFIX_SEPARATOR, key)
    row = conn.execute(sql, parameters).fetchone()
    return row[0] if row else None


# 5. Questions for the LLM systems, and reading their answer back
def build_question(prefix: Sequence[str]) -> str:
    sequence = PREFIX_SEPARATOR.join(prefix)
    return (
        "Un container ha gia' completato, nell'ordine, queste attivita': "
        f"{sequence}. Nello storico dei container, qual e' l'attivita' che ha "
        "seguito piu' spesso questa esatta sequenza di attivita'?"
    )


def extract_predicted_activity(query_results, vocabulary: Sequence[str]) -> str:
    """
    Reads the prediction from the RAW query result (not from the model's
    prose): the first cell of the first row that is a known activity name
    (case-insensitive). Returns UNKNOWN if the query failed, returned
    nothing, or returned something that is not an activity.
    """
    if not query_results:
        return UNKNOWN_ACTIVITY
    try:
        value = ast.literal_eval(query_results) if isinstance(query_results, str) else query_results
    except (ValueError, SyntaxError):
        return UNKNOWN_ACTIVITY
    rows = value if isinstance(value, list) else [value]
    if not rows:
        return UNKNOWN_ACTIVITY

    canoniche = {a.strip().lower(): a for a in vocabulary}
    first_row = rows[0]
    for cella in (first_row if isinstance(first_row, tuple) else (first_row,)):
        if isinstance(cella, str) and cella.strip().lower() in canoniche:
            return canoniche[cella.strip().lower()]
    return UNKNOWN_ACTIVITY


# 6. Metrics (same three as the paper)
def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true) if y_true else 0.0


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Macro F1 over the labels present in y_true or y_pred, 0 where undefined
    (same convention as sklearn's f1_score(average='macro', zero_division=0))."""
    f1s = []
    for label in sorted(set(y_true) | set(y_pred)):
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        precisione = tp / (tp + fp) if tp + fp else 0.0
        richiamo = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precisione * richiamo / (precisione + richiamo) if precisione + richiamo else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def is_valid_transition(prefix: Sequence[str], predicted_activity: str, training_sequences: Sequence[Sequence[str]]) -> bool:
    """
    PC-Accuracy building block, same rule as the original script: the
    sequence [prefix + predicted] occurs as a CONTIGUOUS run in at least one
    training trace (anywhere in it, not only at its start).
    """
    if predicted_activity == UNKNOWN_ACTIVITY:
        return False
    target_sequence = list(prefix) + [predicted_activity]
    n = len(target_sequence)
    for sequence in training_sequences:
        for i in range(len(sequence) - n + 1):
            if list(sequence[i:i + n]) == target_sequence:
                return True
    return False


def prefix_consistency_accuracy(prefixes, predictions, training_sequences) -> float:
    if not prefixes:
        return 0.0
    return sum(is_valid_transition(p, y, training_sequences) for p, y in zip(prefixes, predictions)) / len(prefixes)


# 7. Dataset statistics and the trivial baseline
def trace_statistics(traces: Dict[str, List[str]]) -> dict:
    lengths = sorted(len(t) for t in traces.values())
    n = len(lengths)
    return {
        "container_totali": n,
        "tracce_utilizzabili": sum(l > 1 for l in lengths),
        "lunghezza_min": lengths[0] if n else 0,
        "lunghezza_mediana": lengths[n // 2] if n else 0,
        "lunghezza_max": lengths[-1] if n else 0,
        "attivita_distinte": len({a for t in traces.values() for a in t}),
    }


def most_frequent_successor(training_sequences: Sequence[Sequence[str]]) -> str:
    """Trivial baseline: always predict the activity that most often comes next, whatever the prefix."""
    counts = Counter(s[i + 1] for s in training_sequences for i in range(len(s) - 1))
    return max(sorted(counts), key=lambda a: counts[a]) if counts else UNKNOWN_ACTIVITY


def training_samples(sequences: Sequence[Sequence[str]]) -> List[Tuple[List[str], str]]:
    """Every (prefix, next activity) pair of every training trace."""
    return [(list(s[:i]), s[i]) for s in sequences for i in range(1, len(s))]
