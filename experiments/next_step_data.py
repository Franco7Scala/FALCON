# Data side of the Next-Step Prediction experiment
import ast
import random
import shutil
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

TIPO_CASO = "Container"
SEPARATORE_PREFISSO = " > "
TABELLA_TRACCE = "container_trace"
ATTIVITA_SCONOSCIUTA = "UNKNOWN"


# 1. Traces of the case objects
def _tabelle_evento(conn: sqlite3.Connection) -> Dict[str, str]:
    """Event type name -> name of its per-type table (e.g. 'Reschedule Container' -> 'event_RescheduleContainer')."""
    righe = conn.execute("SELECT ocel_type, ocel_type_map FROM event_map_type").fetchall()
    return {tipo: f"event_{mappa}" for tipo, mappa in righe}


def _tabella_oggetto(conn: sqlite3.Connection, tipo_oggetto: str) -> str:
    riga = conn.execute(
        "SELECT ocel_type_map FROM object_map_type WHERE ocel_type = ?", (tipo_oggetto,)
    ).fetchone()
    if riga is None:
        raise ValueError(f"Tipo di oggetto '{tipo_oggetto}' assente da object_map_type")
    return f"object_{riga[0]}"


def carica_tracce_con_tempi(
    conn: sqlite3.Connection, tipo_caso: str = TIPO_CASO
) -> Dict[str, List[Tuple[str, str]]]:
    """
    {case_id: [(timestamp, activity), ...]} ordered by (timestamp, event id).
    The UNION ALL over the per-type tables is built from event_map_type, so
    no table name is hardcoded.
    """
    tabelle = _tabelle_evento(conn)
    unioni = " UNION ALL ".join(f'SELECT ocel_id, ocel_time FROM "{t}"' for t in tabelle.values())
    sql = f"""
        WITH tempi AS ({unioni})
        SELECT DISTINCT eo.ocel_object_id, e.ocel_type, t.ocel_time, e.ocel_id
        FROM event_object eo
        JOIN object o ON o.ocel_id = eo.ocel_object_id
        JOIN event e ON e.ocel_id = eo.ocel_event_id
        JOIN tempi t ON t.ocel_id = e.ocel_id
        WHERE o.ocel_type = ?
    """
    per_caso: Dict[str, list] = defaultdict(list)
    for caso, attivita, tempo, id_evento in conn.execute(sql, (tipo_caso,)):
        per_caso[caso].append((tempo or "", id_evento, attivita))
    return {
        caso: [(tempo, attivita) for tempo, _, attivita in sorted(eventi)]
        for caso, eventi in per_caso.items()
    }


def carica_tracce(conn: sqlite3.Connection, tipo_caso: str = TIPO_CASO) -> Dict[str, List[str]]:
    """{case_id: [activity, ...]} in temporal order."""
    return {
        caso: [attivita for _, attivita in eventi]
        for caso, eventi in carica_tracce_con_tempi(conn, tipo_caso).items()
    }


# 2. Split: dev / test prefixes + the training cases
def crea_split(
    tracce: Dict[str, List[str]], n_dev: int, n_test: int, seed: int = 42
) -> Tuple[List[Tuple[str, List[str], str]], List[Tuple[str, List[str], str]], List[str]]:
    """
    Returns (casi_dev, casi_test, id_training). The first two are lists of
    (case_id, prefix, target): for each held-out case a random cut point
    1..len-1 is drawn, prefix = activities before it, target = the next one
    (same procedure as the original PMAgent script). Only traces with at
    least 2 events are usable. Deterministic for a given seed.
    """
    usabili = sorted(caso for caso, t in tracce.items() if len(t) > 1)
    if len(usabili) < n_dev + n_test + 10:
        raise ValueError(
            f"Solo {len(usabili)} tracce utilizzabili (>= 2 eventi): non bastano per "
            f"{n_dev} dev + {n_test} test + training. Riduci --n-dev / --n-test."
        )

    rng = random.Random(seed)
    rng.shuffle(usabili)
    id_dev, id_test, id_training = (
        usabili[:n_dev], usabili[n_dev:n_dev + n_test], usabili[n_dev + n_test:]
    )

    rng_tagli = random.Random(seed + 1)

    def con_prefisso(id_casi):
        risultato = []
        for caso in id_casi:
            attivita = tracce[caso]
            taglio = rng_tagli.randint(1, len(attivita) - 1)
            risultato.append((caso, attivita[:taglio], attivita[taglio]))
        return risultato

    return con_prefisso(id_dev), con_prefisso(id_test), id_training


# 3. Training database: the original dump minus the held-out containers
def crea_db_training(
    db_origine: Path, db_training: Path, casi_esclusi: Sequence[str],
    facilitato: bool, tipo_caso: str = TIPO_CASO,
) -> None:
    """
    Copies db_origine to db_training, removes everything that belongs to
    casi_esclusi, then adds the flattened `container_trace` table built from
    what is left. Rebuilt from scratch on every call (deterministic).
    """
    db_training = Path(db_training)
    if db_training.exists():
        db_training.unlink()
    shutil.copyfile(db_origine, db_training)

    conn = sqlite3.connect(db_training)
    try:
        tabelle_evento = _tabelle_evento(conn)
        tabella_caso = _tabella_oggetto(conn, tipo_caso)

        conn.execute("CREATE TEMP TABLE _esclusi (ocel_id TEXT PRIMARY KEY)")
        conn.executemany("INSERT INTO _esclusi VALUES (?)", [(c,) for c in casi_esclusi])
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
        conn.execute(f'DELETE FROM "{tabella_caso}" WHERE ocel_id IN (SELECT ocel_id FROM _esclusi)')
        conn.execute("DELETE FROM object WHERE ocel_id IN (SELECT ocel_id FROM _esclusi)")

        orfani = (
            "SELECT ocel_id FROM _eventi_esclusi "
            "WHERE ocel_id NOT IN (SELECT ocel_event_id FROM event_object)"
        )
        for tabella in tabelle_evento.values():
            conn.execute(f'DELETE FROM "{tabella}" WHERE ocel_id IN ({orfani})')
        conn.execute(f"DELETE FROM event WHERE ocel_id IN ({orfani})")
        conn.commit()

        crea_tabella_tracce(conn, facilitato, tipo_caso)
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


def crea_tabella_tracce(conn: sqlite3.Connection, facilitato: bool, tipo_caso: str = TIPO_CASO) -> None:
    """Builds `container_trace` from the traces currently in the database."""
    tracce = carica_tracce_con_tempi(conn, tipo_caso)

    conn.execute(f"DROP TABLE IF EXISTS {TABELLA_TRACCE}")
    colonne = "case_id TEXT, step INTEGER, activity TEXT, ts TEXT"
    if facilitato:
        colonne += ", prefix TEXT, next_activity TEXT"
    conn.execute(f"CREATE TABLE {TABELLA_TRACCE} ({colonne}, PRIMARY KEY (case_id, step))")

    righe = []
    for caso in sorted(tracce):
        eventi = tracce[caso]
        attivita = [a for _, a in eventi]
        for i, (tempo, nome) in enumerate(eventi):
            riga = [caso, i + 1, nome, tempo]
            if facilitato:
                riga += [
                    SEPARATORE_PREFISSO.join(attivita[: i + 1]),
                    attivita[i + 1] if i + 1 < len(attivita) else None,
                ]
            righe.append(tuple(riga))

    segnaposto = ", ".join("?" * (6 if facilitato else 4))
    conn.executemany(f"INSERT INTO {TABELLA_TRACCE} VALUES ({segnaposto})", righe)
    if facilitato:
        conn.execute(f"CREATE INDEX idx_trace_prefix ON {TABELLA_TRACCE} (prefix)")
    conn.commit()


# 4. Oracle: the hand-written query a perfect text-to-SQL system would write
def previsione_oracolo(conn: sqlite3.Connection, prefisso: Sequence[str], facilitato: bool) -> Optional[str]:
    """
    Activity that most often followed this exact prefix in the training
    traces, or None if the prefix never occurs. Ties are broken
    alphabetically so the result is deterministic. It separates "is the
    information in the database enough?" from "can the model write the query?".
    """
    chiave = SEPARATORE_PREFISSO.join(prefisso)
    if facilitato:
        sql = (
            f"SELECT next_activity FROM {TABELLA_TRACCE} "
            "WHERE prefix = ? AND next_activity IS NOT NULL "
            "GROUP BY next_activity ORDER BY COUNT(*) DESC, next_activity ASC LIMIT 1"
        )
    else:
        sql = (
            "WITH passi AS ("
            " SELECT group_concat(activity, ?) OVER (PARTITION BY case_id ORDER BY step "
            "           ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS prefix,"
            "        LEAD(activity) OVER (PARTITION BY case_id ORDER BY step) AS next_activity"
            f" FROM {TABELLA_TRACCE}) "
            "SELECT next_activity FROM passi WHERE prefix = ? AND next_activity IS NOT NULL "
            "GROUP BY next_activity ORDER BY COUNT(*) DESC, next_activity ASC LIMIT 1"
        )
    parametri = (chiave,) if facilitato else (SEPARATORE_PREFISSO, chiave)
    riga = conn.execute(sql, parametri).fetchone()
    return riga[0] if riga else None


# 5. Questions for the LLM systems, and reading their answer back
def costruisci_domanda(prefisso: Sequence[str]) -> str:
    sequenza = SEPARATORE_PREFISSO.join(prefisso)
    return (
        "Un container ha gia' completato, nell'ordine, queste attivita': "
        f"{sequenza}. Nello storico dei container, qual e' l'attivita' che ha "
        "seguito piu' spesso questa esatta sequenza di attivita'?"
    )


def estrai_attivita_predetta(risultati_query, vocabolario: Sequence[str]) -> str:
    """
    Reads the prediction from the RAW query result (not from the model's
    prose): the first cell of the first row that is a known activity name
    (case-insensitive). Returns UNKNOWN if the query failed, returned
    nothing, or returned something that is not an activity.
    """
    if not risultati_query:
        return ATTIVITA_SCONOSCIUTA
    try:
        valore = ast.literal_eval(risultati_query) if isinstance(risultati_query, str) else risultati_query
    except (ValueError, SyntaxError):
        return ATTIVITA_SCONOSCIUTA
    righe = valore if isinstance(valore, list) else [valore]
    if not righe:
        return ATTIVITA_SCONOSCIUTA

    canoniche = {a.strip().lower(): a for a in vocabolario}
    prima = righe[0]
    for cella in (prima if isinstance(prima, tuple) else (prima,)):
        if isinstance(cella, str) and cella.strip().lower() in canoniche:
            return canoniche[cella.strip().lower()]
    return ATTIVITA_SCONOSCIUTA


# 6. Metrics (same three as the paper)
def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true) if y_true else 0.0


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Macro F1 over the labels present in y_true or y_pred, 0 where undefined
    (same convention as sklearn's f1_score(average='macro', zero_division=0))."""
    f1s = []
    for etichetta in sorted(set(y_true) | set(y_pred)):
        tp = sum(t == etichetta and p == etichetta for t, p in zip(y_true, y_pred))
        fp = sum(t != etichetta and p == etichetta for t, p in zip(y_true, y_pred))
        fn = sum(t == etichetta and p != etichetta for t, p in zip(y_true, y_pred))
        precisione = tp / (tp + fp) if tp + fp else 0.0
        richiamo = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precisione * richiamo / (precisione + richiamo) if precisione + richiamo else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def transizione_valida(prefisso: Sequence[str], predetta: str, sequenze_training: Sequence[Sequence[str]]) -> bool:
    """
    PC-Accuracy building block, same rule as the original script: the
    sequence [prefix + predicted] occurs as a CONTIGUOUS run in at least one
    training trace (anywhere in it, not only at its start).
    """
    if predetta == ATTIVITA_SCONOSCIUTA:
        return False
    cercata = list(prefisso) + [predetta]
    n = len(cercata)
    for sequenza in sequenze_training:
        for i in range(len(sequenza) - n + 1):
            if list(sequenza[i:i + n]) == cercata:
                return True
    return False


def pc_accuracy(prefissi, predette, sequenze_training) -> float:
    if not prefissi:
        return 0.0
    return sum(transizione_valida(p, y, sequenze_training) for p, y in zip(prefissi, predette)) / len(prefissi)


# 7. Dataset statistics and the trivial baseline
def statistiche(tracce: Dict[str, List[str]]) -> dict:
    lunghezze = sorted(len(t) for t in tracce.values())
    n = len(lunghezze)
    return {
        "container_totali": n,
        "tracce_utilizzabili": sum(l > 1 for l in lunghezze),
        "lunghezza_min": lunghezze[0] if n else 0,
        "lunghezza_mediana": lunghezze[n // 2] if n else 0,
        "lunghezza_max": lunghezze[-1] if n else 0,
        "attivita_distinte": len({a for t in tracce.values() for a in t}),
    }


def successore_piu_frequente(sequenze_training: Sequence[Sequence[str]]) -> str:
    """Trivial baseline: always predict the activity that most often comes next, whatever the prefix."""
    conteggi = Counter(s[i + 1] for s in sequenze_training for i in range(len(s) - 1))
    return max(sorted(conteggi), key=lambda a: conteggi[a]) if conteggi else ATTIVITA_SCONOSCIUTA


def campioni_addestramento(sequenze: Sequence[Sequence[str]]) -> List[Tuple[List[str], str]]:
    """Every (prefix, next activity) pair of every training trace."""
    return [(list(s[:i]), s[i]) for s in sequenze for i in range(1, len(s))]