# Run with: pytest test.py -v -s
import ast
import re
import sqlite3
from pathlib import Path
import pytest
from orchestratore.orchestratore import PIPELINE, PipelineState
from utils.utils import DB, DB_PATH


# Test database setup
def _crea_e_popola_database() -> None:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    connessione = sqlite3.connect(DB_PATH)
    try:
        cursore = connessione.cursor()

        cursore.execute("DROP TABLE IF EXISTS ordini")
        cursore.execute("DROP TABLE IF EXISTS clienti")

        cursore.execute(
            """
            CREATE TABLE clienti (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nome TEXT NOT NULL,
                cognome TEXT NOT NULL,
                email TEXT,
                citta TEXT
            )
            """
        )
        cursore.execute(
            """
            CREATE TABLE ordini (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cliente_id INTEGER NOT NULL,
                data_ordine TEXT NOT NULL,
                totale REAL NOT NULL,
                FOREIGN KEY (cliente_id) REFERENCES clienti (id)
            )
            """
        )

        clienti = \
        [
            ("Mario", "Rossi", "mario.rossi@example.com", "Milano"),
            ("Giulia", "Bianchi", "giulia.bianchi@example.com", "Roma"),
            ("Luca", "Verdi", "luca.verdi@example.com", "Torino"),
            ("Anna", "Ferrari", "anna.ferrari@example.com", "Napoli"),
            ("Marco", "Colombo", "marco.colombo@example.com", "Bologna"),
            ("Sara", "Ricci", "sara.ricci@example.com", "Firenze"),
            ("Paolo", "Marino", "paolo.marino@example.com", "Catanzaro"),
            ("Elena", "Greco", "elena.greco@example.com", "Bari"),
        ]
        cursore.executemany(
            "INSERT INTO clienti (nome, cognome, email, citta) VALUES (?, ?, ?, ?)",
            clienti,
        )

        ordini = \
        [
            (1, "2026-01-15", 150.00),
            (1, "2026-03-02", 45.50),
            (2, "2026-01-20", 320.00),
            (3, "2026-02-05", 89.90),
            (3, "2026-04-11", 210.00),
            (4, "2026-02-18", 60.00),
            (5, "2026-03-22", 500.00),
            (5, "2026-05-01", 75.00),
            (6, "2026-01-30", 130.00),
            (7, "2026-04-07", 99.99),
            (8, "2026-05-14", 275.00),
            (8, "2026-06-01", 40.00),
        ]
        cursore.executemany(
            "INSERT INTO ordini (cliente_id, data_ordine, totale) VALUES (?, ?, ?)",
            ordini,
        )

        connessione.commit()
    finally:
        connessione.close()


@pytest.fixture(scope="session", autouse=True)
def database_di_test():
    """Resets negozio.db to the known sample dataset once for the whole test session."""
    _crea_e_popola_database()


# Helpers
def esegui_pipeline(richiesta: str) -> PipelineState:
    """Builds the pipeline's initial state and invokes it for a single request."""
    stato_iniziale: PipelineState = \
    {
        "richiesta_utente": richiesta,
        "schema_database": DB.get_table_info(),
        "informazioni_estratte": None,
        "query_corrente": None,
        "feedback_valutazione": None,
        "valutazione": None,
        "tentativi": 0,
        "risultati_query": None,
        "risposta_finale": None,
        "esito": None,
        "messaggio_errore": None,
    }
    return PIPELINE.invoke(stato_iniziale)


def righe_attese(sql: str) -> list:
    """Runs a ground-truth SQL query directly on negozio.db and returns all rows."""
    connessione = sqlite3.connect(DB_PATH)
    try:
        return connessione.execute(sql).fetchall()
    finally:
        connessione.close()


def valore_atteso(sql: str):
    """Ground-truth value for single-value queries (first column of the first row)."""
    return righe_attese(sql)[0][0]


def estrai_numeri(testo) -> list:
    """Extracts every number from a raw DB-result string, e.g. "[(575.0,)]" -> [575.0]."""
    return [float(n) for n in re.findall(r"\d+(?:\.\d+)?", testo or "")]


def descrivi(risultato: PipelineState) -> str:
    """Multi-line summary of a pipeline run, used in assertion messages."""
    return (
        f"\nesito: {risultato['esito']}"
        f"\ninformazioni estratte: {risultato['informazioni_estratte']}"
        f"\nquery corrente: {risultato['query_corrente']}"
        f"\ntentativi: {risultato['tentativi']}"
        f"\nvalutazione: {risultato['valutazione']}"
        f"\nfeedback valutazione: {risultato['feedback_valutazione']}"
        f"\nrisultati grezzi: {risultato['risultati_query']}"
        f"\nrisposta finale: {risultato['risposta_finale']}"
        f"\nerrore: {risultato['messaggio_errore']}"
    )


def assert_numero_presente(risultato: PipelineState, atteso: float) -> None:
    """
    LOOSE check: the expected number appears somewhere in the raw result.
    Kept only for cases where the correct answer could legitimately come
    back shaped in more than one way (see test_aggregazioni_cliente_con_spesa_maggiore
    below). Do NOT use this for COUNT/SUM/AVG/MIN/MAX-style requests: a
    query that wrongly lists raw rows instead of aggregating can still
    contain the right number by coincidence (e.g. an id matching the
    expected count) and this check would not catch it.
    """
    numeri = estrai_numeri(risultato["risultati_query"])
    assert any(abs(n - atteso) < 0.01 for n in numeri), \
        f"expected value {atteso} not found in the query results.{descrivi(risultato)}"


def parse_risultati(testo) -> list:
    """
    Parses the raw string returned by DB.run() - which is just str() applied
    to a list of row tuples, e.g. "[(12,)]" or "[(1, 1, '2026-01-15', 150.0), ...]"
    - back into real Python data via ast.literal_eval, instead of a loose
    regex over the digits it contains. This lets us check the SHAPE of the
    result (how many rows, how many columns per row), not just whether a
    number shows up somewhere. Returns [] if testo is None/empty/unparseable.
    """
    if not testo:
        return []
    try:
        valore = ast.literal_eval(testo)
    except (ValueError, SyntaxError):
        return []
    return valore if isinstance(valore, list) else [valore]


def assert_valore_aggregato(risultato: PipelineState, atteso: float) -> None:
    """
    STRICT check for a request whose correct answer is a single aggregate
    value (COUNT/SUM/AVG/MIN/MAX, or a single-column lookup like "show me
    X's email"). Requires the raw result to be exactly one row with exactly
    one column, equal to the expected value - this is what catches a query
    like "SELECT * FROM ordini" passed off as an answer to "how many orders
    are there": it returns many rows with many columns, not one number, even
    if one of those columns happens to contain the right value by chance.
    """
    righe = parse_risultati(risultato["risultati_query"])
    assert len(righe) == 1, \
        f"expected exactly one row for an aggregate/single-value request, " \
        f"got {len(righe)}.{descrivi(risultato)}"

    riga = righe[0]
    valori = riga if isinstance(riga, tuple) else (riga,)
    assert len(valori) == 1, \
        f"expected exactly one column for an aggregate/single-value request, " \
        f"got {len(valori)}.{descrivi(risultato)}"

    assert abs(float(valori[0]) - atteso) < 0.01, \
        f"expected {atteso}, got {valori[0]}.{descrivi(risultato)}"


def assert_insieme_valori_singola_colonna(risultato: PipelineState, attesi: list) -> None:
    """
    STRICT check for a request whose correct answer is a fixed-size set of
    single-column rows (e.g. "the 3 highest order totals"). Requires exactly
    len(attesi) rows, each with exactly one column, matching attesi as a set
    (order-independent).
    """
    righe = parse_risultati(risultato["risultati_query"])
    assert len(righe) == len(attesi), \
        f"expected {len(attesi)} rows, got {len(righe)}.{descrivi(risultato)}"

    valori = []
    for riga in righe:
        valori_riga = riga if isinstance(riga, tuple) else (riga,)
        assert len(valori_riga) == 1, \
            f"expected a single column per row, got {len(valori_riga)}.{descrivi(risultato)}"
        valori.append(float(valori_riga[0]))

    for atteso in attesi:
        assert any(abs(v - atteso) < 0.01 for v in valori), \
            f"expected value {atteso} not found among {valori}.{descrivi(risultato)}"


def verifica_valore_numerico(richiesta: str, sql_verita: str) -> None:
    atteso = valore_atteso(sql_verita)
    risultato = esegui_pipeline(richiesta)

    assert risultato["esito"] == "successo", \
        f"the pipeline did not succeed.{descrivi(risultato)}"
    assert_valore_aggregato(risultato, atteso)


# Family 1a: simple aggregations on a single table
CASI_AGGREGAZIONE_SEMPLICI = \
[
    pytest.param("Quanti ordini ci sono in totale?",
                 "SELECT COUNT(*) FROM ordini", id="conteggio-ordini"),
    pytest.param("Quanti clienti ci sono?",
                 "SELECT COUNT(*) FROM clienti", id="conteggio-clienti"),
    pytest.param("Qual e' la somma dei totali di tutti gli ordini?",
                 "SELECT SUM(totale) FROM ordini", id="somma-totali"),
    pytest.param("Qual e' il totale medio degli ordini?",
                 "SELECT AVG(totale) FROM ordini", id="media-totali"),
    pytest.param("Qual e' il totale dell'ordine piu' alto?",
                 "SELECT MAX(totale) FROM ordini", id="massimo"),
    pytest.param("Qual e' il totale dell'ordine piu' basso?",
                 "SELECT MIN(totale) FROM ordini", id="minimo"),
]


@pytest.mark.parametrize("richiesta, sql_verita", CASI_AGGREGAZIONE_SEMPLICI)
def test_aggregazioni_semplici(richiesta, sql_verita):
    verifica_valore_numerico(richiesta, sql_verita)


# Family 1b: aggregations needing a JOIN, a filter, a date range or a subquery.
CASI_AGGREGAZIONE_COMPLESSI = \
[
    pytest.param(
        "Quanto ha speso in totale Marco Colombo?",
        "SELECT SUM(o.totale) FROM ordini o JOIN clienti c ON c.id = o.cliente_id "
        "WHERE c.nome = 'Marco' AND c.cognome = 'Colombo'",
        id="spesa-di-un-cliente"),
    pytest.param(
        "Quanti ordini ha fatto Luca Verdi?",
        "SELECT COUNT(*) FROM ordini o JOIN clienti c ON c.id = o.cliente_id "
        "WHERE c.nome = 'Luca' AND c.cognome = 'Verdi'",
        id="ordini-di-un-cliente"),
    pytest.param(
        "Quanti ordini hanno un totale superiore a 100?",
        "SELECT COUNT(*) FROM ordini WHERE totale > 100",
        id="ordini-sopra-soglia"),
    pytest.param(
        "Qual e' il totale degli ordini di gennaio 2026?",
        "SELECT SUM(totale) FROM ordini "
        "WHERE data_ordine >= '2026-01-01' AND data_ordine < '2026-02-01'",
        id="somma-per-mese"),
    pytest.param(
        "Quanto hanno speso in totale i clienti di Milano?",
        "SELECT SUM(o.totale) FROM ordini o JOIN clienti c ON c.id = o.cliente_id "
        "WHERE c.citta = 'Milano'",
        id="somma-per-citta"),
    pytest.param(
        "Quanti clienti hanno fatto piu' di un ordine?",
        "SELECT COUNT(*) FROM (SELECT cliente_id FROM ordini "
        "GROUP BY cliente_id HAVING COUNT(*) > 1)",
        id="group-by-having"),
]


@pytest.mark.parametrize("richiesta, sql_verita", CASI_AGGREGAZIONE_COMPLESSI)
def test_aggregazioni_complesse(richiesta, sql_verita):
    verifica_valore_numerico(richiesta, sql_verita)


# Family 1c: ranking requests (ORDER BY + LIMIT)
def test_aggregazioni_cliente_con_spesa_maggiore():
    cognome_atteso = valore_atteso(
        "SELECT c.cognome FROM clienti c JOIN ordini o ON o.cliente_id = c.id "
        "GROUP BY c.id ORDER BY SUM(o.totale) DESC LIMIT 1"
    )
    risultato = esegui_pipeline("Qual e' il cliente che ha speso di piu' in totale?")

    assert risultato["esito"] == "successo", \
        f"the pipeline did not succeed.{descrivi(risultato)}"

    testo = f"{risultato['risultati_query']} {risultato['risposta_finale']}".lower()
    assert cognome_atteso.lower() in testo, \
        f"expected '{cognome_atteso}' in the answer.{descrivi(risultato)}"


def test_aggregazioni_tre_ordini_piu_alti():
    attesi = [riga[0] for riga in righe_attese(
        "SELECT totale FROM ordini ORDER BY totale DESC LIMIT 3"
    )]
    risultato = esegui_pipeline("Mostrami i tre ordini con il totale piu' alto")

    assert risultato["esito"] == "successo", \
        f"the pipeline did not succeed.{descrivi(risultato)}"
    assert_insieme_valori_singola_colonna(risultato, attesi)


# Family 2: positive controls - existing columns must be answered.
CONTROLLI_POSITIVI = \
[
    pytest.param("Mostrami l'email di Mario Rossi",
                 "SELECT email FROM clienti WHERE nome = 'Mario' AND cognome = 'Rossi'",
                 id="colonna-email"),
    pytest.param("In quale citta' abita Paolo Marino?",
                 "SELECT citta FROM clienti WHERE nome = 'Paolo' AND cognome = 'Marino'",
                 id="colonna-citta"),
]


@pytest.mark.parametrize("richiesta, sql_verita", CONTROLLI_POSITIVI)
def test_controllo_positivo_colonna_esistente(richiesta, sql_verita):
    atteso = valore_atteso(sql_verita)
    risultato = esegui_pipeline(richiesta)

    assert risultato["esito"] == "successo", \
        f"the pipeline did not succeed.{descrivi(risultato)}"

    righe = parse_risultati(risultato["risultati_query"])
    assert len(righe) == 1, \
        f"expected exactly one row, got {len(righe)}.{descrivi(risultato)}"
    riga = righe[0]
    valori = riga if isinstance(riga, tuple) else (riga,)
    assert len(valori) == 1, \
        f"expected exactly one column, got {len(valori)} - the query likely " \
        f"returned extra columns instead of just the requested one.{descrivi(risultato)}"
    assert str(valori[0]).lower() == str(atteso).lower(), \
        f"expected '{atteso}', got '{valori[0]}'.{descrivi(risultato)}"


# Family 3: requests about entities that do NOT exist
CASI_ENTITA_INESISTENTI = \
[
    pytest.param("Mostrami le serre del cliente Mario Rossi", id="tabella-serre"),
    pytest.param("Mostrami le fatture del cliente Luca Verdi", id="tabella-fatture"),
    pytest.param("Elenca i prodotti acquistati da Giulia Bianchi", id="tabella-prodotti"),
    pytest.param("Mostrami il numero di telefono di Mario Rossi", id="colonna-telefono"),
    pytest.param("Qual e' lo stipendio di Anna Ferrari?", id="colonna-stipendio"),
]


@pytest.mark.parametrize("richiesta", CASI_ENTITA_INESISTENTI)
def test_dati_inesistenti_vengono_rifiutati_subito(richiesta):
    risultato = esegui_pipeline(richiesta)

    assert risultato["esito"] == "errore", \
        "the pipeline answered even though the requested data does not exist " \
        f"in the schema.{descrivi(risultato)}"
    assert risultato["query_corrente"] is None, \
        "Agent 2 was invoked even though Agent 1 should have rejected the " \
        f"request before reaching it.{descrivi(risultato)}"
    assert risultato["tentativi"] == 0, \
        "the pipeline entered the generation/revision loop instead of " \
        f"rejecting immediately at extraction.{descrivi(risultato)}"
    assert risultato["messaggio_errore"] not in (None, ""), \
        f"esito is 'errore' but no message is available for the user.{descrivi(risultato)}"