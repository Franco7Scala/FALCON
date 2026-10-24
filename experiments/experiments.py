# Experimental evaluation of our text-to-SQL multi-agent pipeline on the
# OCEL 2.0 "Logistics" simulation log
import ast
import json
import sqlite3
import threading
import time
from pathlib import Path
from langchain_community.utilities import SQLDatabase
import agenti.response_agent as agente_3_modulo
import orchestratore.orchestratore as pipeline_modulo
from orchestratore.orchestratore import PIPELINE, crea_stato_iniziale, MAX_TENTATIVI
from agenti.query_agent import genera_query
from agenti.request_agent import LLM as LLM_AGENTE_1
from agenti.query_agent import LLM as LLM_AGENTE_2
from agenti.response_agent import LLM as LLM_AGENTE_3


# 0. Dataset connection
LOGISTICS_DB_PATH = Path(__file__).resolve().parent.parent / "databases" / "container_logistics.sqlite"

# Hard limit on how long any single SQL query may run, in seconds.
QUERY_TIMEOUT_SECONDI = 60

if not LOGISTICS_DB_PATH.exists():
    raise FileNotFoundError(
        f"'{LOGISTICS_DB_PATH}' non esiste. Se ci si connettesse comunque, "
        "sqlite3/SQLAlchemy creerebbero silenziosamente un file vuoto "
        "invece di dare un errore subito. Converti prima il dataset OCEL "
        "(vedi il commento in cima al file) e/o aggiorna questo percorso."
    )


def crea_indici_ocel() -> None:
    """
    Creates indexes on the OCEL core tables if they don't exist yet. SQLite
    does NOT automatically index foreign-key columns, so without these every
    JOIN / WHERE on the core tables is a full scan. Idempotent, data-neutral.
    """
    connessione = sqlite3.connect(LOGISTICS_DB_PATH)
    try:
        connessione.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_event_type ON event (ocel_type);
            CREATE INDEX IF NOT EXISTS idx_object_type ON object (ocel_type);
            CREATE INDEX IF NOT EXISTS idx_event_object_event ON event_object (ocel_event_id);
            CREATE INDEX IF NOT EXISTS idx_event_object_object ON event_object (ocel_object_id);
            CREATE INDEX IF NOT EXISTS idx_object_object_source ON object_object (ocel_source_id);
            CREATE INDEX IF NOT EXISTS idx_object_object_target ON object_object (ocel_target_id);
            """
        )
        connessione.commit()
    finally:
        connessione.close()


def _esegui_query_con_timeout(sql: str, timeout_secondi: int = QUERY_TIMEOUT_SECONDI) -> list:
    """
    Executes sql on the OCEL SQLite file and returns all rows, really
    interrupting the statement (Connection.interrupt from a timer thread) if
    it runs longer than timeout_secondi (-> sqlite3.OperationalError).
    """
    connessione = sqlite3.connect(LOGISTICS_DB_PATH, check_same_thread=False)
    timer = threading.Timer(timeout_secondi, connessione.interrupt)
    timer.start()
    try:
        return connessione.execute(sql).fetchall()
    finally:
        timer.cancel()
        connessione.close()


class DBConTimeout:
    """
    Drop-in replacement for the SQLDatabase the agents use as `DB`: run()
    goes through _esegui_query_con_timeout and returns str(rows) (the same
    format SQLDatabase.run() produces); everything else is delegated.
    """

    def __init__(self, db_reale: SQLDatabase, timeout_secondi: int = QUERY_TIMEOUT_SECONDI):
        self._db_reale = db_reale
        self._timeout_secondi = timeout_secondi

    def run(self, command: str, *args, **kwargs) -> str:
        return str(_esegui_query_con_timeout(command, self._timeout_secondi))

    def __getattr__(self, nome):
        return getattr(self._db_reale, nome)


crea_indici_ocel()

DB_LOGISTICS = DBConTimeout(SQLDatabase.from_uri(f"sqlite:///{LOGISTICS_DB_PATH}"))

# Redirect the agents' `DB` to the OCEL dataset
agente_3_modulo.DB = DB_LOGISTICS
pipeline_modulo.DB = DB_LOGISTICS


def stampa_schema_ocel() -> None:
    """Prints the real schema of the converted OCEL SQLite file, and its size."""
    schema = DB_LOGISTICS.get_table_info()
    print(schema)
    print(f"\n[Dimensione dello schema passato agli agenti: {len(schema)} caratteri]")


# 1. Benchmark questions
CASI_BENCHMARK = \
[
    ("Quanti eventi ci sono in totale nel log?",
     "SELECT COUNT(*) FROM event"),
    ("Quanti oggetti ci sono in totale nel log?",
     "SELECT COUNT(*) FROM object"),
    ("Quanti tipi di evento diversi sono presenti?",
     "SELECT COUNT(DISTINCT ocel_type) FROM event"),
    ("Quanti tipi di oggetto diversi sono presenti?",
     "SELECT COUNT(DISTINCT ocel_type) FROM object"),
    ("Quanti eventi di tipo 'Depart' sono stati registrati?",
     "SELECT COUNT(*) FROM event WHERE ocel_type = 'Depart'"),
    ("Quante relazioni evento-oggetto sono registrate in totale?",
     "SELECT COUNT(*) FROM event_object"),
    ("Quanti oggetti di tipo Vehicle ci sono?",
     "SELECT COUNT(*) FROM object WHERE ocel_type = 'Vehicle'"),
    ("Quanti eventi di tipo 'Reschedule Container' ci sono?",
     "SELECT COUNT(*) FROM event WHERE ocel_type = 'Reschedule Container'"),
    ("Quanti container diversi sono stati rischedulati?",
     "SELECT COUNT(DISTINCT eo.ocel_object_id) FROM event_object eo "
     "JOIN event e ON e.ocel_id = eo.ocel_event_id "
     "JOIN object o ON o.ocel_id = eo.ocel_object_id "
     "WHERE e.ocel_type = 'Reschedule Container' AND o.ocel_type = 'Container'"),
]

# UNANSWERABLE questions:
CASI_NON_RISPONDIBILI = \
[
    "Qual e' il costo medio di trasporto per container spedito?",
    "Qual e' il nome dell'autista di ciascun camion?",
    "Quanti litri di carburante ha consumato ogni veicolo?",
    "Qual e' il prezzo totale degli ordini dei clienti?",
    "Quanti chilometri ha percorso ciascun camion?",
    "Quali sono gli stipendi orari degli operai del magazzino?",
    "Che tempo fa oggi a Milano?",
]


# 2. Qualitative case study
CASO_FATTO_DEDUCIBILE = "Quanti container sono stati rischedulati?"
CASO_NON_DEDUCIBILE = "Qual e' il costo medio di trasporto per container spedito?"
CASO_FUORI_DOMINIO = "Quali sono gli stipendi orari degli operai del magazzino?"


# 3. The two systems under comparison
def esegui_pipeline_completa(domanda: str) -> dict:
    """The full multi-agent system: Agent 1 (extraction) -> Agent 2 (generation)
    -> Agent 3 (execution + evaluation), with the revision loop."""
    return PIPELINE.invoke(crea_stato_iniziale(domanda))


def esegui_baseline_one_shot(domanda: str) -> dict:
    """
    Baseline: a SINGLE call to Agent 2's model, skipping Agent 1's extraction
    (the raw question is used directly as "extracted information") and
    Agent 3's evaluation/revision loop. The query is executed once and
    whatever comes back - or the execution error - is the final result.
    """
    schema = DB_LOGISTICS.get_table_info()
    query = genera_query(informazioni_estratte=domanda, schema_database=schema)

    print(f"[baseline] Esecuzione della query (timeout {QUERY_TIMEOUT_SECONDI}s)...", flush=True)
    try:
        risultati = DB_LOGISTICS.run(query)
        esito = "successo"
        messaggio_errore = None
    except Exception as errore:
        risultati = None
        esito = "errore"
        messaggio_errore = str(errore)
    print(f"[baseline] Esecuzione terminata: {esito}", flush=True)

    return \
    {
        "query_corrente": query,
        "risultati_query": risultati,
        "esito": esito,
        "messaggio_errore": messaggio_errore,
    }


# 4. Metrics
def _esegui_sql_verita(sql: str) -> set:
    """Runs a ground-truth SQL query on the converted OCEL SQLite file."""
    return set(_esegui_query_con_timeout(sql))


def _parse_risultati(testo) -> list:
    """Parses DB.run()'s raw string output into real Python data via ast.literal_eval."""
    if not testo:
        return []
    try:
        valore = ast.literal_eval(testo)
    except (ValueError, SyntaxError):
        return []
    return valore if isinstance(valore, list) else [valore]


def accuratezza_esecuzione(risultati_ottenuti, sql_verita: str) -> bool:
    """
    STRICT check (analogous to the paper's "Accuracy standard"): the rows
    returned by the system's query, compared as sets, must exactly match
    the rows returned by the ground-truth SQL.
    """
    attesi = _esegui_sql_verita(sql_verita)
    ottenuti = set(tuple(riga) if isinstance(riga, tuple) else (riga,)
                    for riga in _parse_risultati(risultati_ottenuti))
    return ottenuti == attesi


def risposta_accettabile(risultato: dict, sql_verita: str) -> bool:
    """
    LOOSER check (analogous to the paper's PC-Accuracy): accepts the answer
    if the exact result matches, OR if every expected value appears in the
    raw result or in the natural-language final answer. Credits answers that
    are right in substance but not minimal (e.g. a list of rows whose count
    is then stated correctly in the final answer).
    """
    if accuratezza_esecuzione(risultato.get("risultati_query"), sql_verita):
        return True

    attesi = _esegui_sql_verita(sql_verita)
    testo = f"{risultato.get('risultati_query')} {risultato.get('risposta_finale', '')}"
    return all(str(valore) in testo for riga in attesi for valore in riga)


def _ha_risposto(risultato: dict) -> bool:
    """
    For UNANSWERABLE questions: True if the system produced something that
    looks like an answer - it ended successfully AND the result contains at
    least one non-NULL value. A result like [(None,)] (e.g. AVG over a column
    that does not hold what was asked) is treated as "no answer".
    """
    if risultato.get("esito") != "successo":
        return False
    return any(valore is not None
               for riga in _parse_risultati(risultato.get("risultati_query"))
               for valore in (riga if isinstance(riga, tuple) else (riga,)))


def _dettaglio(risultato: dict) -> dict:
    """What happened inside a run, saved to the JSON so failures can be diagnosed."""
    risultati = risultato.get("risultati_query")
    return \
    {
        "esito": risultato.get("esito"),
        "classificazione": risultato.get("classificazione"),
        "informazioni_estratte": risultato.get("informazioni_estratte"),
        "query": risultato.get("query_corrente"),
        "tentativi": risultato.get("tentativi_miglioramento"),
        "feedback_valutazione": risultato.get("feedback_valutazione"),
        "risultati": None if risultati is None else str(risultati)[:300],
        "risposta_finale": risultato.get("risposta_finale"),
        "errore": risultato.get("messaggio_errore"),
    }


# 5. Computational footprint
# Parameters ACTIVE per token, where they differ from the plain parameter
# count: the Qwen3-Coder-30B-A3B model is a mixture-of-experts that uses
# ~3.3B of its 30.5B parameters per token. (Also needed because summing
# p.numel() on a 4-bit quantized model under-counts the packed weights.)
PARAMETRI_ATTIVI_NOTI = {"Qwen/Qwen3-Coder-30B-A3B-Instruct": 3.3e9}


def _modello_hf(chat_model):
    """Reaches into a LangChain ChatHuggingFace/HuggingFacePipeline to get the raw HF model."""
    return chat_model.llm.pipeline.model


def _parametri_attivi(modello_hf) -> float:
    nome = getattr(getattr(modello_hf, "config", None), "_name_or_path", "")
    if nome in PARAMETRI_ATTIVI_NOTI:
        return PARAMETRI_ATTIVI_NOTI[nome]
    return float(sum(p.numel() for p in modello_hf.parameters()))


class ContatoreFlops:
    """Wraps model.generate of each distinct model, accumulating estimated FLOPs."""

    def __init__(self, modelli_hf):
        self.totale = 0.0
        visti = set()
        for modello in modelli_hf:
            if id(modello) in visti:
                continue
            visti.add(id(modello))
            self._strumenta(modello)

    def _strumenta(self, modello):
        parametri = _parametri_attivi(modello)
        generate_originale = modello.generate

        def generate_contato(*args, **kwargs):
            uscita = generate_originale(*args, **kwargs)
            sequenze = getattr(uscita, "sequences", uscita)
            self.totale += 2.0 * parametri * sequenze.numel()
            return uscita

        modello.generate = generate_contato


CONTATORE_FLOPS = ContatoreFlops(
    [_modello_hf(LLM_AGENTE_1), _modello_hf(LLM_AGENTE_2), _modello_hf(LLM_AGENTE_3)]
)


def misura_footprint(funzione_da_misurare):
    """Runs funzione_da_misurare() once; returns (risultato, {secondi, flops_stimati})."""
    flops_prima = CONTATORE_FLOPS.totale
    inizio = time.perf_counter()
    risultato = funzione_da_misurare()
    durata = time.perf_counter() - inizio
    return risultato, {"secondi": durata, "flops_stimati": CONTATORE_FLOPS.totale - flops_prima}


# 6. Running the experiments
def esegui_caso_qualitativo(etichetta: str, domanda: str) -> None:
    print(f"\n{'=' * 70}\n{etichetta}: \"{domanda}\"\n{'=' * 70}")

    risultato_pipeline = esegui_pipeline_completa(domanda)
    print("\n[Pipeline completa]")
    print(f"  esito: {risultato_pipeline['esito']}")
    print(f"  classificazione: {risultato_pipeline['classificazione']}")
    print(f"  informazioni estratte: {risultato_pipeline['informazioni_estratte']}")
    print(f"  query: {risultato_pipeline['query_corrente']}")
    print(f"  tentativi: {risultato_pipeline['tentativi_miglioramento']}")
    print(f"  risposta: {risultato_pipeline.get('risposta_finale') or risultato_pipeline.get('messaggio_errore')}")

    risultato_baseline = esegui_baseline_one_shot(domanda)
    print("\n[Baseline one-shot]")
    print(f"  esito: {risultato_baseline['esito']}")
    print(f"  query: {risultato_baseline['query_corrente']}")
    print(f"  risultato/errore: {risultato_baseline.get('risultati_query') or risultato_baseline.get('messaggio_errore')}")


def esegui_benchmark_quantitativo() -> dict:
    risultati_pipeline, risultati_baseline = [], []

    for indice, (domanda, sql_verita) in enumerate(CASI_BENCHMARK, start=1):
        print(f"\n--- Domanda {indice}/{len(CASI_BENCHMARK)}: {domanda}", flush=True)

        esito_p, metriche_p = misura_footprint(lambda: esegui_pipeline_completa(domanda))
        esito_b, metriche_b = misura_footprint(lambda: esegui_baseline_one_shot(domanda))

        for lista, esito, metriche in ((risultati_pipeline, esito_p, metriche_p),
                                       (risultati_baseline, esito_b, metriche_b)):
            lista.append(
                {
                    "domanda": domanda,
                    "accuratezza_esecuzione": accuratezza_esecuzione(esito.get("risultati_query"), sql_verita),
                    "risposta_accettabile": risposta_accettabile(esito, sql_verita),
                    **metriche,
                    "dettaglio": _dettaglio(esito),
                }
            )

    return {"pipeline": risultati_pipeline, "baseline": risultati_baseline}


def esegui_benchmark_non_rispondibili() -> dict:
    risultati_pipeline, risultati_baseline = [], []

    for indice, domanda in enumerate(CASI_NON_RISPONDIBILI, start=1):
        print(f"\n--- Non rispondibile {indice}/{len(CASI_NON_RISPONDIBILI)}: {domanda}", flush=True)

        esito_p, metriche_p = misura_footprint(lambda: esegui_pipeline_completa(domanda))
        esito_b, metriche_b = misura_footprint(lambda: esegui_baseline_one_shot(domanda))

        for lista, esito, metriche in ((risultati_pipeline, esito_p, metriche_p),
                                       (risultati_baseline, esito_b, metriche_b)):
            lista.append(
                {
                    "domanda": domanda,
                    "ha_risposto": _ha_risposto(esito),
                    **metriche,
                    "dettaglio": _dettaglio(esito),
                }
            )

    return {"pipeline": risultati_pipeline, "baseline": risultati_baseline}


def riassumi(nome: str, righe: list) -> None:
    n = len(righe) or 1
    accuratezza = sum(r["accuratezza_esecuzione"] for r in righe) / n
    accettabile = sum(r["risposta_accettabile"] for r in righe) / n
    flops_medi = sum(r["flops_stimati"] for r in righe) / n
    secondi_medi = sum(r["secondi"] for r in righe) / n
    tentativi = [r["dettaglio"]["tentativi"] for r in righe if r["dettaglio"].get("tentativi") is not None]
    extra = f" | Tentativi medi: {sum(tentativi) / len(tentativi):.1f}" if tentativi else ""
    print(f"{nome:20s} | Accuratezza esecuzione: {accuratezza:.2f} | "
          f"Risposta accettabile: {accettabile:.2f} | "
          f"FLOPs medi: {flops_medi:.3e} | Tempo medio: {secondi_medi:.2f}s{extra}")


def riassumi_rifiuti(nome: str, righe: list) -> None:
    n = len(righe) or 1
    risposte = sum(r["ha_risposto"] for r in righe)
    print(f"{nome:20s} | Rifiuto corretto: {(n - risposte) / n:.2f} "
          f"({n - risposte}/{n}) | Risposte inventate: {risposte}/{n}")


if __name__ == "__main__":
    print("Schema del dataset OCEL (Logistics):")
    stampa_schema_ocel()

    esegui_caso_qualitativo("Fatto deducibile", CASO_FATTO_DEDUCIBILE)
    esegui_caso_qualitativo("Dato non derivabile (verifica anti-allucinazione)", CASO_NON_DEDUCIBILE)
    esegui_caso_qualitativo("Fuori dominio (adversarial check)", CASO_FUORI_DOMINIO)

    print(f"\n{'=' * 70}\nValutazione quantitativa su {len(CASI_BENCHMARK)} domande "
          f"(MAX_TENTATIVI={MAX_TENTATIVI})\n{'=' * 70}")
    esiti = esegui_benchmark_quantitativo()

    print(f"\n{'=' * 70}\nDomande NON rispondibili: {len(CASI_NON_RISPONDIBILI)}\n{'=' * 70}")
    rifiuti = esegui_benchmark_non_rispondibili()

    print("\n=== Domande con risposta ===")
    riassumi("Pipeline completa", esiti["pipeline"])
    riassumi("Baseline one-shot", esiti["baseline"])
    print("\n=== Domande senza risposta (il comportamento corretto e' NON rispondere) ===")
    riassumi_rifiuti("Pipeline completa", rifiuti["pipeline"])
    riassumi_rifiuti("Baseline one-shot", rifiuti["baseline"])

    with open("risultati_esperimenti.json", "w", encoding="utf-8") as f:
        json.dump({**esiti, "non_rispondibili": rifiuti}, f, indent=2, default=str, ensure_ascii=False)
    print("\nRisultati dettagliati salvati in risultati_esperimenti.json")