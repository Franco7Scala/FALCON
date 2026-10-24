# Next-Step Prediction experiment on the OCEL 2.0
# "Logistics" log, with Container as the case notion.
# python -m experiments.next_step --sistemi maggioranza,oracolo,lstm
# python -m experiments.next_step --insieme dev --max-casi 5 --sistemi baseline,pipeline
# python -m experiments.next_step --insieme test
# add --facilitato for the variant whose container_trace table already
# carries the `prefix` and `next_activity` columns (no window functions).
import argparse
import json
import random
import sqlite3
import threading
import time
from pathlib import Path
from types import SimpleNamespace

try:
    from experiments import next_step_data as nd
except ImportError:
    import next_step_data as nd


QUERY_TIMEOUT_SECONDI = 60
BATCH_LSTM = 128
PARAMETRI_ATTIVI_NOTI = {"Qwen/Qwen3-Coder-30B-A3B-Instruct": 3.3e9}


# Shared infrastructure for the LLM systems (same ideas as experiments.py)
def _esegui_query_con_timeout(percorso_db, sql: str, timeout_secondi: int = QUERY_TIMEOUT_SECONDI) -> list:
    """Runs sql and returns all rows, really interrupting it after timeout_secondi."""
    connessione = sqlite3.connect(percorso_db, check_same_thread=False)
    timer = threading.Timer(timeout_secondi, connessione.interrupt)
    timer.start()
    try:
        return connessione.execute(sql).fetchall()
    finally:
        timer.cancel()
        connessione.close()


class DBConTimeout:
    """Drop-in replacement for the SQLDatabase the agents use as `DB`; run() has a hard timeout."""

    def __init__(self, db_reale, percorso_db, timeout_secondi: int = QUERY_TIMEOUT_SECONDI):
        self._db_reale = db_reale
        self._percorso_db = percorso_db
        self._timeout_secondi = timeout_secondi

    def run(self, command: str, *args, **kwargs) -> str:
        return str(_esegui_query_con_timeout(self._percorso_db, command, self._timeout_secondi))

    def __getattr__(self, nome):
        return getattr(self._db_reale, nome)


def _modello_hf(chat_model):
    return chat_model.llm.pipeline.model


def _parametri_attivi(modello_hf) -> float:
    nome = getattr(getattr(modello_hf, "config", None), "_name_or_path", "")
    if nome in PARAMETRI_ATTIVI_NOTI:
        return PARAMETRI_ATTIVI_NOTI[nome]
    return float(sum(p.numel() for p in modello_hf.parameters()))


class ContatoreFlops:
    """Wraps model.generate of each distinct model; FLOPs ~= 2 * active params * tokens processed."""

    def __init__(self, modelli_hf):
        self.totale = 0.0
        visti = set()
        for modello in modelli_hf:
            if id(modello) not in visti:
                visti.add(id(modello))
                self._strumenta(modello)

    def _strumenta(self, modello):
        parametri = _parametri_attivi(modello)
        generate_originale = modello.generate

        def generate_contato(*args, **kwargs):
            uscita = generate_originale(*args, **kwargs)
            self.totale += 2.0 * parametri * getattr(uscita, "sequences", uscita).numel()
            return uscita

        modello.generate = generate_contato


def prepara_sistemi_llm(percorso_db_training):
    """
    Imports the agents (loading the models), points every `DB` they use at
    the training database, and installs the FLOP counter. `from X import Y`
    binds the name inside the importing module, so reassigning utils.utils.DB
    would not work: the name is patched in each module that uses it.
    """
    from langchain_community.utilities import SQLDatabase
    import agenti.response_agent as agente_3_modulo
    import orchestratore.orchestratore as pipeline_modulo
    from orchestratore.orchestratore import PIPELINE, crea_stato_iniziale, MAX_TENTATIVI
    from agenti.query_agent import genera_query, LLM as llm_2
    from agenti.request_agent import LLM as llm_1
    from agenti.response_agent import LLM as llm_3

    db = DBConTimeout(SQLDatabase.from_uri(f"sqlite:///{percorso_db_training}"), percorso_db_training)
    agente_3_modulo.DB = db
    pipeline_modulo.DB = db

    return SimpleNamespace(
        db=db, PIPELINE=PIPELINE, crea_stato_iniziale=crea_stato_iniziale,
        genera_query=genera_query, MAX_TENTATIVI=MAX_TENTATIVI,
        contatore=ContatoreFlops([_modello_hf(llm_1), _modello_hf(llm_2), _modello_hf(llm_3)]),
    )


def esegui_pipeline(llm, domanda: str) -> dict:
    return llm.PIPELINE.invoke(llm.crea_stato_iniziale(domanda))


def esegui_baseline_one_shot(llm, domanda: str) -> dict:
    """One call to Agent 2's model: the raw question is used as the 'extracted information'."""
    query = llm.genera_query(informazioni_estratte=domanda, schema_database=llm.db.get_table_info())
    try:
        risultati, esito, errore = llm.db.run(query), "successo", None
    except Exception as eccezione:
        risultati, esito, errore = None, "errore", str(eccezione)
    return {"query_corrente": query, "risultati_query": risultati, "esito": esito, "messaggio_errore": errore}


# Running one system over the evaluation prefixes
def _dettaglio(risultato: dict) -> dict:
    risultati = risultato.get("risultati_query")
    return {
        "esito": risultato.get("esito"),
        "classificazione": risultato.get("classificazione"),
        "query": risultato.get("query_corrente"),
        "tentativi": risultato.get("tentativi_miglioramento"),
        "risultati": None if risultati is None else str(risultati)[:300],
        "errore": risultato.get("messaggio_errore"),
    }


def esegui_sistema_llm(nome, funzione, llm, casi, vocabolario) -> list:
    """
    Runs funzione(llm, question) for each held-out prefix. A failure on one
    case (out of memory, ...) is recorded as UNKNOWN and the run goes on.
    A pipeline run that did not END successfully (refusal, attempts
    exhausted) is UNKNOWN even if its state still holds the last rejected
    query's rows.
    """
    record = []
    for i, (caso, prefisso, target) in enumerate(casi, start=1):
        domanda = nd.costruisci_domanda(prefisso)
        flops_prima, inizio = llm.contatore.totale, time.perf_counter()
        try:
            risultato, errore = funzione(llm, domanda), None
        except Exception as eccezione:
            risultato, errore = {}, f"{type(eccezione).__name__}: {eccezione}"
        durata = time.perf_counter() - inizio

        if risultato.get("esito") == "successo":
            predetta = nd.estrai_attivita_predetta(risultato.get("risultati_query"), vocabolario)
        else:
            predetta = nd.ATTIVITA_SCONOSCIUTA
        dettaglio = _dettaglio(risultato)
        dettaglio["errore"] = dettaglio["errore"] or errore

        record.append({
            "caso": caso, "prefisso": prefisso, "target": target, "predetta": predetta,
            "secondi": durata, "flops": llm.contatore.totale - flops_prima, "dettaglio": dettaglio,
        })
        print(f"[{nome} {i}/{len(casi)}] len(prefisso)={len(prefisso)} target={target!r} "
              f"predetta={predetta!r} {'OK' if predetta == target else '--'} ({durata:.1f}s)", flush=True)
    return record


def esegui_sistema_semplice(nome, previsore, casi) -> list:
    """For the systems that need no model: previsore(prefisso) -> activity name or None."""
    record = []
    for caso, prefisso, target in casi:
        inizio = time.perf_counter()
        previsione = previsore(prefisso)
        durata = time.perf_counter() - inizio
        record.append({
            "caso": caso, "prefisso": prefisso, "target": target,
            "predetta": previsione or nd.ATTIVITA_SCONOSCIUTA,
            "secondi": durata, "flops": 0.0, "dettaglio": {},
        })
    return record


# The LSTM baseline (same architecture and optimiser as the paper's script)
def addestra_e_valuta_lstm(sequenze_training, vocabolario, casi, epoche: int, seed: int):
    """
    Trains on every (prefix -> next activity) pair of the training traces and
    predicts the held-out prefixes. Batched, with padded sequences packed so
    the LSTM reads the true last step of each prefix (the original script
    trained one sample at a time).

    FLOPs are an ANALYTICAL estimate: forward ~ 2 * recurrent params * tokens
    (+ the output layer once per sample), training ~ 3x forward (forward +
    backward), embedding lookups ignored. FLOPpy is also attempted, as in the
    original script, and its raw report is printed if it works.
    """
    import torch
    import torch.nn as nn
    from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

    random.seed(seed)
    torch.manual_seed(seed)
    act2idx = {a: i + 1 for i, a in enumerate(vocabolario)}      # 0 is the padding index
    idx2act = {i: a for a, i in act2idx.items()}

    class NextStepLSTM(nn.Module):
        def __init__(self, n_classi, embed_dim=32, hidden_dim=64):
            super().__init__()
            self.embedding = nn.Embedding(n_classi, embed_dim, padding_idx=0)
            self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, n_classi)

        def forward(self, x, lunghezze):
            impacchettato = pack_padded_sequence(self.embedding(x), lunghezze, batch_first=True, enforce_sorted=False)
            _, (nascosto, _) = self.lstm(impacchettato)
            return self.fc(nascosto[-1])

    campioni = nd.campioni_addestramento(sequenze_training)
    X = [torch.tensor([act2idx[a] for a in prefisso], dtype=torch.long) for prefisso, _ in campioni]
    y = torch.tensor([act2idx[t] for _, t in campioni], dtype=torch.long)

    modello = NextStepLSTM(len(vocabolario) + 1)
    ottimizzatore = torch.optim.Adam(modello.parameters(), lr=0.005)
    criterio = nn.CrossEntropyLoss()

    tracker = None
    try:
        from floppy import FLOPpyTracker
        tracker = FLOPpyTracker(run_name="lstm")
        tracker.run(model=modello, optimizer=ottimizzatore, criterion=criterio)
    except Exception as eccezione:
        print(f"[lstm] FLOPpy non utilizzabile ({type(eccezione).__name__}): uso la sola stima analitica")
        tracker = None

    inizio_addestramento = time.perf_counter()
    modello.train()
    for epoca in range(epoche):
        permutazione = torch.randperm(len(X)).tolist()
        perdita_totale = 0.0
        for inizio in range(0, len(X), BATCH_LSTM):
            indici = permutazione[inizio:inizio + BATCH_LSTM]
            batch = pad_sequence([X[i] for i in indici], batch_first=True)
            lunghezze = torch.tensor([len(X[i]) for i in indici], dtype=torch.long)
            ottimizzatore.zero_grad()
            perdita = criterio(modello(batch, lunghezze), y[indici])
            perdita.backward()
            ottimizzatore.step()
            perdita_totale += perdita.item() * len(indici)
        print(f"[lstm] epoca {epoca + 1}/{epoche} | perdita {perdita_totale / len(X):.4f}", flush=True)
    secondi_addestramento = time.perf_counter() - inizio_addestramento

    if tracker is not None:
        try:
            print("[lstm] report FLOPpy:", tracker.report())
        except Exception as eccezione:
            print(f"[lstm] report FLOPpy non disponibile ({type(eccezione).__name__})")

    parametri_ricorrenti = sum(p.numel() for p in modello.lstm.parameters())
    parametri_uscita = sum(p.numel() for p in modello.fc.parameters())
    token_totali = sum(len(prefisso) for prefisso, _ in campioni)
    flops_addestramento = epoche * 3 * (2 * parametri_ricorrenti * token_totali + 2 * parametri_uscita * len(campioni))

    modello.eval()
    record = []
    with torch.no_grad():
        for caso, prefisso, target in casi:
            inizio = time.perf_counter()
            x = torch.tensor([[act2idx[a] for a in prefisso]], dtype=torch.long)
            logit = modello(x, torch.tensor([len(prefisso)], dtype=torch.long))[0]
            logit[0] = float("-inf")                                  # never predict the padding class
            predetta = idx2act[int(logit.argmax())]
            record.append({
                "caso": caso, "prefisso": prefisso, "target": target, "predetta": predetta,
                "secondi": time.perf_counter() - inizio,
                "flops": 2 * parametri_ricorrenti * len(prefisso) + 2 * parametri_uscita, "dettaglio": {},
            })

    costo = {"secondi_addestramento": secondi_addestramento, "flops_addestramento_stimati": flops_addestramento,
             "campioni_addestramento": len(campioni), "epoche": epoche,
             "flops_addestramento_floppy": "vedi report stampato" if tracker is not None else None}
    return record, costo


# Metrics and reporting
def calcola_metriche(record, sequenze_training) -> dict:
    target = [r["target"] for r in record]
    predette = [r["predetta"] for r in record]
    prefissi = [r["prefisso"] for r in record]
    n = len(record) or 1
    return {
        "accuracy": nd.accuracy(target, predette),
        "macro_f1": nd.macro_f1(target, predette),
        "pc_accuracy": nd.pc_accuracy(prefissi, predette, sequenze_training),
        "senza_previsione": sum(p == nd.ATTIVITA_SCONOSCIUTA for p in predette),
        "flops_medi": sum(r["flops"] for r in record) / n,
        "secondi_medi": sum(r["secondi"] for r in record) / n,
    }


def stampa_tabella(metriche_per_sistema: dict, n_casi: int) -> None:
    print(f"\n{'Sistema':<13} | {'Accuracy':>8} | {'Macro-F1':>8} | {'PC-Acc.':>7} | "
          f"{'Senza prev.':>11} | {'FLOPs/prefisso':>14} | {'Tempo/prefisso':>14}")
    print("-" * 96)
    for nome, m in metriche_per_sistema.items():
        print(f"{nome:<13} | {m['accuracy']:>8.2f} | {m['macro_f1']:>8.2f} | {m['pc_accuracy']:>7.2f} | "
              f"{m['senza_previsione']:>5}/{n_casi:<5} | {m['flops_medi']:>14.3e} | {m['secondi_medi']:>13.2f}s")


def parse_argomenti(argv=None):
    base = Path(__file__).resolve().parent.parent / "databases"
    p = argparse.ArgumentParser(description="Next-Step Prediction sul log OCEL Logistics (caso = Container)")
    p.add_argument("--insieme", choices=["dev", "test"], default="dev",
                   help="insieme di valutazione (default: dev; 'test' va usato UNA volta, per i numeri finali)")
    p.add_argument("--sistemi", default="maggioranza,oracolo,lstm,baseline,pipeline",
                   help="sistemi da eseguire, separati da virgola")
    p.add_argument("--n-dev", type=int, default=20)
    p.add_argument("--n-test", type=int, default=100)
    p.add_argument("--max-casi", type=int, default=None, help="usa solo i primi N casi dell'insieme (prove rapide)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epoche", type=int, default=5, help="epoche di addestramento dell'LSTM")
    p.add_argument("--facilitato", action="store_true",
                   help="container_trace con le colonne prefix e next_activity precalcolate")
    p.add_argument("--db", type=Path, default=base / "container_logistics.sqlite")
    p.add_argument("--db-training", type=Path, default=base / "container_logistics_training.sqlite")
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args(argv)


SISTEMI_VALIDI = ["maggioranza", "oracolo", "lstm", "baseline", "pipeline"]


def main(argv=None) -> None:
    args = parse_argomenti(argv)
    sistemi = [s.strip() for s in args.sistemi.split(",") if s.strip()]
    sconosciuti = [s for s in sistemi if s not in SISTEMI_VALIDI]
    if sconosciuti:
        raise SystemExit(f"Sistemi non validi: {sconosciuti}. Validi: {SISTEMI_VALIDI}")
    if args.insieme == "test":
        print("*** INSIEME DI TEST: usalo una sola volta, dopo aver fissato prompt e parametri su 'dev' ***\n")

    if not args.db.exists():
        raise FileNotFoundError(f"'{args.db}' non esiste: converti prima il dataset OCEL (vedi experiments.py).")

    conn = sqlite3.connect(args.db)
    tracce = nd.carica_tracce(conn)
    conn.close()
    stat = nd.statistiche(tracce)
    print("Tracce dei container:", stat)

    casi_dev, casi_test, id_training = nd.crea_split(tracce, args.n_dev, args.n_test, args.seed)
    casi = (casi_dev if args.insieme == "dev" else casi_test)[: args.max_casi]
    esclusi = [c for c, _, _ in casi_dev + casi_test]
    sequenze_training = [tracce[c] for c in id_training]
    vocabolario = sorted({a for t in tracce.values() for a in t})
    print(f"Training: {len(id_training)} container | valutazione ({args.insieme}): {len(casi)} prefissi | "
          f"{len(esclusi)} container esclusi dal database di training | attivita': {len(vocabolario)}")

    print(f"Creo il database di training (senza i container di dev e test): {args.db_training}")
    nd.crea_db_training(args.db, args.db_training, esclusi, args.facilitato)

    record_per_sistema, metriche, costo_lstm = {}, {}, None

    if "maggioranza" in sistemi:
        fisso = nd.successore_piu_frequente(sequenze_training)
        record_per_sistema["maggioranza"] = esegui_sistema_semplice("maggioranza", lambda p: fisso, casi)

    if "oracolo" in sistemi:
        conn_training = sqlite3.connect(args.db_training)
        record_per_sistema["oracolo"] = esegui_sistema_semplice(
            "oracolo", lambda p: nd.previsione_oracolo(conn_training, p, args.facilitato), casi)
        conn_training.close()

    if "lstm" in sistemi:
        record_per_sistema["lstm"], costo_lstm = addestra_e_valuta_lstm(
            sequenze_training, vocabolario, casi, args.epoche, args.seed)

    llm_richiesti = [s for s in ("baseline", "pipeline") if s in sistemi]
    if llm_richiesti:
        llm = prepara_sistemi_llm(args.db_training)
        if "baseline" in sistemi:
            record_per_sistema["baseline"] = esegui_sistema_llm(
                "baseline", esegui_baseline_one_shot, llm, casi, vocabolario)
        if "pipeline" in sistemi:
            record_per_sistema["pipeline"] = esegui_sistema_llm(
                "pipeline", esegui_pipeline, llm, casi, vocabolario)

    for nome in SISTEMI_VALIDI:
        if nome in record_per_sistema:
            metriche[nome] = calcola_metriche(record_per_sistema[nome], sequenze_training)

    print(f"\n=== Next-Step Prediction | insieme {args.insieme} | {len(casi)} prefissi | "
          f"{'schema facilitato' if args.facilitato else 'schema base'} ===")
    stampa_tabella(metriche, len(casi))
    if "oracolo" in metriche:
        copertura = 1 - metriche["oracolo"]["senza_previsione"] / (len(casi) or 1)
        print(f"\nCopertura dell'oracolo: {copertura:.0%} dei prefissi compare nello storico di training "
              "(per gli altri nessun sistema basato su frequenze esatte puo' rispondere).")
    if costo_lstm:
        print(f"LSTM, addestramento: ~{costo_lstm['flops_addestramento_stimati']:.3e} FLOPs (stima analitica), "
              f"{costo_lstm['secondi_addestramento']:.1f}s su {costo_lstm['campioni_addestramento']} campioni.")
    if "pipeline" in metriche or "baseline" in metriche:
        print("I FLOPs dei sistemi LLM sono sommati su tutte le chiamate a generate() dei tre modelli.")

    output = args.output or Path(f"risultati_next_step_{args.insieme}{'_facilitato' if args.facilitato else ''}.json")
    with open(output, "w", encoding="utf-8") as f:
        json.dump({
            "configurazione": {k: str(v) for k, v in vars(args).items()},
            "statistiche_tracce": stat, "costo_lstm": costo_lstm, "metriche": metriche,
            "casi": record_per_sistema,
        }, f, indent=2, default=str, ensure_ascii=False)
    print(f"\nRisultati dettagliati salvati in {output}")


if __name__ == "__main__":
    main()