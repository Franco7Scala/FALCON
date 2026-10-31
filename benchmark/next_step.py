# Next-Step Prediction experiment on the OCEL 2.0
import argparse
import json
import random
import sqlite3
import threading
import time
from pathlib import Path
from types import SimpleNamespace

try:
    from benchmark import next_step_data as nd
except ImportError:
    import benchmark.next_step_data as nd


QUERY_TIMEOUT_SECONDS = 60
BATCH_LSTM = 128
KNOWN_ACTIVE_PARAMETERS = {"Qwen/Qwen3-Coder-30B-A3B-Instruct": 3.3e9}


# Shared infrastructure for the LLM systems (same ideas as experiments.py)
def _execute_query_with_timeout(database_path, sql: str, timeout_seconds: int = QUERY_TIMEOUT_SECONDS) -> list:
    """Runs sql and returns all rows, really interrupting it after timeout_secondi."""
    connessione = sqlite3.connect(database_path, check_same_thread=False)
    timer = threading.Timer(timeout_seconds, connessione.interrupt)
    timer.start()
    try:
        return connessione.execute(sql).fetchall()
    finally:
        timer.cancel()
        connessione.close()


class DatabaseWithTimeout:
    """Drop-in replacement for the SQLDatabase the agents use as `DB`; run() has a hard timeout."""

    def __init__(self, db_reale, database_path, timeout_seconds: int = QUERY_TIMEOUT_SECONDS):
        self._db_reale = db_reale
        self._percorso_db = database_path
        self._timeout_secondi = timeout_seconds

    def run(self, command: str, *args, **kwargs) -> str:
        return str(_execute_query_with_timeout(self._percorso_db, command, self._timeout_secondi))

    def __getattr__(self, name):
        return getattr(self._db_reale, name)


def _get_hf_model(chat_model):
    return chat_model.llm.pipeline.model


def _get_active_parameters(hf_model) -> float:
    name = getattr(getattr(hf_model, "config", None), "_name_or_path", "")
    if name in KNOWN_ACTIVE_PARAMETERS:
        return KNOWN_ACTIVE_PARAMETERS[name]
    return float(sum(p.numel() for p in hf_model.parameters()))


class FlopsCounter:
    """Wraps model.generate of each distinct model; FLOPs ~= 2 * active params * tokens processed."""

    def __init__(self, hf_models):
        self.totale = 0.0
        visti = set()
        for model in hf_models:
            if id(model) not in visti:
                visti.add(id(model))
                self._instrument_model(model)

    def _instrument_model(self, model):
        parameters = _get_active_parameters(model)
        generate_originale = model.generate

        def tracked_generate(*args, **kwargs):
            uscita = generate_originale(*args, **kwargs)
            self.totale += 2.0 * parameters * getattr(uscita, "sequences", uscita).numel()
            return uscita

        model.generate = tracked_generate


def prepare_llm_systems(training_database_path):
    """
    Imports the agents (loading the models), points every `DB` they use at
    the training database, and installs the FLOP counter. `from X import Y`
    binds the name inside the importing module, so reassigning utils.utils.DB
    would not work: the name is patched in each module that uses it.
    """
    from langchain_community.utilities import SQLDatabase
    import agenti.response_agent as agente_3_modulo
    import orchestratore.orchestratore as pipeline_modulo
    from orchestratore.orchestratore import PIPELINE, create_initial_state, MAX_ATTEMPTS
    from agenti.query_agent import generate_query, LLM as llm_2
    from agenti.request_agent import LLM as llm_1
    from agenti.response_agent import LLM as llm_3

    db = DatabaseWithTimeout(SQLDatabase.from_uri(f"sqlite:///{training_database_path}"), training_database_path)
    agente_3_modulo.DB = db
    pipeline_modulo.DB = db

    return SimpleNamespace(
        db=db, PIPELINE=PIPELINE, create_initial_state=create_initial_state,
        generate_query=generate_query, MAX_ATTEMPTS=MAX_ATTEMPTS,
        contatore=FlopsCounter([_get_hf_model(llm_1), _get_hf_model(llm_2), _get_hf_model(llm_3)]),
    )


def run_pipeline(llm, question: str) -> dict:
    return llm.PIPELINE.invoke(llm.create_initial_state(question))


def run_one_shot_baseline(llm, question: str) -> dict:
    """One call to Agent 2's model: the raw question is used as the 'extracted information'."""
    query = llm.generate_query(extracted_information=question, database_schema=llm.db.get_table_info())
    try:
        results, outcome, error = llm.db.run(query), "SUCCESS", None
    except Exception as exception:
        results, outcome, error = None, "ERROR", str(exception)
    return {"current_query": query, "query_results": results, "outcome": outcome, "error_message": error}


# Running one system over the evaluation prefixes
def _get_details(result: dict) -> dict:
    results = result.get("query_results")
    return {
        "outcome": result.get("outcome"),
        "classification": result.get("classification"),
        "query": result.get("current_query"),
        "attempts": result.get("improvement_attempts"),
        "risultati": None if results is None else str(results)[:300],
        "error": result.get("error_message"),
    }


def run_llm_system(name, callable_fn, llm, cases, vocabulary) -> list:
    """
    Runs callable_fn(llm, question) for each held-out prefix. A failure on one
    case (out of memory, ...) is recorded as UNKNOWN and the run goes on.
    A pipeline run that did not END successfully (refusal, attempts
    exhausted) is UNKNOWN even if its state still holds the last rejected
    query's rows.
    """
    record = []
    for i, (case, prefix, target) in enumerate(cases, start=1):
        question = nd.build_question(prefix)
        flops_before, inizio = llm.contatore.totale, time.perf_counter()
        try:
            result, error = callable_fn(llm, question), None
        except Exception as exception:
            result, error = {}, f"{type(exception).__name__}: {exception}"
        duration = time.perf_counter() - inizio

        if result.get("outcome") == "SUCCESS":
            predicted_activity = nd.extract_predicted_activity(result.get("query_results"), vocabulary)
        else:
            predicted_activity = nd.UNKNOWN_ACTIVITY
        detail = _get_details(result)
        detail["error"] = detail["error"] or error

        record.append({
            "caso": case, "prefisso": prefix, "target": target, "predetta": predicted_activity,
            "secondi": duration, "flops": llm.contatore.totale - flops_before, "dettaglio": detail,
        })
        print(f"[{name} {i}/{len(cases)}] len(prefisso)={len(prefix)} target={target!r} "
              f"predetta={predicted_activity!r} {'OK' if predicted_activity == target else '--'} ({duration:.1f}s)", flush=True)
    return record


def run_simple_system(name, previsore, cases) -> list:
    """For the systems that need no model: previsore(prefisso) -> activity name or None."""
    record = []
    for case, prefix, target in cases:
        inizio = time.perf_counter()
        previsione = previsore(prefix)
        duration = time.perf_counter() - inizio
        record.append({
            "caso": case, "prefisso": prefix, "target": target,
            "predetta": previsione or nd.UNKNOWN_ACTIVITY,
            "secondi": duration, "flops": 0.0, "dettaglio": {},
        })
    return record


# The LSTM baseline (same architecture and optimiser as the paper's script)
def train_and_evaluate_lstm(training_sequences, vocabulary, cases, epochs: int, seed: int):
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
    act2idx = {a: i + 1 for i, a in enumerate(vocabulary)}
    idx2act = {i: a for a, i in act2idx.items()}

    class NextStepLSTM(nn.Module):
        def __init__(self, n_classes, embed_dim=32, hidden_dim=64):
            super().__init__()
            self.embedding = nn.Embedding(n_classes, embed_dim, padding_idx=0)
            self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, n_classes)

        def forward(self, x, lengths):
            impacchettato = pack_padded_sequence(self.embedding(x), lengths, batch_first=True, enforce_sorted=False)
            _, (nascosto, _) = self.lstm(impacchettato)
            return self.fc(nascosto[-1])

    campioni = nd.training_samples(training_sequences)
    X = [torch.tensor([act2idx[a] for a in prefix], dtype=torch.long) for prefix, _ in campioni]
    y = torch.tensor([act2idx[t] for _, t in campioni], dtype=torch.long)

    model = NextStepLSTM(len(vocabulary) + 1)
    ottimizzatore = torch.optim.Adam(model.parameters(), lr=0.005)
    criterio = nn.CrossEntropyLoss()

    tracker = None
    try:
        from floppy import FLOPpyTracker
        tracker = FLOPpyTracker(run_name="lstm")
        tracker.run(model=model, optimizer=ottimizzatore, criterion=criterio)
    except Exception as exception:
        print(f"[lstm] FLOPpy unavailable ({type(exception).__name__}); using the analytical estimate only")
        tracker = None

    training_start = time.perf_counter()
    model.train()
    for epoca in range(epochs):
        permutazione = torch.randperm(len(X)).tolist()
        perdita_totale = 0.0
        for inizio in range(0, len(X), BATCH_LSTM):
            indici = permutazione[inizio:inizio + BATCH_LSTM]
            batch = pad_sequence([X[i] for i in indici], batch_first=True)
            lengths = torch.tensor([len(X[i]) for i in indici], dtype=torch.long)
            ottimizzatore.zero_grad()
            perdita = criterio(model(batch, lengths), y[indici])
            perdita.backward()
            ottimizzatore.step()
            perdita_totale += perdita.item() * len(indici)
        print(f"[lstm] epoch {epoca + 1}/{epochs} | loss {perdita_totale / len(X):.4f}", flush=True)
    training_seconds = time.perf_counter() - training_start

    if tracker is not None:
        try:
            print("[lstm] report FLOPpy:", tracker.report())
        except Exception as exception:
            print(f"[lstm] FLOPpy report unavailable ({type(exception).__name__})")

    recurring_parameters = sum(p.numel() for p in model.lstm.parameters())
    output_parameters = sum(p.numel() for p in model.fc.parameters())
    token_totali = sum(len(prefix) for prefix, _ in campioni)
    training_flops = epochs * 3 * (2 * recurring_parameters * token_totali + 2 * output_parameters * len(campioni))

    model.eval()
    record = []
    with torch.no_grad():
        for case, prefix, target in cases:
            inizio = time.perf_counter()
            x = torch.tensor([[act2idx[a] for a in prefix]], dtype=torch.long)
            logit = model(x, torch.tensor([len(prefix)], dtype=torch.long))[0]
            logit[0] = float("-inf")                                  # never predict the padding class
            predicted_activity = idx2act[int(logit.argmax())]
            record.append({
                "caso": case, "prefisso": prefix, "target": target, "predetta": predicted_activity,
                "secondi": time.perf_counter() - inizio,
                "flops": 2 * recurring_parameters * len(prefix) + 2 * output_parameters, "dettaglio": {},
            })

    costo = {"secondi_addestramento": training_seconds, "flops_addestramento_stimati": training_flops,
             "campioni_addestramento": len(campioni), "epoche": epochs,
             "flops_addestramento_floppy": "see printed report" if tracker is not None else None}
    return record, costo


# Metrics and reporting
def calculate_metrics(record, training_sequences) -> dict:
    target = [r["target"] for r in record]
    predictions = [r["predetta"] for r in record]
    prefixes = [r["prefisso"] for r in record]
    n = len(record) or 1
    return {
        "accuracy": nd.accuracy(target, predictions),
        "macro_f1": nd.macro_f1(target, predictions),
        "pc_accuracy": nd.prefix_consistency_accuracy(prefixes, predictions, training_sequences),
        "senza_previsione": sum(p == nd.UNKNOWN_ACTIVITY for p in predictions),
        "flops_medi": sum(r["flops"] for r in record) / n,
        "secondi_medi": sum(r["secondi"] for r in record) / n,
    }


def print_table(metrics_by_system: dict, n_cases: int) -> None:
    print(f"\n{'System':<13} | {'Accuracy':>8} | {'Macro-F1':>8} | {'PC-Acc.':>7} | "
          f"{'No prediction':>13} | {'FLOPs/prefix':>14} | {'Time/prefix':>14}")
    print("-" * 96)
    for name, m in metrics_by_system.items():
        print(f"{name:<13} | {m['accuracy']:>8.2f} | {m['macro_f1']:>8.2f} | {m['pc_accuracy']:>7.2f} | "
              f"{m['senza_previsione']:>5}/{n_cases:<5} | {m['flops_medi']:>14.3e} | {m['secondi_medi']:>13.2f}s")


def parse_arguments(argv=None):
    base = Path(__file__).resolve().parent.parent / "databases"
    p = argparse.ArgumentParser(description="Next-Step Prediction on the OCEL Logistics log (case = Container)")
    p.add_argument("--insieme", choices=["dev", "test"], default="dev",
                   help="evaluation split (default: dev; use 'test' ONCE for final metrics)")
    p.add_argument("--sistemi", default="maggioranza,oracolo,lstm,baseline,pipeline",
                   help="sistemi da eseguire, separati da virgola")
    p.add_argument("--n-dev", type=int, default=20)
    p.add_argument("--n-test", type=int, default=100)
    p.add_argument("--max-casi", type=int, default=None, help="use only the first N cases in the split (quick runs)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epoche", type=int, default=5, help="number of LSTM training epochs")
    p.add_argument("--facilitato", action="store_true",
                   help="use container_trace with precomputed prefix and next_activity columns")
    p.add_argument("--db", type=Path, default=base / "container_logistics.sqlite")
    p.add_argument("--db-training", type=Path, default=base / "container_logistics_training.sqlite")
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args(argv)


VALID_SYSTEMS = ["maggioranza", "oracolo", "lstm", "baseline", "pipeline"]


def main(argv=None) -> None:
    args = parse_arguments(argv)
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    sconosciuti = [s for s in systems if s not in VALID_SYSTEMS]
    if sconosciuti:
        raise SystemExit(f"Invalid systems: {sconosciuti}. Valid systems: {VALID_SYSTEMS}")
    if args.insieme == "test":
        print("*** TEST SPLIT: use it once, after finalizing prompts and parameters on 'dev' ***\n")

    if not args.db.exists():
        raise FileNotFoundError(f"'{args.db}' does not exist: convert the OCEL dataset first (see experiments.py).")

    conn = sqlite3.connect(args.db)
    traces = nd.load_traces(conn)
    conn.close()
    stat = nd.trace_statistics(traces)
    print("Container traces:", stat)

    dev_cases, test_cases, training_ids = nd.create_split(traces, args.n_dev, args.n_test, args.seed)
    cases = (dev_cases if args.insieme == "dev" else test_cases)[: args.max_casi]
    esclusi = [c for c, _, _ in dev_cases + test_cases]
    training_sequences = [traces[c] for c in training_ids]
    vocabulary = sorted({a for t in traces.values() for a in t})
    print(f"Training: {len(training_ids)} containers | evaluation ({args.insieme}): {len(cases)} prefixes | "
          f"{len(esclusi)} containers excluded from the training database | activities: {len(vocabulary)}")

    print(f"Creating the training database (excluding dev and test containers): {args.training_database}")
    nd.create_training_database(args.db, args.training_database, esclusi, args.simplified)

    records_by_system, metrics, costo_lstm = {}, {}, None

    if "maggioranza" in systems:
        fisso = nd.most_frequent_successor(training_sequences)
        records_by_system["maggioranza"] = run_simple_system("maggioranza", lambda p: fisso, cases)

    if "oracolo" in systems:
        conn_training = sqlite3.connect(args.training_database)
        records_by_system["oracolo"] = run_simple_system(
            "oracolo", lambda p: nd.oracle_prediction(conn_training, p, args.simplified), cases)
        conn_training.close()

    if "lstm" in systems:
        records_by_system["lstm"], costo_lstm = train_and_evaluate_lstm(
            training_sequences, vocabulary, cases, args.epochs, args.seed)

    required_llms = [s for s in ("baseline", "pipeline") if s in systems]
    if required_llms:
        llm = prepare_llm_systems(args.training_database)
        if "baseline" in systems:
            records_by_system["baseline"] = run_llm_system(
                "baseline", run_one_shot_baseline, llm, cases, vocabulary)
        if "pipeline" in systems:
            records_by_system["pipeline"] = run_llm_system(
                "pipeline", run_pipeline, llm, cases, vocabulary)

    for name in VALID_SYSTEMS:
        if name in records_by_system:
            metrics[name] = calculate_metrics(records_by_system[name], training_sequences)

    print(f"\n=== Next-Step Prediction | split {args.insieme} | {len(cases)} prefixes | "
          f"{'simplified schema' if args.simplified else 'base schema'} ===")
    print_table(metrics, len(cases))
    if "oracolo" in metrics:
        copertura = 1 - metrics["oracolo"]["senza_previsione"] / (len(cases) or 1)
        print(f"\nOracle coverage: {copertura:.0%} of prefixes occur in the training history "
              "(no exact-frequency system can answer for the others).")
    if costo_lstm:
        print(f"LSTM training: ~{costo_lstm['flops_addestramento_stimati']:.3e} FLOPs (analytical estimate), "
              f"{costo_lstm['secondi_addestramento']:.1f}s on {costo_lstm['campioni_addestramento']} samples.")
    if "pipeline" in metrics or "baseline" in metrics:
        print("LLM system FLOPs are summed across all generate() calls from the three models.")

    output = args.output or Path(f"risultati_next_step_{args.insieme}{'_facilitato' if args.simplified else ''}.json")
    with open(output, "w", encoding="utf-8") as f:
        json.dump({
            "configurazione": {k: str(v) for k, v in vars(args).items()},
            "statistiche_tracce": stat, "costo_lstm": costo_lstm, "metriche": metrics,
            "casi": records_by_system,
        }, f, indent=2, default=str, ensure_ascii=False)
    print(f"\nDetailed results saved to {output}")


if __name__ == "__main__":
    main()
