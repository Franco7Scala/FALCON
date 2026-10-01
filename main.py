# Main
from agenti.request_agent import estrai_informazioni
from agenti.query_agent import genera_query
from agenti.response_agent import esegui_e_rispondi
from orchestratore.orchestratore import PIPELINE, PipelineState
from utils.utils import DB


def demo_agente_1() -> None:
    """Runs Agent 1's two functions directly, outside the pipeline."""
    schema = DB.get_table_info()

    richieste_di_prova = \
    [
        "Mostrami i clienti che vanno di nome Mario",
        "Mostrami tutti i prodotti disponibili",
        "Che tempo fa oggi a Roma?",
    ]

    for richiesta in richieste_di_prova:
        classificazione, informazioni = estrai_informazioni(richiesta, schema)
        print(f"\nRichiesta: {richiesta}")
        print(f"Classificazione: {classificazione}")
        print(f"Informazioni estratte: {informazioni}")


def demo_agente_2() -> None:
    """Runs Agent 2's function directly, outside the pipeline."""
    schema = DB.get_table_info()
    informazioni = (
        "- entita': clienti\n"
        "- campi richiesti: nome, cognome\n"
        "- filtro: hanno un ordine con totale > 100"
    )

    query = genera_query(informazioni, schema)

    print(f"Informazioni estratte: {informazioni}")
    print(f"Query generata: {query}")


def demo_agente_3() -> None:
    """Runs Agent 3's function directly, outside the pipeline."""
    richiesta = "Mostrami i clienti che hanno un ordine con totale superiore a 100"
    query = (
        "SELECT c.nome, c.cognome FROM clienti c "
        "JOIN ordini o ON o.cliente_id = c.id "
        "WHERE o.totale > 100"
    )

    risultato = esegui_e_rispondi(richiesta, query)

    print(f"Risultati grezzi: {risultato['risultati_query']}")
    print(f"Risposta finale: {risultato['risposta_finale']}")


def avvia_pipeline_interattiva() -> None:
    """Runs the full pipeline (the only LangGraph graph) interactively from the terminal."""

    print("Pipeline pronta. Scrivi una richiesta (o 'exit' per uscire).")

    while True:
        richiesta = input("\n> ").strip()
        if richiesta.lower() in {"exit", "quit"}:
            break

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

        risultato = PIPELINE.invoke(stato_iniziale)

        if risultato["esito"] == "errore":
            print(f"\n{risultato['messaggio_errore']}")
        else:
            print(f"\n{risultato['risposta_finale']}")


if __name__ == "__main__":
    avvia_pipeline_interattiva()