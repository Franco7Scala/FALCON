# Orchestrator - chains Agent 1, Agent 2 and Agent 3 into a single pipeline
from request_agent import AGENTE_1, Agent1State
from query_agent import AGENTE_2, Agent2State
from response_agent import AGENTE_3, Agent3State
from utils import DB

def elabora_richiesta_utente(richiesta: str, verbose: bool = False) -> str:
    """
    Runs the full pipeline (Agent 1 -> Agent 2 -> Agent 3) for a single user
    request and returns the final natural-language answer to show the user.
    """

    # Agent 1 - extracts (and validates) the relevant information
    stato_agente_1: Agent1State = \
    {
        "richiesta_utente": richiesta,
        "informazioni_estratte": None,
        "tentativi_estrazione": 0,
        "classificazione": None,
        "valutazione": None,
        "feedback_valutazione": None,
        "esito": None,
        "informazioni_finali": None,
        "estrazione_forzata": False,
        "messaggio_errore": None,
    }

    risultato_1 = AGENTE_1.invoke(stato_agente_1)

    if verbose:
        print(f"[Agente 1] esito={risultato_1['esito']} "
              f"tentativi={risultato_1['tentativi_estrazione']} "
              f"forzata={risultato_1['estrazione_forzata']}")

    if risultato_1["esito"] == "errore":
        return risultato_1["messaggio_errore"]

    # Agent 2 - generates (and validates) the SQL query
    stato_agente_2: Agent2State = \
    {
        "informazioni_estratte": risultato_1["informazioni_finali"],
        "schema_database": DB.get_table_info(),
        "query_corrente": None,
        "valutazione": None,
        "feedback_valutazione": None,
        "tentativi_miglioramento": 0,
        "esito": None,
        "query_finale": None,
        "accettata_forzatamente": False,
        "messaggio_errore": None,
    }

    risultato_2 = AGENTE_2.invoke(stato_agente_2)

    if verbose:
        print(f"[Agente 2] tentativi={risultato_2['tentativi_miglioramento']} "
              f"forzata={risultato_2['accettata_forzatamente']} "
              f"query={risultato_2['query_finale']}")

    # Agent 3 - executes the query and produces the final answer
    stato_agente_3: Agent3State = \
    {
        "informazioni_estratte": risultato_1["informazioni_finali"],
        "query_finale": risultato_2["query_finale"],
        "risultati_query": None,
        "esito": None,
        "risposta_finale": None,
        "messaggio_errore": None,
    }

    risultato_3 = AGENTE_3.invoke(stato_agente_3)

    if verbose and risultato_3["esito"] == "errore":
        print(f"[Agente 3] errore: {risultato_3['messaggio_errore']}")

    return risultato_3["risposta_finale"]


if __name__ == "__main__":

    print("Scrivi una richiesta (o 'exit' per uscire).")

    while True:
        richiesta = input("\n> ").strip()
        if richiesta.lower() in {"exit", "quit"}:
            break

        risposta = elabora_richiesta_utente(richiesta, verbose=True)
        print(f"\n{risposta}")