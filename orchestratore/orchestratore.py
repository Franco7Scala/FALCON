# Pipeline
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from agenti.request_agent import estrai_informazioni, valuta_query
from agenti.query_agent import genera_query
from agenti.response_agent import esegui_e_rispondi


MAX_TENTATIVI = 6  # Maximum number of query generation/revision attempts


# Shared pipeline state - the only state in the whole project
class PipelineState(TypedDict):
    richiesta_utente: str
    schema_database: str
    informazioni_estratte: Optional[str]
    query_corrente: Optional[str]
    feedback_valutazione: Optional[str]
    valutazione: Optional[str]  # "CORRETTA" | "ERRATA"
    tentativi: int
    risultati_query: Optional[str]
    risposta_finale: Optional[str]
    esito: Optional[str]  # "successo" | "errore"
    messaggio_errore: Optional[str]


# Node "Agente 1 - Estrazione"
def nodo_estrai_informazioni(state: PipelineState) -> PipelineState:

    classificazione, informazioni = estrai_informazioni(
        state["richiesta_utente"], state["schema_database"]
    )

    if "FUORI_CONTESTO" in classificazione:
        return \
        {
            "richiesta_utente": state["richiesta_utente"],
            "schema_database": state["schema_database"],
            "informazioni_estratte": None,
            "query_corrente": state["query_corrente"],
            "feedback_valutazione": state["feedback_valutazione"],
            "valutazione": state["valutazione"],
            "tentativi": state["tentativi"],
            "risultati_query": state["risultati_query"],
            "risposta_finale": state["risposta_finale"],
            "esito": "errore",
            "messaggio_errore": "Interrogazione non chiara o non valida. Potresti essere piu' preciso?",
        }

    return \
    {
        "richiesta_utente": state["richiesta_utente"],
        "schema_database": state["schema_database"],
        "informazioni_estratte": informazioni,
        "query_corrente": state["query_corrente"],
        "feedback_valutazione": state["feedback_valutazione"],
        "valutazione": state["valutazione"],
        "tentativi": state["tentativi"],
        "risultati_query": state["risultati_query"],
        "risposta_finale": state["risposta_finale"],
        "esito": None,
        "messaggio_errore": None,
    }


def route_dopo_estrazione(state: PipelineState) -> str:
    if "errore" in (state["esito"] or ""):
        return "fuori_contesto"
    return "in_contesto"


# Node "Agente 2 - Generazione/Correzione query"
def nodo_genera_query(state: PipelineState) -> PipelineState:

    nuova_query = genera_query(
        informazioni_estratte=state["informazioni_estratte"],
        schema_database=state["schema_database"],
        query_precedente=state["query_corrente"],
        feedback=state["feedback_valutazione"],
    )

    return \
    {
        "richiesta_utente": state["richiesta_utente"],
        "schema_database": state["schema_database"],
        "informazioni_estratte": state["informazioni_estratte"],
        "query_corrente": nuova_query,
        "feedback_valutazione": state["feedback_valutazione"],
        "valutazione": state["valutazione"],
        "tentativi": state["tentativi"] + 1,
        "risultati_query": state["risultati_query"],
        "risposta_finale": state["risposta_finale"],
        "esito": state["esito"],
        "messaggio_errore": state["messaggio_errore"],
    }


# Node "Agente 1 - Valutazione query"
def nodo_valuta_query(state: PipelineState) -> PipelineState:

    valutazione, feedback, informazioni_aggiornate = valuta_query(
        richiesta_utente=state["richiesta_utente"],
        informazioni_estratte=state["informazioni_estratte"],
        query_corrente=state["query_corrente"],
        schema_database=state["schema_database"],
    )

    return \
    {
        "richiesta_utente": state["richiesta_utente"],
        "schema_database": state["schema_database"],
        "informazioni_estratte": informazioni_aggiornate or state["informazioni_estratte"],
        "query_corrente": state["query_corrente"],
        "feedback_valutazione": feedback,
        "valutazione": valutazione,
        "tentativi": state["tentativi"],
        "risultati_query": state["risultati_query"],
        "risposta_finale": state["risposta_finale"],
        "esito": state["esito"],
        "messaggio_errore": state["messaggio_errore"],
    }


def route_dopo_valutazione(state: PipelineState) -> str:
    if "CORRETTA" in (state["valutazione"] or ""):
        return "corretta"
    if state["tentativi"] >= MAX_TENTATIVI:
        return "limite_raggiunto"
    return "errata"


# Node "Limite di tentativi raggiunto" (nessun successo forzato)
def nodo_limite_raggiunto(state: PipelineState) -> PipelineState:

    messaggio = (
        "Non sono riuscito a generare una query corretta per la tua richiesta "
        f"dopo {MAX_TENTATIVI} tentativi. Potresti riformulare la domanda?"
    )
    return \
    {
        "richiesta_utente": state["richiesta_utente"],
        "schema_database": state["schema_database"],
        "informazioni_estratte": state["informazioni_estratte"],
        "query_corrente": state["query_corrente"],
        "feedback_valutazione": state["feedback_valutazione"],
        "valutazione": state["valutazione"],
        "tentativi": state["tentativi"],
        "risultati_query": state["risultati_query"],
        "risposta_finale": state["risposta_finale"],
        "esito": "errore",
        "messaggio_errore": messaggio,
    }


# Node "Agente 3 - Esecuzione e risposta"
def nodo_esegui_e_rispondi(state: PipelineState) -> PipelineState:

    risultato = esegui_e_rispondi(state["richiesta_utente"], state["query_corrente"])

    return \
    {
        "richiesta_utente": state["richiesta_utente"],
        "schema_database": state["schema_database"],
        "informazioni_estratte": state["informazioni_estratte"],
        "query_corrente": state["query_corrente"],
        "feedback_valutazione": state["feedback_valutazione"],
        "valutazione": state["valutazione"],
        "tentativi": state["tentativi"],
        "risultati_query": risultato["risultati_query"],
        "risposta_finale": risultato["risposta_finale"],
        "esito": risultato["esito"],
        "messaggio_errore": risultato["messaggio_errore"],
    }


# LangGraph construction
def costruisci_pipeline():
    graph = StateGraph(PipelineState)

    graph.add_node("nodo_estrai_informazioni", nodo_estrai_informazioni)
    graph.add_node("nodo_genera_query", nodo_genera_query)
    graph.add_node("nodo_valuta_query", nodo_valuta_query)
    graph.add_node("nodo_limite_raggiunto", nodo_limite_raggiunto)
    graph.add_node("nodo_esegui_e_rispondi", nodo_esegui_e_rispondi)

    graph.set_entry_point("nodo_estrai_informazioni")

    graph.add_conditional_edges(
        "nodo_estrai_informazioni",
        route_dopo_estrazione,
        {
            "fuori_contesto": END,
            "in_contesto": "nodo_genera_query",
        },
    )

    graph.add_edge("nodo_genera_query", "nodo_valuta_query")

    graph.add_conditional_edges(
        "nodo_valuta_query",
        route_dopo_valutazione,
        {
            "corretta": "nodo_esegui_e_rispondi",
            "errata": "nodo_genera_query",
            "limite_raggiunto": "nodo_limite_raggiunto",
        },
    )

    graph.add_edge("nodo_limite_raggiunto", END)
    graph.add_edge("nodo_esegui_e_rispondi", END)

    return graph.compile()


PIPELINE = costruisci_pipeline()