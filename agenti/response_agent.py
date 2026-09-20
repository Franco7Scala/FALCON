# Agent 3 - Executes the SQL query on the database and produces the final answer for the user
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils import crea_llm, DB

# Shared graph state
class Agent3State(TypedDict):
    informazioni_estratte: str
    query_finale: str
    risultati_query: Optional[str]
    esito: Optional[str] # "successo" | "errore"
    risposta_finale: Optional[str] # natural-language answer to show to the user
    messaggio_errore: Optional[str]

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# Node "Applicazione della query sul Database"
def applica_query_sul_database(state: Agent3State) -> Agent3State:
    """
    Executes the SQL query received from Agent 2 against the shared database
    connection (utils.DB) and stores its raw result. If execution fails
    (invalid SQL, schema mismatch, etc.), the graph is routed towards the
    error-handling node instead of raising.
    """
    try:
        risultati = DB.run(state["query_finale"])
        return \
        {
            "informazioni_estratte": state["informazioni_estratte"],
            "query_finale": state["query_finale"],
            "risultati_query": risultati,
            "esito": "successo",
            "risposta_finale": state["risposta_finale"],
            "messaggio_errore": None,
        }
    except Exception as errore:
        return \
        {
            "informazioni_estratte": state["informazioni_estratte"],
            "query_finale": state["query_finale"],
            "risultati_query": None,
            "esito": "errore",
            "risposta_finale": state["risposta_finale"],
            "messaggio_errore": f"Errore durante l'esecuzione della query: {errore}",
        }

# Node "Elaborazione della risposta"
def get_prompt_elaborazione() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are a module that turns the raw results of an SQL query "
             "into a clear, concise natural language answer for the end "
             "user, consistent with the information extracted from their "
             "original request. Do not mention SQL, table or column names "
             "explicitly; just answer conversationally based on the data. "
             "If the results are empty, say so clearly. Respond in Italian."),
            ("human",
             "Extracted information:\n{informazioni_estratte}\n\n"
             "Query results:\n{risultati_query}\n\n"
             "Write the final answer for the user."),
        ]
    )

def elabora_risposta(state: Agent3State) -> Agent3State:

    output_grezzo = CHAIN_ELABORAZIONE.invoke \
    (
        {
            "informazioni_estratte": state["informazioni_estratte"],
            "risultati_query": state["risultati_query"],
        }
    )

    return \
    {
        "informazioni_estratte": state["informazioni_estratte"],
        "query_finale": state["query_finale"],
        "risultati_query": state["risultati_query"],
        "esito": "successo",
        "risposta_finale": output_grezzo.strip(),
        "messaggio_errore": state["messaggio_errore"],
    }

# Node "Gestione errore di esecuzione"
def gestisci_errore_esecuzione(state: Agent3State) -> Agent3State:

    messaggio = "Non sono riuscito a recuperare i dati richiesti. Potresti riformulare la domanda?"
    return \
    {
        "informazioni_estratte": state["informazioni_estratte"],
        "query_finale": state["query_finale"],
        "risultati_query": state["risultati_query"],
        "esito": "errore",
        "risposta_finale": messaggio,
        "messaggio_errore": state["messaggio_errore"],
    }

# Conditional routing
def route_dopo_esecuzione(state: Agent3State) -> str:
    if state["esito"] == "errore":
        return "errore"
    return "successo"

# LangGraph construction
def costruisci_agente_3():
    graph = StateGraph(Agent3State)

    graph.add_node("applica_query_sul_database", applica_query_sul_database)
    graph.add_node("elabora_risposta", elabora_risposta)
    graph.add_node("gestisci_errore_esecuzione", gestisci_errore_esecuzione)

    graph.set_entry_point("applica_query_sul_database")

    graph.add_conditional_edges(
        "applica_query_sul_database",
        route_dopo_esecuzione,
        {
            "successo": "elabora_risposta",
            "errore": "gestisci_errore_esecuzione",
        },
    )

    graph.add_edge("elabora_risposta", END)
    graph.add_edge("gestisci_errore_esecuzione", END)

    return graph.compile()

# Initialization of model, prompts, chains and graph
LLM = crea_llm(MODEL_NAME)

PROMPT_ELABORAZIONE = get_prompt_elaborazione()
CHAIN_ELABORAZIONE = PROMPT_ELABORAZIONE | LLM | StrOutputParser()

AGENTE_3 = costruisci_agente_3()

# Usage example
if __name__ == "__main__":
    # Simulates the outputs of Agent 1 and Agent 2
    informazioni_di_prova = \
    (
        "- entita': clienti\n"
        "- campi richiesti: nome, cognome\n"
        "- filtro: hanno un ordine con totale > 100"
    )
    query_di_prova = \
    (
        "SELECT c.nome, c.cognome FROM clienti c "
        "JOIN ordini o ON o.cliente_id = c.id "
        "WHERE o.totale > 100"
    )

    stato_iniziale: Agent3State = \
    {
        "informazioni_estratte": informazioni_di_prova,
        "query_finale": query_di_prova,
        "risultati_query": None,
        "esito": None,
        "risposta_finale": None,
        "messaggio_errore": None,
    }

    risultato = AGENTE_3.invoke(stato_iniziale)

    print(f"Risultati grezzi: {risultato['risultati_query']}")
    print(f"\nRisposta finale: {risultato['risposta_finale']}")