# Agent 2 - Converts the extracted information into an SQL query
from typing import TypedDict, Optional, Tuple
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils import crea_llm, DB

# Shared graph state
class Agent2State(TypedDict):
    informazioni_estratte: str # input from Agent 1
    schema_database: str # DB schema
    query_corrente: Optional[str]
    valutazione: Optional[str] #"CORRETTA" or "DA_MIGLIORARE"
    feedback_valutazione: Optional[str]
    tentativi_miglioramento: int
    esito: Optional[str] # "successo" (the graph has no real blocking error state)
    query_finale: Optional[str] # output to be passed to Agent 3
    accettata_forzatamente: bool # True if sent after the attempt limit, without validation
    messaggio_errore: Optional[str]

# "google/gemma-2-9b-it"
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
MAX_TENTATIVI_MIGLIORAMENTO = 3 # Used to avoid infinite loops

# Utility: cleaning up the model output
def _pulisci_query(testo: str) -> str:
    """
    Removes any markdown code blocks (```sql ... ```) that the model might
    return despite the prompt instructions.
    """
    testo = testo.strip()
    if testo.startswith("```"):
        righe = testo.split("\n")
        righe = righe[1:] if len(righe) > 1 else righe
        testo = "\n".join(righe)
        testo = testo.rsplit("```", 1)[0]
    return testo.strip()

def _estrai_valutazione(testo: str) -> Tuple[str, Optional[str]]:
    """
    Parses the raw output of the evaluation node and returns a tuple
    (classification, feedback). The expected format is: "CORRETTA" or "DA_MIGLIORARE: <reason>"
    """
    testo_pulito = testo.strip()
    testo_upper = testo_pulito.upper()

    if testo_upper.startswith("CORRETTA"):
        return "CORRETTA", None

    if "DA_MIGLIORARE" in testo_upper:
        if ":" in testo_pulito:
            feedback = testo_pulito.split(":", 1)[1].strip()
        else:
            feedback = testo_pulito
        return "DA_MIGLIORARE", feedback or "No specific reason provided."

    return "DA_MIGLIORARE", testo_pulito

# Node "Genera Query SQL"
def get_prompt_generazione() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages \
    (
        [
            ("system",
             "Sei un modulo che traduce informazioni estratte da una "
             "richiesta in linguaggio naturale in query SQL valide, "
             "basandoti esclusivamente sullo schema del database fornito. "
             "Rispondi SOLO con la query SQL, senza commenti, spiegazioni o "
             "blocchi di codice markdown."),
            ("human",
             "Schema del database:\n{schema}\n\n"
             "Informazioni estratte dalla richiesta dell'utente:\n{informazioni_estratte}\n\n"
             "Genera la query SQL corrispondente."),
        ]
    )

def genera_query(state: Agent2State) -> Agent2State:
    output_grezzo = CHAIN_GENERAZIONE.invoke \
    (
        {
            "schema": state["schema_database"],
            "informazioni_estratte": state["informazioni_estratte"],
        }
    )

    return \
    {
        "informazioni_estratte": state["informazioni_estratte"],
        "schema_database": state["schema_database"],
        "query_corrente": _pulisci_query(output_grezzo),
        "valutazione": state["valutazione"],
        "feedback_valutazione": state["feedback_valutazione"],
        "tentativi_miglioramento": state["tentativi_miglioramento"],
        "esito": state["esito"],
        "query_finale": state["query_finale"],
        "accettata_forzatamente": state["accettata_forzatamente"],
        "messaggio_errore": state["messaggio_errore"],
    }


# Node "Valuta Query SQL"
def get_prompt_valutazione() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages \
    (
        [
            ("system",
             "Sei un modulo che verifica se una query SQL risponde "
             "correttamente alle informazioni estratte dalla richiesta "
             "originale dell'utente, rispetto allo schema del database "
             "fornito. Controlla: 1) la sintassi SQL, 2) che tabelle e "
             "colonne usate esistano nello schema, 3) che la query risponda "
             "realmente alle informazioni fornite.\n\n"
             "Se la query e' corretta sotto tutti questi aspetti, rispondi "
             "ESATTAMENTE con la parola CORRETTA.\n"
             "Altrimenti rispondi con 'DA_MIGLIORARE: ' seguito da una breve "
             "spiegazione del problema riscontrato."),
            ("human",
             "Schema del database:\n{schema}\n\n"
             "Informazioni estratte dalla richiesta originale:\n{informazioni_estratte}\n\n"
             "Query da valutare:\n{query}"),
        ]
    )


def valuta_query(state: Agent2State) -> Agent2State:
    output_grezzo = CHAIN_VALUTAZIONE.invoke \
    (
        {
            "schema": state["schema_database"],
            "informazioni_estratte": state["informazioni_estratte"],
            "query": state["query_corrente"],
        }
    )

    classificazione, feedback = _estrai_valutazione(output_grezzo)

    return \
    {
        "informazioni_estratte": state["informazioni_estratte"],
        "schema_database": state["schema_database"],
        "query_corrente": state["query_corrente"],
        "valutazione": classificazione,
        "feedback_valutazione": feedback,
        "tentativi_miglioramento": state["tentativi_miglioramento"],
        "esito": state["esito"],
        "query_finale": state["query_finale"],
        "accettata_forzatamente": state["accettata_forzatamente"],
        "messaggio_errore": state["messaggio_errore"],
    }

# Node "Migliora Query SQL"
def get_prompt_miglioramento() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages \
    (
        [
            ("system",
             "Sei un modulo che corregge una query SQL non ancora corretta, "
             "sulla base del feedback ricevuto e dello schema del database. "
             "Rispondi SOLO con la query SQL corretta, senza commenti, "
             "spiegazioni o blocchi di codice markdown."),
            ("human",
             "Schema del database:\n{schema}\n\n"
             "Informazioni estratte dalla richiesta originale:\n{informazioni_estratte}\n\n"
             "Query attuale:\n{query_corrente}\n\n"
             "Problema riscontrato:\n{feedback}\n\n"
             "Fornisci la query corretta."),
        ]
    )


def migliora_query(state: Agent2State) -> Agent2State:
    output_grezzo = CHAIN_MIGLIORAMENTO.invoke \
    (
        {
            "schema": state["schema_database"],
            "informazioni_estratte": state["informazioni_estratte"],
            "query_corrente": state["query_corrente"],
            "feedback": state["feedback_valutazione"] or "",
        }
    )

    return \
    {
        "informazioni_estratte": state["informazioni_estratte"],
        "schema_database": state["schema_database"],
        "query_corrente": _pulisci_query(output_grezzo),
        "valutazione": state["valutazione"],
        "feedback_valutazione": state["feedback_valutazione"],
        "tentativi_miglioramento": state["tentativi_miglioramento"] + 1,
        "esito": state["esito"],
        "query_finale": state["query_finale"],
        "accettata_forzatamente": state["accettata_forzatamente"],
        "messaggio_errore": state["messaggio_errore"],
    }

# Final node: prepares the output for Agent 3
def prepara_output_successo(state: Agent2State) -> Agent2State:
    forzato = state["valutazione"] != "CORRETTA"
    return \
    {
        "informazioni_estratte": state["informazioni_estratte"],
        "schema_database": state["schema_database"],
        "query_corrente": state["query_corrente"],
        "valutazione": state["valutazione"],
        "feedback_valutazione": state["feedback_valutazione"],
        "tentativi_miglioramento": state["tentativi_miglioramento"],
        "esito": "successo",
        "query_finale": state["query_corrente"],
        "accettata_forzatamente": forzato,
        "messaggio_errore": state["messaggio_errore"],
    }


# Conditional routing after evaluation
def route_dopo_valutazione(state: Agent2State) -> str:
    if state["valutazione"] == "CORRETTA":
        return "corretta"
    if state["tentativi_miglioramento"] >= MAX_TENTATIVI_MIGLIORAMENTO:
        return "limite_raggiunto"
    return "da_migliorare"


# LangGraph construction
def costruisci_agente_2():
    graph = StateGraph(Agent2State)

    graph.add_node("genera_query", genera_query)
    graph.add_node("valuta_query", valuta_query)
    graph.add_node("migliora_query", migliora_query)
    graph.add_node("prepara_output_successo", prepara_output_successo)

    graph.set_entry_point("genera_query")

    graph.add_edge("genera_query", "valuta_query")

    graph.add_conditional_edges(
        "valuta_query",
        route_dopo_valutazione,
        {
            "corretta": "prepara_output_successo",
            "limite_raggiunto": "prepara_output_successo",
            "da_migliorare": "migliora_query",
        },
    )

    graph.add_edge("migliora_query", "valuta_query")
    graph.add_edge("prepara_output_successo", END)

    return graph.compile()

# Initialization of model, prompts, chains and graph
LLM = crea_llm(MODEL_NAME)

PROMPT_GENERAZIONE = get_prompt_generazione()
CHAIN_GENERAZIONE = PROMPT_GENERAZIONE | LLM | StrOutputParser()

PROMPT_VALUTAZIONE = get_prompt_valutazione()
CHAIN_VALUTAZIONE = PROMPT_VALUTAZIONE | LLM | StrOutputParser()

PROMPT_MIGLIORAMENTO = get_prompt_miglioramento()
CHAIN_MIGLIORAMENTO = PROMPT_MIGLIORAMENTO | LLM | StrOutputParser()

AGENTE_2 = costruisci_agente_2()

if __name__ == "__main__":

    informazioni_di_prova = \
    (
        "- entita': clienti\n"
        "- campi richiesti: nome, cognome\n"
        "- filtro: hanno un ordine con totale > 100"
    )

    stato_iniziale: Agent2State = \
    {
        "informazioni_estratte": informazioni_di_prova,
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

    risultato = AGENTE_2.invoke(stato_iniziale)

    print(f"Informazioni estratte: {informazioni_di_prova}")
    print(f"Query finale: {risultato['query_finale']}")
    print(f"Tentativi di miglioramento: {risultato['tentativi_miglioramento']}")
    print(f"Accettata forzatamente: {risultato['accettata_forzatamente']}")