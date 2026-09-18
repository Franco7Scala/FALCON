# Agent 1 - Extraction and evaluation of the relevant information in the user's request
from typing import TypedDict, Optional, Tuple
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils import crea_llm, get_descrizione_dominio

# Shared graph state
class Agent1State(TypedDict):
    richiesta_utente: str
    informazioni_estratte: Optional[str]
    tentativi_estrazione: int
    classificazione: Optional[str]
    valutazione: Optional[str]
    feedback_valutazione: Optional[str]
    esito: Optional[str] # "successo" or "errore"
    informazioni_finali: Optional[str] # output to be passed to Agent 2
    estrazione_forzata: bool # True if sent forward after the attempt limit, without a positive evaluation
    messaggio_errore: Optional[str]

# "google/gemma-2-9b-it"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_TENTATIVI_ESTRAZIONE = 3  # Used to avoid infinite loops

# Utility: parsing the raw output of the LLM nodes
def _analizza_estrazione(testo: str) -> Tuple[str, Optional[str]]:
    """
    Parses the raw output of the "Estrapola Informazioni" node and returns a
    tuple (classification, extracted_information).
    If the model recognized the request as out of context it responds with
    the single word FUORI_CONTESTO; in that case there is no new information
    to return.
    """
    testo_pulito = testo.strip()

    if testo_pulito.upper().startswith("FUORI_CONTESTO"):
        return "FUORI_CONTESTO", None
    return "IN_CONTESTO", testo_pulito

def _analizza_valutazione(testo: str) -> Tuple[str, Optional[str]]:
    """
    Parses the raw output of the "Valuta Informazioni" node. The expected
    format is: "SUFFICIENTI" or "DA_COMPLETARE: <what is missing>"
    """
    testo_pulito = testo.strip()
    testo_upper = testo_pulito.upper()

    if testo_upper.startswith("SUFFICIENTI"):
        return "SUFFICIENTI", None

    if "DA_COMPLETARE" in testo_upper:
        if ":" in testo_pulito:
            motivo = testo_pulito.split(":", 1)[1].strip()
        else:
            motivo = testo_pulito
        return "DA_COMPLETARE", motivo or "No specific reason provided."
    return "DA_COMPLETARE", testo_pulito

# Node "Estrapola Informazioni"
def get_prompt_estrazione() -> ChatPromptTemplate:

    descrizione_dominio = get_descrizione_dominio()

    return ChatPromptTemplate.from_messages \
    (
        [
            ("system",
             "You are a module that extracts, from a user request, the "
             "information relevant for querying a specific database. "
             f"{descrizione_dominio}\n\n"
             "If the request is NOT related to this database at all "
             "(e.g., greetings, small talk, weather, general knowledge "
             "questions, or anything that cannot be answered by querying "
             "this database), respond with exactly one word: FUORI_CONTESTO.\n\n"
             "Otherwise, respond with a short list of the relevant "
             "information found in the request so far (entities, filters, "
             "conditions, requested fields), updating and completing the "
             "information already extracted in previous turns, if any. "
             "Respond only with this list, without additional explanations.\n\n"
             "Examples:\n"
             "Request: \"Show me all customers who spent more than 100 last month\"\n"
             "Already extracted: Nessuna informazione estratta finora.\n"
             "Answer: - entity: customers\n- filter: total spending > 100\n- period: last month\n\n"
             "Request: \"What's the weather like today in Rome?\"\n"
             "Already extracted: Nessuna informazione estratta finora.\n"
             "Answer: FUORI_CONTESTO\n\n"
             "Request: \"Hello, how are you?\"\n"
             "Already extracted: Nessuna informazione estratta finora.\n"
             "Answer: FUORI_CONTESTO"),
            ("human", "Request: \"{richiesta}\"\n" "Already extracted: {informazioni_finora}"),
        ]
    )

def estrapola_informazioni(state: Agent1State) -> Agent1State:

    informazioni_finora = state["informazioni_estratte"] or "Nessuna informazione estratta finora."

    output_grezzo = CHAIN_ESTRAZIONE.invoke \
    (
        {
            "richiesta": state["richiesta_utente"],
            "informazioni_finora": informazioni_finora,
        }
    )

    classificazione, nuove_informazioni = _analizza_estrazione(output_grezzo)

    return \
    {
        "richiesta_utente": state["richiesta_utente"],
        "informazioni_estratte": nuove_informazioni if classificazione == "IN_CONTESTO" else state["informazioni_estratte"],
        "tentativi_estrazione": state["tentativi_estrazione"] + 1,
        "classificazione": classificazione,
        "valutazione": state["valutazione"],
        "feedback_valutazione": state["feedback_valutazione"],
        "esito": state["esito"],
        "informazioni_finali": state["informazioni_finali"],
        "estrazione_forzata": state["estrazione_forzata"],
        "messaggio_errore": state["messaggio_errore"],
    }

# Node "Valuta Informazioni"
def get_prompt_valutazione() -> ChatPromptTemplate:

    descrizione_dominio = get_descrizione_dominio()

    return ChatPromptTemplate.from_messages \
    (
        [
            ("system",
             "You are a module that checks whether the information "
             "extracted so far from a user request is sufficient to build a "
             f"correct SQL query against a specific database. {descrizione_dominio}\n\n"
             "If the extracted information is sufficient, respond EXACTLY "
             "with the word SUFFICIENTI.\n"
             "Otherwise, respond with 'DA_COMPLETARE: ' followed by a brief "
             "explanation of what is still missing or ambiguous."),
            ("human",
             "Original request:\n{richiesta}\n\n"
             "Extracted information:\n{informazioni}"),
        ]
    )

def valuta_informazioni(state: Agent1State) -> Agent1State:

    output_grezzo = CHAIN_VALUTAZIONE.invoke \
    (
        {
            "richiesta": state["richiesta_utente"],
            "informazioni": state["informazioni_estratte"],
        }
    )

    valutazione, feedback = _analizza_valutazione(output_grezzo)

    return \
    {
        "richiesta_utente": state["richiesta_utente"],
        "informazioni_estratte": state["informazioni_estratte"],
        "tentativi_estrazione": state["tentativi_estrazione"],
        "classificazione": state["classificazione"],
        "valutazione": valutazione,
        "feedback_valutazione": feedback,
        "esito": state["esito"],
        "informazioni_finali": state["informazioni_finali"],
        "estrazione_forzata": state["estrazione_forzata"],
        "messaggio_errore": state["messaggio_errore"],
    }

# Node "Invio di un messaggio di errore" (request out of context)
def invio_messaggio_errore(state: Agent1State) -> Agent1State:
    messaggio = "Interrogazione non chiara o non valida. Potresti essere piu' preciso?"
    return \
    {
        "richiesta_utente": state["richiesta_utente"],
        "informazioni_estratte": state["informazioni_estratte"],
        "tentativi_estrazione": state["tentativi_estrazione"],
        "classificazione": state["classificazione"],
        "valutazione": state["valutazione"],
        "feedback_valutazione": state["feedback_valutazione"],
        "esito": "errore",
        "informazioni_finali": state["informazioni_finali"],
        "estrazione_forzata": state["estrazione_forzata"],
        "messaggio_errore": messaggio,
    }

# Final node: prepares the output for Agent 2
def prepara_output_successo(state: Agent1State) -> Agent1State:

    forzato = state["valutazione"] != "SUFFICIENTI"
    return \
    {
        "richiesta_utente": state["richiesta_utente"],
        "informazioni_estratte": state["informazioni_estratte"],
        "tentativi_estrazione": state["tentativi_estrazione"],
        "classificazione": state["classificazione"],
        "valutazione": state["valutazione"],
        "feedback_valutazione": state["feedback_valutazione"],
        "esito": "successo",
        "informazioni_finali": state["informazioni_estratte"],
        "estrazione_forzata": forzato,
        "messaggio_errore": state["messaggio_errore"],
    }

# Conditional routing
def route_dopo_estrazione(state: Agent1State) -> str:

    if state["classificazione"] == "FUORI_CONTESTO":
        return "fuori_contesto"
    return "in_contesto"

def route_dopo_valutazione(state: Agent1State) -> str:

    if state["valutazione"] == "SUFFICIENTI":
        return "sufficienti"
    if state["tentativi_estrazione"] >= MAX_TENTATIVI_ESTRAZIONE:
        return "limite_raggiunto"
    return "da_completare"

# LangGraph construction
def costruisci_agente_1():
    graph = StateGraph(Agent1State)

    graph.add_node("estrapola_informazioni", estrapola_informazioni)
    graph.add_node("valuta_informazioni", valuta_informazioni)
    graph.add_node("invio_messaggio_errore", invio_messaggio_errore)
    graph.add_node("prepara_output_successo", prepara_output_successo)

    graph.set_entry_point("estrapola_informazioni")

    graph.add_conditional_edges \
    (
        "estrapola_informazioni",
        route_dopo_estrazione,
        {
            "fuori_contesto": "invio_messaggio_errore",
            "in_contesto": "valuta_informazioni",
        },
    )

    graph.add_conditional_edges \
    (
        "valuta_informazioni",
        route_dopo_valutazione,
        {
            "sufficienti": "prepara_output_successo",
            "limite_raggiunto": "prepara_output_successo",
            "da_completare": "estrapola_informazioni",
        },
    )

    graph.add_edge("invio_messaggio_errore", END)
    graph.add_edge("prepara_output_successo", END)

    return graph.compile()

# Initialization of model, prompts, chains and graph
LLM = crea_llm(MODEL_NAME)

PROMPT_ESTRAZIONE = get_prompt_estrazione()
CHAIN_ESTRAZIONE = PROMPT_ESTRAZIONE | LLM | StrOutputParser()

PROMPT_VALUTAZIONE = get_prompt_valutazione()
CHAIN_VALUTAZIONE = PROMPT_VALUTAZIONE | LLM | StrOutputParser()

AGENTE_1 = costruisci_agente_1()

if __name__ == "__main__":

    richieste_di_prova = \
    [
        "Mostrami i clienti che vanno di nome Mario",
        "dimmi qualcosa su quelli",
        "Che tempo fa oggi a Roma?",
    ]

    for richiesta in richieste_di_prova:
        stato_iniziale: Agent1State = \
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

        risultato = AGENTE_1.invoke(stato_iniziale)

        print(f"\nRichiesta originale: {richiesta}")
        if risultato["esito"] == "successo":
            print(f"-> Da inoltrare all'Agente 2: {risultato['informazioni_finali']}")
            if risultato["estrazione_forzata"]:
                print("(attenzione: inviato dopo il limite di tentativi, senza validazione)")
        else:
            print(f"-> {risultato['messaggio_errore']}")