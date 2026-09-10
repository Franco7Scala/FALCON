# Agente 1 - Elaborazione della domanda dell'utente
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils import crea_llm, get_descrizione_dominio

# Stato condiviso del grafo
class Agent1State(TypedDict):
    richiesta_originale: str
    richiesta_corrente: str
    tentativi_miglioramento: int
    classificazione: Optional[str]
    esito: Optional[str] # Può assumere due valori: "successo" oppure "errore"
    richiesta_elaborata: Optional[str] # Output da passare all'Agente 2
    messaggio_errore: Optional[str]

# Estrae una delle tre etichette valide dall'output grezzo del LLM.
def _estrai_classificazione(testo: str) -> str:
    testo_upper = testo.strip().upper()

    if "OUT_OF_CONTEXT" in testo_upper:
        return "OUT_OF_CONTEXT"
    
    if "UNCLEAR" in testo_upper:
        return "UNCLEAR"
    
    if "CLEAR" in testo_upper:
        return "CLEAR"
    
    return "UNCLEAR"

# Nodo "Valuta Request": classifica la richiesta
def valuta_request(state: Agent1State) -> Agent1State:

    output_grezzo = CHAIN_VALUTAZIONE.invoke({"richiesta": state["richiesta_corrente"]})

    classificazione = _estrai_classificazione(output_grezzo)
    return \
    {
        "richiesta_originale": state["richiesta_originale"],
        "richiesta_corrente": state["richiesta_corrente"],
        "tentativi_miglioramento": state["tentativi_miglioramento"],
        "classificazione": classificazione,
        "esito": state["esito"],
        "richiesta_elaborata": state["richiesta_elaborata"],
        "messaggio_errore": state["messaggio_errore"]
    }

# Nodo "Migliora Request": riformula una richiesta poco chiara
def migliora_request(state: Agent1State) -> Agent1State:

    richiesta_migliorata = CHAIN_MIGLIORAMENTO.invoke({"richiesta": state["richiesta_corrente"]}).strip()
    return \
    {
        "richiesta_originale": state["richiesta_originale"],
        "richiesta_corrente": richiesta_migliorata,
        "tentativi_miglioramento": state["tentativi_miglioramento"] + 1,
        "classificazione": state["classificazione"],
        "esito": state["esito"],
        "richiesta_elaborata": state["richiesta_elaborata"],
        "messaggio_errore": state["messaggio_errore"]
    }

# Nodo "Invio di un messaggio di errore" (richiesta fuori contesto)
def invio_messaggio_errore(state: Agent1State) -> Agent1State:

    messaggio = "Interrogazione non chiara o non valida. Potresti essere piu' preciso?"
    return \
    {
        "richiesta_originale": state["richiesta_originale"],
        "richiesta_corrente": state["richiesta_corrente"],
        "tentativi_miglioramento": state["tentativi_miglioramento"],
        "classificazione": state["classificazione"],
        "esito": "errore",
        "richiesta_elaborata": state["richiesta_elaborata"],
        "messaggio_errore": messaggio
    }

# Nodo di successo: prepara l'output per l'Agente 2
def prepara_output_successo(state: Agent1State) -> Agent1State:

    return \
    {
        "richiesta_originale": state["richiesta_originale"],
        "richiesta_corrente": state["richiesta_corrente"],
        "tentativi_miglioramento": state["tentativi_miglioramento"],
        "classificazione": state["classificazione"],
        "esito": "successo",
        "richiesta_elaborata": state["richiesta_corrente"],
        "messaggio_errore": state["messaggio_errore"]
    }

# Routing condizionale dopo la valutazione
def route_dopo_valutazione(state: Agent1State) -> str:

    if state["tentativi_miglioramento"] >= MAX_TENTATIVI_MIGLIORAMENTO:
        return "OUT_OF_CONTEXT"

    classificazione = state["classificazione"]

    if classificazione == "CLEAR":
        return "CLEAR"
    
    if classificazione == "OUT_OF_CONTEXT":
        return "OUT_OF_CONTEXT"
    
    return "UNCLEAR"

# Costruzione del grafo LangGraph
def costruisci_agente_1():
    graph = StateGraph(Agent1State)

    graph.add_node("valuta_request", valuta_request)
    graph.add_node("migliora_request", migliora_request)
    graph.add_node("invio_messaggio_errore", invio_messaggio_errore)
    graph.add_node("prepara_output_successo", prepara_output_successo)

    graph.set_entry_point("valuta_request")

    graph.add_conditional_edges(
        "valuta_request",
        route_dopo_valutazione,
        {
            "CLEAR": "prepara_output_successo",
            "UNCLEAR": "migliora_request",
            "OUT_OF_CONTEXT": "invio_messaggio_errore",
        },
    )

    graph.add_edge("migliora_request", "valuta_request")
    graph.add_edge("invio_messaggio_errore", END)
    graph.add_edge("prepara_output_successo", END)

    return graph.compile()

def get_prompt_valutazione() -> ChatPromptTemplate:

    descrizione_dominio = get_descrizione_dominio()

    return ChatPromptTemplate.from_messages \
    (
        [
            ("system",
            "You are a classification module for a system that converts natural language "
            "questions into SQL queries over a specific database. "
            f"{descrizione_dominio}\n\n"
            "Classify the user's request into one of these three categories:\n"
            "- CLEAR: a question that is clear and specific enough to be "
            "converted into an SQL query about this database.\n"
            "- UNCLEAR: relates to a query about this database, but is vague, "
            "ambiguous, or incomplete.\n"
            "- OUT_OF_CONTEXT: not related to this database at all "
            "(e.g., greetings, small talk, weather, general knowledge questions, "
            "or anything that cannot be answered by querying this database).\n\n"
            "Examples:\n"
            "Request: \"Show me all customers who spent more than 100 last month\"\n"
            "Answer: CLEAR\n\n"
            "Request: \"Show me the customers\"\n"
            "Answer: UNCLEAR\n\n"
            "Request: \"What's the weather like today in Rome?\"\n"
            "Answer: OUT_OF_CONTEXT\n\n"
            "Request: \"Hello, how are you?\"\n"
            "Answer: OUT_OF_CONTEXT\n\n"
            "Respond with only one of these three words, with no other text."),
            ("human", "{richiesta}")
        ]
    )

def get_prompt_miglioramento() -> ChatPromptTemplate:

    return ChatPromptTemplate.from_messages \
    (
        [
            ("system",
            "You are a module that reformulates ambiguous or incomplete requests made to a "
            "database, making them clear and precise while preserving the user's "
            "original intent. Respond only with the reformulated request in a single "
            "sentence, without additional explanations."),
            ("human", "{richiesta}")
        ]
    )

# Verrà sostituito con un modello più potente, ad esempio "google/gemma-2-9b-it"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_TENTATIVI_MIGLIORAMENTO = 3 # Serve ad evitare loop infiniti
LLM = crea_llm(MODEL_NAME)

PROMPT_VALUTAZIONE = get_prompt_valutazione()
# Viene utilizzato StrOutParser() per estrarre la risposta da parte dell'LLM come stringa, siccome l'LLM restituisce un
# oggetto di tipo AIMessage
CHAIN_VALUTAZIONE = PROMPT_VALUTAZIONE | LLM | StrOutputParser()

PROMPT_MIGLIORAMENTO = get_prompt_miglioramento()
CHAIN_MIGLIORAMENTO = PROMPT_MIGLIORAMENTO | LLM | StrOutputParser()

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
            "richiesta_originale": richiesta,
            "richiesta_corrente": richiesta,
            "tentativi_miglioramento": 0,
            "classificazione": None,
            "esito": None,
            "richiesta_elaborata": None,
            "messaggio_errore": None,
        }

        risultato = AGENTE_1.invoke(stato_iniziale)

        print(f"\nRichiesta originale: {richiesta}")
        if risultato["esito"] == "successo":
            print(f"-> Da inoltrare all'Agente 2: {risultato['richiesta_elaborata']}")
        else:
            print(f"-> {risultato['messaggio_errore']}")