# Agente 1 - Elaborazione della domanda dell'utente
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import get_pipeline

# Stato condiviso del grafo
class Agent1State(TypedDict):
    richiesta_originale: str
    richiesta_corrente: str
    tentativi_miglioramento: int
    classificazione: Optional[str]
    esito: Optional[str] # Può assumere due valori: "successo" oppure "errore"
    richiesta_elaborata: Optional[str] # Output da passare all'Agente 2
    messaggio_errore: Optional[str]

# Inizializzazione del modello LLM (Qwen2.5-0.5B-Instruct)
def crea_llm(model_name : str) -> ChatHuggingFace:
    """
    Carica Qwen2.5-0.5B-Instruct in locale e lo espone come ChatModel
    LangChain, cosi' da poter usare i prompt template a messaggi
    (system/human) e sfruttare il chat template del modello.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    hf_pipeline = get_pipeline("text-generation", model, tokenizer)

    llm = HuggingFacePipeline(pipeline = hf_pipeline)
    return ChatHuggingFace(llm = llm)

# Estrae una delle tre etichette valide dall'output grezzo del LLM.
def _estrai_classificazione(testo: str) -> str:
    testo_upper = testo.strip().upper()

    if "FUORI_CONTESTO" in testo_upper:
        return "FUORI_CONTESTO"
    
    if "NON_CHIARA" in testo_upper:
        return "NON_CHIARA"
    
    if "CHIARA" in testo_upper:
        return "CHIARA"
    
    return "NON_CHIARA"

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
        return "fuori_contesto"

    classificazione = state["classificazione"]

    if classificazione == "CHIARA":
        return "chiara"
    
    if classificazione == "FUORI_CONTESTO":
        return "fuori_contesto"
    
    return "non_chiara"

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
            "chiara": "prepara_output_successo",
            "non_chiara": "migliora_request",
            "fuori_contesto": "invio_messaggio_errore",
        },
    )

    graph.add_edge("migliora_request", "valuta_request")
    graph.add_edge("invio_messaggio_errore", END)
    graph.add_edge("prepara_output_successo", END)

    return graph.compile()

def get_prompt_valutazione() -> ChatPromptTemplate:

    return ChatPromptTemplate.from_messages \
    (
        [
            ("system",
            "Sei un modulo di classificazione di un sistema che trasforma domande in "
            "linguaggio naturale in query SQL. Classifica la richiesta dell'utente "
            "in una di queste tre categorie:\n"
            "- CHIARA: e' una domanda chiara e sufficientemente specifica per essere "
            "trasformata in una query SQL.\n"
            "- NON_CHIARA: riguarda un'interrogazione al database ma e' vaga, "
            "ambigua o incompleta.\n"
            "- FUORI_CONTESTO: non e' affatto un'interrogazione su un database "
            "(es. saluti, richieste completamente estranee).\n\n"
            "Rispondi solo con una di queste tre parole, senza altro testo."),
            ("human", "{richiesta}")
        ]
    )

def get_prompt_miglioramento() -> ChatPromptTemplate:

    return ChatPromptTemplate.from_messages \
    (
        [
            ("system",
            "Sei un modulo che riformula richieste ambigue o incomplete rivolte a un "
            "database, rendendole chiare e precise, mantenendo l'intento originale "
            "dell'utente. Rispondi SOLO con la richiesta riformulata in una singola "
            "frase, senza spiegazioni aggiuntive."),
            ("human", "{richiesta}")
        ]
    )

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
        "Mostrami i clienti che hanno speso di piu' nell'ultimo anno",
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