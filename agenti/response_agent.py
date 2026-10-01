# Agent 3 - Executes the approved SQL query and produces the final answer.
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.utils import crea_llm, DB

# "Qwen/Qwen3-8B"
MODEL_NAME = "Qwen/Qwen3-0.6B"


def get_prompt_elaborazione() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are a module that turns the raw results of an SQL query "
             "into a clear, concise natural language answer for the end "
             "user, consistent with their original request. Do not mention "
             "SQL, table or column names explicitly; just answer "
             "conversationally based on the data. If the results are empty, "
             "say so clearly. Respond in Italian."),
            ("human",
             "Original request:\n{richiesta}\n\n"
             "Query results:\n{risultati_query}\n\n"
             "Write the final answer for the user."),
        ]
    )


def esegui_e_rispondi(richiesta_utente: str, query_approvata: str) -> dict:
    """
    Executes query_approvata against the shared database connection
    (utils.DB) and turns its results into a natural-language answer in
    Italian, consistent with the user's original request (not with the
    extracted information: this function receives the request as-is, per
    the updated architecture).

    Returns a dict with keys:
        risultati_query - raw result of the query execution, or None on failure
        risposta_finale - natural-language answer to show the user
        esito - "successo" | "errore"
        messaggio_errore - technical detail of the failure, or None
    """
    try:
        risultati = DB.run(query_approvata)
    except Exception as errore:
        return \
        {
            "risultati_query": None,
            "risposta_finale": "Non sono riuscito a recuperare i dati richiesti. Potresti riformulare la domanda?",
            "esito": "errore",
            "messaggio_errore": f"Errore durante l'esecuzione della query: {errore}",
        }

    output_grezzo = CHAIN_ELABORAZIONE.invoke(
        {
            "richiesta": richiesta_utente,
            "risultati_query": risultati,
        }
    )

    return \
    {
        "risultati_query": risultati,
        "risposta_finale": output_grezzo.strip(),
        "esito": "successo",
        "messaggio_errore": None,
    }


# Initialization of model, prompt and chain
LLM = crea_llm(MODEL_NAME)

PROMPT_ELABORAZIONE = get_prompt_elaborazione()
CHAIN_ELABORAZIONE = PROMPT_ELABORAZIONE | LLM | StrOutputParser()