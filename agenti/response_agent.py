# Agent 3 - Executes the generated SQL query, checks whether its results
# actually satisfy the user's original request, and either answers or asks
# Agent 2 to regenerate the query.
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.utils import crea_llm, DB


MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"


def _estrai_dopo_etichetta(testo: str, etichetta: str, fino_a: Optional[str] = None) -> Optional[str]:
    """
    Returns the text following "<etichetta>:" up to the next occurrence of
    "<fino_a>:" (if given), or to the end of the string otherwise. Returns
    None if the label itself is not found.
    """
    testo_upper = testo.upper()
    marcatore = f"{etichetta.upper()}:"
    indice_inizio = testo_upper.find(marcatore)
    if indice_inizio == -1:
        return None

    inizio_contenuto = indice_inizio + len(marcatore)

    if fino_a is not None:
        indice_fine = testo_upper.find(f"{fino_a.upper()}:", inizio_contenuto)
        if indice_fine != -1:
            return testo[inizio_contenuto:indice_fine].strip() or None

    return testo[inizio_contenuto:].strip() or None


def _analizza_valutazione_risultati(testo: str):
    """
    Parses the raw output of the result-evaluation prompt. Expected format:
    "ESITO: SODDISFATTA
     RISPOSTA: <final natural-language answer>"
    or
    "ESITO: DA_RIGENERARE
     FEEDBACK: <what's wrong with the query/results>"
    Returns (valutazione, risposta_finale, feedback).
    """
    testo_pulito = testo.strip()

    if "SODDISFATTA" in testo_pulito.upper():
        risposta = _estrai_dopo_etichetta(testo_pulito, "RISPOSTA")
        return "SODDISFATTA", (risposta or testo_pulito), None

    feedback = _estrai_dopo_etichetta(testo_pulito, "FEEDBACK")
    return "DA_RIGENERARE", None, (feedback or testo_pulito)


def get_prompt_valutazione_risultati() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are a module that checks whether the raw results of an "
             "already-executed SQL query actually satisfy a user's original "
             "request, and either writes the final answer or explains what "
             "needs to change.\n\n"
             "If the results correctly and completely answer the request, "
             "respond with:\n"
             "ESITO: SODDISFATTA\n"
             "RISPOSTA: <a clear, concise natural language answer for the "
             "user, in Italian, based on the data. Do not mention SQL, "
             "table or column names explicitly. If the results are empty "
             "and that is a legitimate answer, say so clearly.>\n\n"
             "If the results do NOT satisfy the request (wrong columns, "
             "wrong aggregation, data that doesn't match what was asked, "
             "unexpectedly empty results, etc.), respond with:\n"
             "ESITO: DA_RIGENERARE\n"
             "FEEDBACK: <brief explanation of what is wrong, useful for "
             "regenerating the query>"),
            ("human",
             "Original request:\n{richiesta}\n\n"
             "Query that was executed:\n{query}\n\n"
             "Raw results:\n{risultati_query}\n\n"
             "Evaluate."),
        ]
    )


def esegui_e_valuta(richiesta_utente: str, query: str) -> dict:
    """
    Executes `query` against the shared database connection (utils.DB) and
    evaluates whether its results satisfy richiesta_utente.

    Returns a dict with keys:
        valutazione: "SODDISFATTA" | "DA_RIGENERARE"
        risultati_query: raw result of the execution, or None if it failed
        risposta_finale: final natural-language answer (set only when SODDISFATTA)
        feedback: explanation of what's wrong, for Agent 2's next
        attempt (set only when DA_RIGENERARE - either a
        database execution error or a semantic mismatch
        identified by the evaluation)
    """
    try:
        risultati = DB.run(query)
    except Exception as errore:
        return \
        {
            "valutazione": "DA_RIGENERARE",
            "risultati_query": None,
            "risposta_finale": None,
            "feedback": f"La query ha generato un errore di esecuzione sul database: {errore}",
        }

    output_grezzo = CHAIN_VALUTAZIONE.invoke(
        {
            "richiesta": richiesta_utente,
            "query": query,
            "risultati_query": risultati,
        }
    )

    print("\n===== DEBUG AGENT 3 RESULT EVALUATION =====")
    print(repr(output_grezzo))
    print("=============================================\n")

    valutazione, risposta_finale, feedback = _analizza_valutazione_risultati(output_grezzo)

    return \
    {
        "valutazione": valutazione,
        "risultati_query": risultati,
        "risposta_finale": risposta_finale,
        "feedback": feedback,
    }


LLM = crea_llm(MODEL_NAME)
PROMPT_VALUTAZIONE = get_prompt_valutazione_risultati()
CHAIN_VALUTAZIONE = PROMPT_VALUTAZIONE | LLM | StrOutputParser()