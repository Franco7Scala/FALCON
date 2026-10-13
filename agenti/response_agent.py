# Agent 3 - Executes the generated SQL query, checks whether its results
# actually satisfy the user's original request, and either answers or asks
# Agent 2 to regenerate the query.
import ast
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.utils import crea_llm, DB

# non-thinking Instruct release: never produces <think> blocks
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


# Caps on how much of a query result is shown to the evaluation LLM. The FULL
# raw result is still returned to the pipeline (so metrics and callers see
# everything); only the text placed INSIDE THE PROMPT is bounded. Without a
# cap, a query returning tens of thousands of rows (e.g. a SELECT * where a
# COUNT(*) was needed) puts millions of tokens in the prompt and runs the
# GPU out of memory.
MAX_RIGHE_NEL_PROMPT = 20
MAX_CARATTERI_NEL_PROMPT = 4000


def _riassumi_risultati(risultati: str) -> str:
    """
    Builds a bounded description of a raw query result (the string returned
    by DB.run, i.e. str() of a list of row tuples) for use inside the prompt:
    total number of rows, number of columns, and only the first
    MAX_RIGHE_NEL_PROMPT rows. The total row count is deliberately always
    included: it is exactly what lets the evaluator notice that a "how many"
    question got thousands of raw rows back instead of a single number.
    """
    if not risultati or not risultati.strip() or risultati.strip() == "[]":
        return "Numero totale di righe restituite: 0 (risultato vuoto)"

    righe = None
    try:
        valore = ast.literal_eval(risultati)
        if isinstance(valore, list):
            righe = valore
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        righe = None

    if righe is not None:
        totale = len(righe)
        prima = righe[0] if righe else None
        colonne = len(prima) if isinstance(prima, tuple) else 1
        anteprima = str(righe[:MAX_RIGHE_NEL_PROMPT])
        intestazione = f"Numero totale di righe restituite: {totale}; colonne per riga: {colonne}."
    else:
        # Not parseable as a Python literal (e.g. it contains datetime objects):
        # estimate the row count from the tuple separators and cut by characters.
        totale = risultati.count("), (") + 1
        anteprima = risultati
        intestazione = f"Numero totale di righe restituite (stimato): {totale}."

    if totale > MAX_RIGHE_NEL_PROMPT:
        intestazione += f" Sono mostrate solo le prime {MAX_RIGHE_NEL_PROMPT}:"

    if len(anteprima) > MAX_CARATTERI_NEL_PROMPT:
        anteprima = anteprima[:MAX_CARATTERI_NEL_PROMPT] + " ...[troncato]"

    return f"{intestazione}\n{anteprima}"


def get_prompt_valutazione_risultati():
    return ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are a SQL validation component in a Text-to-SQL system
            for process event logs.
            
            Your task is to evaluate whether an executed SQL query correctly
            implements the user's request.
            
            IMPORTANT: You must evaluate the query's logic, not whether the
            predicted activity seems intuitively plausible.
            
            For NEXT_ACTIVITY_PREDICTION tasks:
            
            1. Determine the complete ordered activity prefix from the
               original request.
            
            2. Verify that the SQL query matches the prefix in the correct
               order, preserving every repeated activity.
            
            3. Verify that the query retrieves the activity immediately
               following the complete prefix.
            
            4. Verify that the query aggregates next activities across
               matching training traces and selects the most frequent one.
            
            5. The predicted next activity does NOT need to appear in the
               original request, because the purpose of the task is to predict
               an activity that follows the supplied prefix.
            
            6. The predicted activity MAY be identical to an activity already
               present in the prefix. Repeated consecutive activities are
               valid process behavior.
            
            7. NEVER reject a query simply because the predicted activity:
               - is not explicitly mentioned in the original request;
               - already appears in the prefix;
               - appears more than once in the trace.
            
            8. Do not use the expected target label. It is not part of the
               request and must not be used to judge the query.
            
            9. A query that executes successfully is not automatically
               correct. Check its SQL logic against the original request.
            
            10. If the query correctly implements the task and returns a
                predicted activity, return:
            
                ESITO: SODDISFATTA
                RISPOSTA: <predicted activity>
            
            11. If the query correctly implements the task but returns no
                rows because no matching training trace or next activity
                exists, return:
            
                ESITO: SODDISFATTA
                RISPOSTA: UNKNOWN
            
            12. Return DA_RIGENERARE only when you can identify a concrete
                logical or SQL error in the query.
            
            When a correction is required, the feedback MUST be in English.
            Explain the specific error and how to correct it. Do not invent
            errors or request changes to a query that is already logically
            correct.
            
            For requests other than next-activity prediction, check whether
            the query correctly applies the requested filters, columns,
            conditions, and aggregations.
            
            Return only the specified output format. Do not include Markdown
            or additional commentary.
            """
        ),
        (
            "human",
            """
            Original user request:
            {richiesta}
            
            Executed SQL query:
            {query}
            
            Raw query results:
            {risultati_query}
            
            Evaluate the query according to the rules above.
            """
        )
    ])


def esegui_e_valuta(richiesta_utente: str, query: str) -> dict:
    """
    Executes `query` against the shared database connection (utils.DB) and
    evaluates whether its results satisfy richiesta_utente.

    Returns a dict with keys:
        valutazione     - "SODDISFATTA" | "DA_RIGENERARE"
        risultati_query - raw result of the execution, or None if it failed
        risposta_finale - final natural-language answer (set only when SODDISFATTA)
        feedback        - explanation of what's wrong, for Agent 2's next
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
            "risultati_query": _riassumi_risultati(risultati),
        }
    )

    # Debug: show the raw output returned by the LLM
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


# Initialization of model, prompt and chain
LLM = crea_llm(MODEL_NAME)

PROMPT_VALUTAZIONE = get_prompt_valutazione_risultati()
CHAIN_VALUTAZIONE = PROMPT_VALUTAZIONE | LLM | StrOutputParser()