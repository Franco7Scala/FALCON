# Agent 1 - Extracts relevant information from the user's request and
from typing import Optional, Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.utils import crea_llm

# "Qwen/Qwen3-8B"
MODEL_NAME = "Qwen/Qwen3-0.6B"


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


def _analizza_estrazione(testo: str) -> Tuple[str, Optional[str]]:
    """
    Parses the raw output of the extraction prompt. Expected format:
    "CLASSIFICAZIONE: FUORI_CONTESTO" or "CLASSIFICAZIONE: IN_CONTESTO
    INFORMAZIONI: <relevant information>"
    Returns (classificazione, informazioni_estratte).
    """
    testo_pulito = testo.strip()

    if "FUORI_CONTESTO" in testo_pulito.upper():
        return "FUORI_CONTESTO", None

    informazioni = _estrai_dopo_etichetta(testo_pulito, "INFORMAZIONI")
    return "IN_CONTESTO", informazioni or testo_pulito


def _analizza_valutazione(testo: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Parses the raw output of the query-evaluation prompt. Expected format:
    "VALUTAZIONE: CORRETTA" or "VALUTAZIONE: ERRATA
    FEEDBACK: <what's wrong or missing>
    INFORMAZIONI: <updated/completed extracted information>"
    Returns (valutazione, feedback, informazioni_aggiornate).
    """
    testo_pulito = testo.strip()

    if "CORRETTA" in testo_pulito.upper():
        return "CORRETTA", None, None

    feedback = _estrai_dopo_etichetta(testo_pulito, "FEEDBACK", fino_a="INFORMAZIONI")
    informazioni_aggiornate = _estrai_dopo_etichetta(testo_pulito, "INFORMAZIONI")

    return "ERRATA", feedback or testo_pulito, informazioni_aggiornate


# estrai_informazioni: classifies the request and extracts relevant info
def get_prompt_estrazione() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are a module that analyses a user request meant to query a "
             "specific database, described below by its full schema.\n\n"
             "{schema}\n\n"
             "First, decide whether the request is answerable using ONLY the "
             "tables and columns listed above. It is NOT answerable (and "
             "must be classified as out of context) if:\n"
             "- it is unrelated to the database at all (greetings, small "
             "talk, weather, general knowledge questions, etc.), OR\n"
             "- it asks about an entity, table or column that does not "
             "exist in the schema above (e.g. asking for 'products' when "
             "there is no products table).\n\n"
             "If the request is out of context, respond with exactly:\n"
             "CLASSIFICAZIONE: FUORI_CONTESTO\n\n"
             "Otherwise, respond with:\n"
             "CLASSIFICAZIONE: IN_CONTESTO\n"
             "INFORMAZIONI: <a short list of the relevant entities, filters, "
             "conditions and requested fields found in the request>\n\n"
             "Examples:\n"
             "Schema: table 'clienti' (id, nome, cognome, email, citta); "
             "table 'ordini' (id, cliente_id, data_ordine, totale)\n"
             "Request: \"Mostrami i clienti di nome Mario\"\n"
             "Answer:\nCLASSIFICAZIONE: IN_CONTESTO\nINFORMAZIONI: - entita': clienti\n- filtro: nome = Mario\n\n"
             "Schema: table 'clienti' (id, nome, cognome, email, citta); "
             "table 'ordini' (id, cliente_id, data_ordine, totale)\n"
             "Request: \"Mostrami tutti i prodotti disponibili\"\n"
             "Answer:\nCLASSIFICAZIONE: FUORI_CONTESTO\n\n"
             "Request: \"Che tempo fa oggi a Roma?\"\n"
             "Answer:\nCLASSIFICAZIONE: FUORI_CONTESTO"),
            ("human", "Request: \"{richiesta}\""),
        ]
    )


def estrai_informazioni(richiesta_utente: str, schema_database: str) -> Tuple[str, Optional[str]]:
    """
    Runs Agent 1's extraction step. Returns (classificazione,
    informazioni_estratte): classificazione is "FUORI_CONTESTO" or
    "IN_CONTESTO"; informazioni_estratte is None in the first case.
    """
    output_grezzo = CHAIN_ESTRAZIONE.invoke(
        {
            "schema": schema_database,
            "richiesta": richiesta_utente,
        }
    )
    return _analizza_estrazione(output_grezzo)


# valuta_query: evaluates a SQL query generated by Agent 2
def get_prompt_valutazione() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are a module that evaluates whether a SQL query correctly "
             "answers a user's request, given the database schema below.\n\n"
             "{schema}\n\n"
             "Check that the query: 1) is syntactically valid SQL, 2) only "
             "uses tables/columns that exist in the schema, 3) correctly "
             "reflects the extracted information, 4) produces the kind of "
             "result the user asked for.\n\n"
             "If the query is correct in every respect, respond with exactly:\n"
             "VALUTAZIONE: CORRETTA\n\n"
             "Otherwise, respond with:\n"
             "VALUTAZIONE: ERRATA\n"
             "FEEDBACK: <brief explanation of what is wrong or missing>\n"
             "INFORMAZIONI: <the extracted information, corrected/completed "
             "if needed>"),
            ("human",
             "Original request:\n{richiesta}\n\n"
             "Extracted information:\n{informazioni}\n\n"
             "Query to evaluate:\n{query}"),
        ]
    )


def valuta_query(
    richiesta_utente: str,
    informazioni_estratte: str,
    query_corrente: str,
    schema_database: str,
) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Runs Agent 1's evaluation step. Returns (valutazione, feedback,
    informazioni_aggiornate): valutazione is "CORRETTA" or "ERRATA"; when
    "ERRATA", feedback explains what's wrong and informazioni_aggiornate may
    hold a corrected/completed version of the extracted information (None
    if Agent 1 did not provide one).
    """
    output_grezzo = CHAIN_VALUTAZIONE.invoke(
        {
            "schema": schema_database,
            "richiesta": richiesta_utente,
            "informazioni": informazioni_estratte,
            "query": query_corrente,
        }
    )

    print("\n===== DEBUG AGENT 1 QUERY EVALUATION =====")
    print(repr(output_grezzo))
    print("==========================================\n")
    
    return _analizza_valutazione(output_grezzo)


# Initialization of model, prompts and chains
LLM = crea_llm(MODEL_NAME)

PROMPT_ESTRAZIONE = get_prompt_estrazione()
CHAIN_ESTRAZIONE = PROMPT_ESTRAZIONE | LLM | StrOutputParser()

PROMPT_VALUTAZIONE = get_prompt_valutazione()
CHAIN_VALUTAZIONE = PROMPT_VALUTAZIONE | LLM | StrOutputParser()