# Agent 1 - Extracts relevant information from the user's request.
from typing import Optional, Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.utils import crea_llm

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


# estrai_informazioni: classifies the request and extracts relevant info
def get_prompt_estrazione() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are the request analysis component of a Text-to-SQL system.
            Your task is to determine whether a user's request can be answered
            using ONLY the database schema provided below.

            DATABASE SCHEMA:
            {schema}

            CLASSIFICATION RULES:

            1. Classify a request as IN_CONTESTO if it can be answered using
               the tables and columns in the provided schema.

            2. Classify a request as FUORI_CONTESTO only if it cannot be answered
               using the available schema, or if it concerns information that is
               fundamentally unrelated to the database.

            3. A request about predicting the next activity in a process trace
               is IN_CONTESTO if the schema provides the information needed to
               reconstruct event sequences, such as case identifiers, activity
               names, and event ordering.
               
            4. In next-activity prediction tasks, the activity names supplied
               by the user are VALUES of an activity column. They do not need
               to be column names or table names.

            5. Never classify a request as FUORI_CONTESTO merely because:
               - the request contains a sequence of activities;
               - an activity appears multiple times;
               - the prefix is long;
               - the request asks for a prediction;
               - the activity names are not explicitly listed in the schema,
                 provided they can be queried as values of an available column.

            INFORMATION EXTRACTION RULES:

            If the request is IN_CONTESTO, extract the information required
            to generate the SQL query.

            For a next-activity prediction request:
            - Identify the task as NEXT_ACTIVITY_PREDICTION.
            - Extract the complete ordered activity prefix.
            - Preserve every activity exactly as provided.
            - Preserve the order of the activities.
            - Preserve repeated activities. Never deduplicate the prefix.
            - Preserve an explicitly named case/object ID and Object Type, if present.
            - If the request names a specific case/object and asks what happened
              next in that case, the objective is the activity immediately after
              the prefix in that object's trace; do not aggregate other objects.
            - Only when no case/object is specified and the request asks for a
              prediction from history, use the most frequent next activity across
              traces matching the complete prefix.
            - Do not confuse the activities contained in the prefix with
              database table or column names.

            Return the result in exactly this format:

            CLASSIFICAZIONE: IN_CONTESTO
            INFORMAZIONI: TASK=NEXT_ACTIVITY_PREDICTION;
            PREFIX=[activity_1, activity_2, ..., activity_k];
            CASE_ID=<explicit case/object id, or NONE>;
            OBJECT_TYPE=<explicit object type, or NONE>;
            OBJECTIVE=<case-specific next event, or most-frequent next activity
            across matching traces, according to the request>.

            For other in-context requests, extract the relevant entities,
            filters, conditions, requested fields, and aggregation requirements.

            If the request is out of context, return exactly:

            CLASSIFICAZIONE: FUORI_CONTESTO

            Do not add explanations before or after the required output.
            Do not invent information that is absent from the user's request.
            """
        ),
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


# Initialization of model, prompt and chain
LLM = crea_llm(MODEL_NAME)

PROMPT_ESTRAZIONE = get_prompt_estrazione()
CHAIN_ESTRAZIONE = PROMPT_ESTRAZIONE | LLM | StrOutputParser()
