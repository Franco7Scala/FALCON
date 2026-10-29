# Agent 3 - Executes the generated SQL query, checks whether its results
# actually satisfy the user's original request, and either answers or asks
# Agent 2 to regenerate the query.
import ast
import re
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.utils import crea_llm, DB

# Text-only instruction model; compatible with the current causal-LM loader.
MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"


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


def _valore_booleano_sql(risultati: str):
    """Return a bool only for a single-cell SQL result with a boolean value."""
    if not risultati or not risultati.strip():
        return None
    try:
        value = ast.literal_eval(risultati)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        value = risultati.strip()
    if isinstance(value, dict):
        if len(value) != 1:
            return None
        value = next(iter(value.values()))
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            return None
        value = value[0]
        if isinstance(value, dict):
            if len(value) != 1:
                return None
            value = next(iter(value.values()))
        elif isinstance(value, (list, tuple)):
            if len(value) != 1:
                return None
            value = value[0]
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1 if value in (0, 1) else None
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"si", "sì", "yes", "true", "1"}:
            return True
        if normalized in {"no", "false", "0"}:
            return False
    return None


def _valida_forma_probabilita(risultati: str, shape: str) -> Optional[str]:
    """Reject probability SQL that does not return the benchmark's declared shape."""
    try:
        value = ast.literal_eval(risultati)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return "The probability query did not return a parseable tabular result."
    if not isinstance(value, list) or not value:
        return "The probability query must return at least one result row."
    rows = [tuple(row) if isinstance(row, (list, tuple)) else (row,) for row in value]
    if shape == "numeric":
        if len(rows) != 1 or len(rows[0]) != 3:
            return "Return one row with exactly numerator, denominator, and probability_percentage."
        numerator, denominator, percentage = rows[0]
        if any(not isinstance(number, (int, float)) or isinstance(number, bool)
               for number in (numerator, denominator, percentage)):
            return "The probability row must contain three numeric values: numerator, denominator, and percentage."
        if denominator <= 0 or numerator < 0 or numerator > denominator:
            return "The probability numerator and denominator are inconsistent; denominator must be positive and numerator between zero and denominator."
        expected_percentage = round(numerator * 100.0 / denominator, 2)
        if abs(float(percentage) - expected_percentage) > 0.011:
            return "probability_percentage must equal ROUND(numerator * 100.0 / denominator, 2)."
        return None
    if any(len(row) != 3 or not isinstance(row[0], str)
           or not isinstance(row[1], (int, float)) or isinstance(row[1], bool)
           or not isinstance(row[2], (int, float)) or isinstance(row[2], bool)
           for row in rows):
        return "Return distribution rows with exactly category_or_activity, count, and probability_percentage."
    if any(row[1] < 0 or row[2] < 0 or row[2] > 100 for row in rows):
        return "Distribution counts must be non-negative and percentages must be between zero and 100."
    if abs(sum(float(row[2]) for row in rows) - 100.0) > 0.11:
        return "The returned probability percentages must sum to 100 (within rounding tolerance)."
    return None


def _controllo_booleano_statico(richiesta: str, query: str, schema: str) -> Optional[str]:
    """Catch common OCEL boolean SQL mistakes before accepting the LLM judge."""
    sql = re.sub(r"\s+", " ", query).casefold()
    question = richiesta.casefold()

    # OCEL event/object identifiers are separate domains. They must be linked
    # through event_object, never by joining their similarly named id columns.
    event_aliases = set(re.findall(r"\b(?:from|join)\s+event\s+(\w+)", sql))
    object_aliases = set(re.findall(r"\bjoin\s+object\s+(\w+)", sql))
    if any(re.search(
        rf"\b{re.escape(event_alias)}\.ocel_id\s*=\s*{re.escape(object_alias)}\.ocel_id\b|"
        rf"\b{re.escape(object_alias)}\.ocel_id\s*=\s*{re.escape(event_alias)}\.ocel_id\b",
        sql,
    ) for event_alias in event_aliases for object_alias in object_aliases):
        return "Event IDs and object IDs are different key domains. Link events to objects through event_object(ocel_event_id, ocel_object_id), then join object on ocel_id."

    # Reject a mapped activity label used as if it were the raw event.ocel_type.
    event_type_by_table = {}
    for table_suffix, raw_type in re.findall(
        r"event_(\w+)\s*\(mapped type\s*'([^']+)'", schema, flags=re.IGNORECASE
    ):
        event_type_by_table[table_suffix.casefold()] = raw_type
        # Reject using an event table's mapped suffix as a raw event.ocel_type.
        if table_suffix.casefold() != raw_type.casefold() and re.search(
            rf"\b\w+\.ocel_type\s*=\s*['\"]{re.escape(table_suffix)}['\"]", sql
        ):
            return (f"The mapped activity label {table_suffix!r} is not the raw event type "
                    f"{raw_type!r}. Use the schema mapping or compare the correct label.")

    # The event subtype table joined for a timestamp must correspond to the
    # raw activity type selected on its parent event row.
    for suffix, subtype_alias, parent_alias in re.findall(
        r"\bjoin\s+event_(\w+)\s+(\w+)\s+on\s+"
        r"(\w+)\.ocel_id\s*=\s*\2\.ocel_id",
        sql,
    ):
        expected_raw = event_type_by_table.get(suffix.casefold())
        if not expected_raw:
            continue
        type_conditions = re.findall(
            rf"\b{re.escape(parent_alias)}\.ocel_type\s*=\s*['\"]([^'\"]+)['\"]",
            sql,
        )
        if type_conditions and all(value.casefold() != expected_raw.casefold() for value in type_conditions):
            return (f"The timestamp is read from event_{suffix}, which belongs to raw event type "
                    f"{expected_raw!r}, but its parent event row is filtered as {type_conditions!r}. "
                    "Join the event subtype table that matches the event type being compared.")

    object_subtype_aliases = set(re.findall(r"\bjoin\s+object_(\w+)\s+(\w+)", sql))
    if len(object_subtype_aliases) >= 2:
        linked_paths = ("object_object" in sql or len(re.findall(r"\bevent_object\s+", sql)) >= 2)
        if not linked_paths:
            return "The query compares multiple distinct object types, but does not establish their relationship. Join them through event_object or object_object to correlate the shared relationship."

    return None


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

            If the request identifies a specific case/object and asks for its
            next event, verify the ordered prefix and position k+1 on that
            object's trace. Do not require aggregation across other cases.
            For a history-based prediction without a specified case/object,
            verify that next activities are aggregated over matching traces
            and the most frequent one is selected.
            
            1. Determine the complete ordered activity prefix from the
               original request.
            
            2. Verify that the SQL query matches the prefix in the correct
               order, preserving every repeated activity.
            
            3. Verify that the query retrieves the activity immediately
               following the complete prefix.
            
            4. For a history-based prediction without a specified case/object,
               verify that the query aggregates next activities across
               matching training traces and selects the most frequent one.
               For a specified case/object, verify that it filters to that
               object's trace and does not aggregate other objects.
            
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
            
            For SCALAR and SET relational queries:
            - If the executed query accesses the correct tables and applies the correct aggregation/filtering, ACCEPT IT.
            - DO NOT reject a query for missing JOINs to generic mapping or parent tables if it correctly extracts the requested data from a specific subtype table directly.
            - Do not reject valid SQL simply because there is an alternative way to write it.
            - If it is a scalar query, returns a single numeric result, and the logic aligns with the question, return ESITO: SODDISFATTA.
            - If it is a set query asking for Top-K, verify it has an ORDER BY ... LIMIT clause and accept it.
            - For SET queries, if the question does not explicitly request a specific order, ignore the ordering of the returned rows when evaluating correctness.

            For a structured TASK=BOOLEAN_QUERY, the SQL must return exactly
            one row and one column with SI/NO or an equivalent boolean value.
            Check the predicate against the request and supplied schema; the
            presence of CASE/EXISTS and a boolean-shaped result is not enough.
            Resolve activity/object labels through mapping tables rather than
            assuming a mapped label equals the raw OCEL type. For a temporal
            condition, verify each timestamp comes from the event_<type>
            table corresponding to that event and that events are linked to
            the same requested object/case. For cross-object claims, verify
            the relationship is established through event_object or
            object_object; unrelated events do not prove a shared scenario.
            Count comparisons must compare separate scalar counts, never
            counts from a Cartesian product. What-if questions must apply the
            stated change before testing the threshold. A raw count, list of
            records, multiple columns, or empty result does not answer yes/no.
            For Italian quantities, a dot before groups of three digits in a
            count/unit context means a thousands separator (2.000 = 2000);
            keep decimal fractions such as 0.5 unchanged.
            
            Return only the specified output format. Do not include Markdown
            or additional commentary.
            """
        ),
        (
            "human",
            """
            Original user request:
            {richiesta}

            Database schema:
            {schema}
            
            Executed SQL query:
            {query}
            
            Raw query results:
            {risultati_query}
            
            Evaluate the query according to the rules above.
            """
        )
    ])


def esegui_e_valuta(richiesta_utente: str, query: str, task_info: Optional[str] = None,
                    schema_database: Optional[str] = None) -> dict:
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

    # Structured NSP queries are generated from the workbook's exact case,
    # object type, and ordered prefix by the deterministic SQL template.
    # Do not let the general-purpose judge turn an empty result into a valid
    # prediction: a known case with no following event has no answer.
    if (task_info or "").startswith("TASK=NEXT_ACTIVITY_PREDICTION"):
        parsed = None
        try:
            parsed = ast.literal_eval(risultati) if risultati else []
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            pass
        no_answer = not risultati or not risultati.strip() or risultati.strip() in {"[]", "None"}
        if isinstance(parsed, (list, tuple)):
            no_answer = no_answer or not parsed
            if parsed:
                first = parsed[0]
                value = first[0] if isinstance(first, (tuple, list)) and first else first
                no_answer = no_answer or value is None or value == ""
        if no_answer:
            return {
                "valutazione": "DA_RIGENERARE",
                "risultati_query": risultati,
                "risposta_finale": None,
                "feedback": "The structured next-activity query returned no matching next event. Verify the case/object, complete prefix, and ordering.",
            }
        try:
            first = parsed[0]
            answer = first[0] if isinstance(first, (tuple, list)) else first
        except (ValueError, SyntaxError, TypeError, IndexError):
            answer = risultati
        return {
            "valutazione": "SODDISFATTA",
            "risultati_query": risultati,
            "risposta_finale": str(answer),
            "feedback": None,
        }

    if (task_info or "").startswith("TASK=BOOLEAN_QUERY"):
        boolean_value = _valore_booleano_sql(risultati)
        if boolean_value is None:
            return {
                "valutazione": "DA_RIGENERARE",
                "risultati_query": risultati,
                "risposta_finale": None,
                "feedback": (
                    "This is a boolean task. Return exactly one row and one column "
                    "containing SI/NO (or a single equivalent 1/0 or true/false). "
                    "Do not return counts, record lists, multiple columns, or no rows."
                ),
            }

    if (task_info or "").startswith("TASK=PROBABILITY_QUERY"):
        shape_match = re.search(r"(?m)^SHAPE=(numeric|distribution)$", task_info or "")
        if not shape_match:
            return {
                "valutazione": "DA_RIGENERARE",
                "risultati_query": risultati,
                "risposta_finale": None,
                "feedback": "The structured probability request is missing its output shape.",
            }
        feedback_probability = _valida_forma_probabilita(risultati, shape_match.group(1))
        if feedback_probability:
            return {
                "valutazione": "DA_RIGENERARE",
                "risultati_query": risultati,
                "risposta_finale": None,
                "feedback": feedback_probability,
            }
        feedback_booleano = _controllo_booleano_statico(
            richiesta_utente, query, schema_database or ""
        )
        if feedback_booleano:
            return {
                "valutazione": "DA_RIGENERARE",
                "risultati_query": risultati,
                "risposta_finale": None,
                "feedback": feedback_booleano,
            }

    output_grezzo = CHAIN_VALUTAZIONE.invoke(
        {
            "richiesta": richiesta_utente,
            "query": query,
            "schema": schema_database or "",
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
LLM = crea_llm(MODEL_NAME, carica_in_4bit=True)

PROMPT_VALUTAZIONE = get_prompt_valutazione_risultati()
CHAIN_VALUTAZIONE = PROMPT_VALUTAZIONE | LLM | StrOutputParser()
