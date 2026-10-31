# Agent 3 - Executes the generated SQL query, checks whether its results
# actually satisfy the user's original request, and either answers or asks
# Agent 2 to regenerate the query.
import ast
import re
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.utils import create_llm, DB

# Text-only instruction model; compatible with the current causal-LM loader.
MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"


def _extract_after_label(text: str, label: str, until_label: Optional[str] = None) -> Optional[str]:
    """
    Return the text following "<label>:" up to the next occurrence of
    "<until_label>:" (if given), or to the end of the string. Return
    None if the label itself is not found.
    """
    uppercase_text = text.upper()
    marker = f"{label.upper()}:"
    start_index = uppercase_text.find(marker)
    if start_index == -1:
        return None

    content_start = start_index + len(marker)

    if until_label is not None:
        end_index = uppercase_text.find(f"{until_label.upper()}:", content_start)
        if end_index != -1:
            return text[content_start:end_index].strip() or None

    return text[content_start:].strip() or None


def _parse_result_evaluation(text: str):
    """
    Parse the English result-evaluation labels and map them to the internal
    status values used by the pipeline.
    """
    clean_text = text.strip()

    if "SATISFIED" in clean_text.upper():
        answer = _extract_after_label(clean_text, "ANSWER")
        return "SATISFIED", (answer or clean_text), None

    feedback = _extract_after_label(clean_text, "FEEDBACK")
    return "REGENERATE", None, (feedback or clean_text)


# Caps on how much of a query result is shown to the evaluation LLM. The FULL
# raw result is still returned to the pipeline (so metrics and callers see
# everything); only the text placed INSIDE THE PROMPT is bounded. Without a
# cap, a query returning tens of thousands of rows (e.g. a SELECT * where a
# COUNT(*) was needed) puts millions of tokens in the prompt and runs the
# GPU out of memory.
MAX_ROWS_IN_PROMPT = 20
MAX_CHARS_IN_PROMPT = 4000


def _summarize_results(results: str) -> str:
    """
    Builds a bounded description of a raw query result (the string returned
    by DB.run, i.e. str() of a list of row tuples) for use inside the prompt:
    total number of rows, number of columns, and only the first
    MAX_ROWS_IN_PROMPT rows. The total row count is deliberately always
    included: it is exactly what lets the evaluator notice that a "how many"
    question got thousands of raw rows back instead of a single number.
    """
    if not results or not results.strip() or results.strip() == "[]":
        return "Total rows returned: 0 (empty result)"

    rows = None
    try:
        value = ast.literal_eval(results)
        if isinstance(value, list):
            rows = value
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        rows = None

    if rows is not None:
        total = len(rows)
        first_row = rows[0] if rows else None
        columns = len(first_row) if isinstance(first_row, tuple) else 1
        preview = str(rows[:MAX_ROWS_IN_PROMPT])
        header_text = f"Total rows returned: {total}; columns per row: {columns}."
    else:
        # Not parseable as a Python literal (e.g. it contains datetime objects):
        # estimate the row count from the tuple separators and cut by characters.
        total = results.count("), (") + 1
        preview = results
        header_text = f"Estimated total rows returned: {total}."

    if total > MAX_ROWS_IN_PROMPT:
        header_text += f" Showing the first {MAX_ROWS_IN_PROMPT} rows:"

    if len(preview) > MAX_CHARS_IN_PROMPT:
        preview = preview[:MAX_CHARS_IN_PROMPT] + " ...[truncated]"

    return f"{header_text}\n{preview}"


def _sql_boolean_value(results: str):
    """Return a bool only for a single-cell SQL result with a boolean value."""
    if not results or not results.strip():
        return None
    try:
        value = ast.literal_eval(results)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        value = results.strip()
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


def _validate_probability_shape(results: str, shape: str) -> Optional[str]:
    """Reject probability SQL that does not return the benchmark's declared shape."""
    try:
        value = ast.literal_eval(results)
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


def _static_boolean_check(request: str, query: str, schema: str) -> Optional[str]:
    """Catch common OCEL boolean SQL mistakes before accepting the LLM judge."""
    sql = re.sub(r"\s+", " ", query).casefold()
    question = request.casefold()

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


def get_result_evaluation_prompt():
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
            
                OUTCOME: SATISFIED
                ANSWER: <predicted activity>
            
            11. If the query correctly implements the task but returns no
                rows because no matching training trace or next activity
                exists, return:
            
                OUTCOME: SATISFIED
                ANSWER: UNKNOWN
            
            12. Return OUTCOME: REGENERATE only when you can identify a concrete
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
            - If it is a scalar query, returns a single numeric result, and the logic aligns with the question, return OUTCOME: SATISFIED.
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
            {request}

            Database schema:
            {schema}
            
            Executed SQL query:
            {query}
            
            Raw query results:
            {query_results}
            
            Evaluate the query according to the rules above.
            """
        )
    ])


def execute_and_evaluate(user_request: str, query: str, task_info: Optional[str] = None,
                    database_schema: Optional[str] = None) -> dict:
    """
    Executes `query` against the shared database connection (utils.DB) and
    evaluates whether its results satisfy the original user request.

    Returns a dict with keys:
        valutazione     - "SATISFIED" | "REGENERATE"
        risultati_query - raw result of the execution, or None if it failed
        risposta_finale - final natural-language answer (set only when SATISFIED)
        feedback        - explanation of what's wrong, for Agent 2's next
                           attempt (set only when REGENERATE - either a
                           database execution error or a semantic mismatch
                           identified by the evaluation)
    """
    try:
        results = DB.run(query)
    except Exception as error:
        return \
        {
            "valutazione": "REGENERATE",
            "risultati_query": None,
            "risposta_finale": None,
            "feedback": f"The query caused a database execution error: {error}",
        }

    # Structured NSP queries are generated from the workbook's exact case,
    # object type, and ordered prefix by the deterministic SQL template.
    # Do not let the general-purpose judge turn an empty result into a valid
    # prediction: a known case with no following event has no answer.
    if (task_info or "").startswith("TASK=NEXT_ACTIVITY_PREDICTION"):
        parsed = None
        try:
            parsed = ast.literal_eval(results) if results else []
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            pass
        no_answer = not results or not results.strip() or results.strip() in {"[]", "None"}
        if isinstance(parsed, (list, tuple)):
            no_answer = no_answer or not parsed
            if parsed:
                first = parsed[0]
                value = first[0] if isinstance(first, (tuple, list)) and first else first
                no_answer = no_answer or value is None or value == ""
        if no_answer:
            return {
                "valutazione": "REGENERATE",
                "risultati_query": results,
                "risposta_finale": None,
                "feedback": "The structured next-activity query returned no matching next event. Verify the case/object, complete prefix, and ordering.",
            }
        try:
            first = parsed[0]
            answer = first[0] if isinstance(first, (tuple, list)) else first
        except (ValueError, SyntaxError, TypeError, IndexError):
            answer = results
        return {
            "valutazione": "SATISFIED",
            "risultati_query": results,
            "risposta_finale": str(answer),
            "feedback": None,
        }

    if (task_info or "").startswith("TASK=BOOLEAN_QUERY"):
        boolean_value = _sql_boolean_value(results)
        if boolean_value is None:
            return {
                "valutazione": "REGENERATE",
                "risultati_query": results,
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
                "valutazione": "REGENERATE",
                "risultati_query": results,
                "risposta_finale": None,
                "feedback": "The structured probability request is missing its output shape.",
            }
        feedback_probability = _validate_probability_shape(results, shape_match.group(1))
        if feedback_probability:
            return {
                "valutazione": "REGENERATE",
                "risultati_query": results,
                "risposta_finale": None,
                "feedback": feedback_probability,
            }
        feedback_booleano = _static_boolean_check(
            user_request, query, database_schema or ""
        )
        if feedback_booleano:
            return {
                "valutazione": "REGENERATE",
                "risultati_query": results,
                "risposta_finale": None,
                "feedback": feedback_booleano,
            }

    output_grezzo = CHAIN_VALUTAZIONE.invoke(
        {
            "request": user_request,
            "query": query,
            "schema": database_schema or "",
            "query_results": _summarize_results(results),
        }
    )

    # Debug: show the raw output returned by the LLM
    print("\n===== DEBUG AGENT 3 RESULT EVALUATION =====")
    print(repr(output_grezzo))
    print("=============================================\n")

    evaluation, final_answer, feedback = _parse_result_evaluation(output_grezzo)

    return \
    {
        "valutazione": evaluation,
        "risultati_query": results,
        "risposta_finale": final_answer,
        "feedback": feedback,
    }


# Initialization of model, prompt and chain
LLM = create_llm(MODEL_NAME, load_in_4bit=True)

PROMPT_VALUTAZIONE = get_result_evaluation_prompt()
CHAIN_VALUTAZIONE = PROMPT_VALUTAZIONE | LLM | StrOutputParser()
