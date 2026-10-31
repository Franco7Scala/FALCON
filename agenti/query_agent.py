# Agent 2 - Generates a SQL query from the information extracted by Agent 1.
from typing import Optional
import json
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.utils import create_llm

# "Qwen/Qwen3-Coder-30B-A3B-Instruct"
MODEL_NAME = "Qwen/Qwen3-Coder-30B-A3B-Instruct"


# Utility: cleaning up the model output
def _clean_query(text: str) -> str:
    """
    Removes a markdown code block (```sql ... ```) from the raw LLM output,
    if present, since the model might return one despite the prompt
    instructions.
    """
    text = getattr(text, "content", text)
    if isinstance(text, list):
        text = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in text
        )
    text = str(text).strip()
    if "```" in text:
        rows = text.split("\n")
        rows = rows[1:] if len(rows) > 1 else rows
        text = "\n".join(rows)
        text = text.rsplit("```", 1)[0]
    return text.strip()


def _build_structured_nsp_query(information: str, schema: str) -> Optional[str]:
    """Build positional next-step SQL for case, history, probability, and what-if tasks."""
    if not information.startswith("TASK=NEXT_ACTIVITY_PREDICTION"):
        return None
    fields = {}
    for key in ("OBJECTIVE", "CASE_ID", "OBJECT_TYPE", "PREFIX_JSON"):
        match = re.search(rf"(?m)^{key}=(.*)$", information)
        if not match:
            raise ValueError(f"Structured NSP request is missing {key}")
        fields[key] = match.group(1).strip()
    objective = fields["OBJECTIVE"]
    case_id = fields["CASE_ID"]
    object_type = fields["OBJECT_TYPE"]
    mode_match = re.search(r"(?m)^MATCH_MODE=(CURRENT_ACTIVITY|FULL_PREFIX)$", information)
    match_mode = mode_match.group(1) if mode_match else "FULL_PREFIX"
    try:
        prefix = json.loads(fields["PREFIX_JSON"])
    except (ValueError, TypeError) as exc:
        raise ValueError("Structured NSP prefix is not valid JSON") from exc
    if objective not in {
        "CASE_SPECIFIC_NEXT_EVENT", "MOST_FREQUENT_NEXT_ACTIVITY",
        "NEXT_ACTIVITY_DISTRIBUTION", "WHAT_IF_NEXT_ACTIVITY",
    }:
        raise ValueError(f"Unsupported structured NSP objective: {objective}")
    if not isinstance(prefix, list) or not prefix or any(not isinstance(x, str) or not x for x in prefix):
        raise ValueError("Structured NSP prefix must be a non-empty list of activity names")
    if objective == "CASE_SPECIFIC_NEXT_EVENT" and (not case_id or case_id.upper() == "NONE"):
        raise ValueError("Case-specific NSP requires an explicit case ID")

    excluded_match = re.search(r"(?m)^EXCLUDE_ACTIVITY=(.*)$", information)
    excluded_activity = excluded_match.group(1).strip() if excluded_match else ""
    if objective == "WHAT_IF_NEXT_ACTIVITY" and not excluded_activity:
        raise ValueError("What-if NSP requires the nominal next activity to exclude")

    quote = lambda value: "'" + str(value).replace("'", "''") + "'"
    activity_key = lambda value: re.sub(r"[^a-z0-9]", "", str(value).casefold())
    quote_activity = lambda value: quote(activity_key(value))
    filters_match = re.search(r"(?m)^FILTERS_JSON=(.*)$", information)
    try:
        filters = json.loads(filters_match.group(1)) if filters_match else []
    except (ValueError, TypeError) as exc:
        raise ValueError("Structured NSP filters are not valid JSON") from exc
    if not isinstance(filters, list):
        raise ValueError("Structured NSP filters must be a JSON list")
    object_table = re.sub(r"[^A-Za-z0-9_]", "", object_type)
    if not object_table:
        raise ValueError("Object type cannot be used to resolve object attributes")
    object_conditions = []
    for item in filters:
        if not isinstance(item, dict):
            raise ValueError("Each structured NSP filter must be a JSON object")
        if item.get("kind") == "numeric":
            column = item.get("column")
            operator = item.get("operator")
            value = item.get("value")
            if column not in {"weight", "price"} or operator not in {">", ">=", "<", "<=", "="}:
                raise ValueError("Unsupported numeric NSP filter")
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError("Numeric NSP filter has an invalid value") from exc
            object_conditions.append(
                "EXISTS (SELECT 1 FROM \"object_" + object_table + "\" AS oa "
                "WHERE oa.ocel_id = {case_ref} AND CAST(oa.\"" + column + "\" AS REAL) " + operator + " " + str(numeric_value) + ")"
            )
        elif item.get("kind") == "relation":
            direction = item.get("direction")
            value = quote(item.get("value", ""))
            if direction == "source_is_value":
                relation = f"rel.ocel_source_id = {value} AND rel.ocel_target_id = {{case_ref}}"
            elif direction == "target_is_value":
                relation = f"rel.ocel_source_id = {{case_ref}} AND rel.ocel_target_id = {value}"
            else:
                raise ValueError("Unsupported NSP object-relation filter")
            object_conditions.append(f"EXISTS (SELECT 1 FROM object_object AS rel WHERE {relation})")
        else:
            raise ValueError("Unsupported structured NSP filter kind")

    if re.search(r"CREATE\s+TABLE\s+[\"`]?container_trace\b", schema, re.IGNORECASE):
        conditions = []
        if objective == "CASE_SPECIFIC_NEXT_EVENT":
            conditions.append(f"case_id = {quote(case_id)}")
        conditions.extend(condition.format(case_ref="container_trace.case_id") for condition in object_conditions)
        case_filter = " WHERE " + " AND ".join(conditions) if conditions else ""
        ordered = (
            "ordered_events AS (SELECT case_id, LOWER(REPLACE(REPLACE(REPLACE(activity, ' ', ''), '_', ''), '-', '')) AS activity, step AS pos "
            f"FROM container_trace{case_filter})"
        )
    else:
        mapped_types = re.findall(r"\bevent_([A-Za-z0-9_]+)\s*\(mapped type", schema, re.IGNORECASE)
        if not mapped_types:
            mapped_types = re.findall(r"CREATE\s+TABLE\s+[\"`]?event_([A-Za-z0-9_]+)", schema, re.IGNORECASE)
        mapped_types = list(dict.fromkeys(name for name in mapped_types if name.casefold() != "object"))
        if not mapped_types:
            raise ValueError("Could not find per-event timestamp tables in the supplied OCEL schema")
        timestamp_union = "\nUNION ALL\n".join(
            f'SELECT ocel_id AS event_id, ocel_time AS timestamp FROM "event_{name}"'
            for name in mapped_types
        )
        conditions = []
        if objective == "CASE_SPECIFIC_NEXT_EVENT":
            conditions.append(f"eo.ocel_object_id = {quote(case_id)}")
        conditions.extend(condition.format(case_ref="eo.ocel_object_id") for condition in object_conditions)
        case_filter = " AND " + " AND ".join(conditions) if conditions else ""
        ordered = f'''event_times AS ({timestamp_union}),
ordered_events AS (
    SELECT eo.ocel_object_id AS case_id,
           LOWER(REPLACE(REPLACE(REPLACE(COALESCE(em.ocel_type_map, e.ocel_type), ' ', ''), '_', ''), '-', '')) AS activity,
           ROW_NUMBER() OVER (
               PARTITION BY eo.ocel_object_id ORDER BY et.timestamp, e.ocel_id
           ) AS pos
    FROM event e
    JOIN event_times et ON et.event_id = e.ocel_id
    JOIN event_object eo ON eo.ocel_event_id = e.ocel_id
    JOIN object o ON o.ocel_id = eo.ocel_object_id
    LEFT JOIN event_map_type em ON em.ocel_type = e.ocel_type
    LEFT JOIN object_map_type om ON om.ocel_type = o.ocel_type
    WHERE COALESCE(om.ocel_type_map, o.ocel_type) = {quote(object_type)}{case_filter}
)'''

    if match_mode == "CURRENT_ACTIVITY":
        current_activity = prefix[-1]
        match = (
            "matching_starts AS (SELECT p0.case_id, p0.pos AS start_pos "
            "FROM ordered_events AS p0 "
            f"WHERE p0.activity = {quote_activity(current_activity)})"
        )
        next_pos = 1
    else:
        joins = []
        for position, activity in enumerate(prefix[1:], start=1):
            joins.append(
                f"JOIN ordered_events AS p{position} ON p{position}.case_id = p0.case_id "
                f"AND p{position}.pos = p0.pos + {position} "
                f"AND p{position}.activity = {quote_activity(activity)}"
            )
        match = (
            "matching_starts AS (SELECT p0.case_id, p0.pos AS start_pos "
            "FROM ordered_events AS p0 " + " ".join(joins) +
            f" WHERE p0.activity = {quote_activity(prefix[0])})"
        )
        next_pos = len(prefix)
    transitions = (
        "next_transitions AS (SELECT m.case_id, m.start_pos, next_event.activity "
        "FROM matching_starts AS m JOIN ordered_events AS next_event "
        f"ON next_event.case_id = m.case_id AND next_event.pos = m.start_pos + {next_pos})"
    )
    if objective == "CASE_SPECIFIC_NEXT_EVENT":
        tail = "SELECT activity FROM next_transitions ORDER BY start_pos ASC LIMIT 1"
    elif objective == "NEXT_ACTIVITY_DISTRIBUTION":
        tail = (
            "SELECT activity, COUNT(*) AS count, "
            "ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS probability_percentage "
            "FROM next_transitions GROUP BY activity ORDER BY count DESC, activity ASC"
        )
    else:
        exclusion = f" WHERE activity <> {quote_activity(excluded_activity)}" if objective == "WHAT_IF_NEXT_ACTIVITY" else ""
        tail = (
            "SELECT activity FROM next_transitions" + exclusion +
            " GROUP BY activity ORDER BY COUNT(*) DESC, activity ASC LIMIT 1"
        )
    return f"WITH {ordered},\n{match},\n{transitions}\n{tail};"


def _build_structured_boolean_query(
    information: str,
    schema: str,
    previous_query: Optional[str] = None,
    feedback: Optional[str] = None,
) -> Optional[str]:
    """Generate yes/no SQL under a dedicated contract for benchmark boolean rows."""
    if not information.startswith("TASK=BOOLEAN_QUERY"):
        return None
    match = re.search(r"(?m)^REQUEST_JSON=(.*)$", information)
    if not match:
        raise ValueError("Structured boolean request is missing REQUEST_JSON")
    try:
        request = json.loads(match.group(1))
    except (ValueError, TypeError) as exc:
        raise ValueError("Structured boolean request is not valid JSON") from exc
    question = request.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Structured boolean request has no question")

    repair = ""
    if previous_query:
        repair = f"""

Previous SQL:
{previous_query}

Validation feedback:
{feedback or "The prior result did not satisfy the required one-cell boolean contract."}

Repair the specific issue and return a complete replacement query.
"""
    prompt = f'''You are a SQLite text-to-SQL system answering a Boolean question about an OCEL process database.
Return exactly one executable, read-only SQLite SELECT query and nothing else.
The query must return exactly one row and one column containing the text 'SI' or 'NO'.
Use this result shape: SELECT CASE WHEN (<the exact requested proposition>) THEN 'SI' ELSE 'NO' END AS answer.
Do not return matching records, a raw count, multiple columns, or an empty result when the question asks yes/no. Use EXISTS for existence questions. For comparisons between counts, calculate both counts in scalar subqueries and compare them; do not create a Cartesian join.

Use only the tables, columns, and mapped values in the supplied schema. For event or object labels, consult the mapping tables when the raw OCEL type differs from the label in the question. Event timestamps live in event_<type> tables as ocel_time and join to event by ocel_id; event.ocel_time does not exist. For temporal comparisons, compare the relevant events on the same object/case and use their timestamps, rather than joining unrelated event rows.
In OCEL, event.ocel_id and object.ocel_id belong to different ID domains: link them through event_object(ocel_event_id, ocel_object_id), never by equality-joining the two IDs. Link different objects through object_object and respect its source/target direction and qualifier.
For missing-value questions, check SQL NULL and text sentinels such as 'null', 'none', or an empty string. Do not require ocel_changed_field to equal the attribute name unless the question asks about a specific change record; an object's attribute snapshot can contain its current value on a different row.

The question is in Italian. Interpret Italian numeric notation carefully: in a quantity/count context, a dot followed by groups of three digits (for example 2.000 or 1.500 unità) is a thousands separator, so use 2000 or 1500. A decimal fraction such as 0.5 hours remains a decimal. Preserve comparison operators and units exactly as asked.

Database schema:
{schema}

Question:
{question}
{repair}
'''
    return _clean_query(LLM.invoke(prompt))


def _build_structured_probability_query(
    information: str,
    schema: str,
    previous_query: Optional[str] = None,
    feedback: Optional[str] = None,
) -> Optional[str]:
    """Generate probability SQL with the correct workbook result shape."""
    if not information.startswith("TASK=PROBABILITY_QUERY"):
        return None
    shape_match = re.search(r"(?m)^SHAPE=(numeric|distribution)$", information)
    request_match = re.search(r"(?m)^REQUEST_JSON=(.*)$", information)
    if not shape_match or not request_match:
        raise ValueError("Structured probability request is missing SHAPE or REQUEST_JSON")
    try:
        request = json.loads(request_match.group(1))
    except (ValueError, TypeError) as exc:
        raise ValueError("Structured probability request is not valid JSON") from exc
    question = request.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Structured probability request has no question")
    if shape_match.group(1) == "numeric":
        contract = (
            "Return exactly one row with three numeric columns: numerator, denominator, probability_percentage. "
            "The percentage is ROUND(numerator * 100.0 / denominator, 2)."
        )
    else:
        contract = (
            "Return the requested categorical distribution with exactly three columns: category_or_activity, count, probability_percentage. "
            "Calculate each percentage over the eligible population stated in the question and round to two decimals."
        )
    repair = ""
    if previous_query:
        repair = f"\nPrevious SQL:\n{previous_query}\nValidation feedback:\n{feedback or 'Repair the probability query.'}\n"
    prompt = f'''You are a SQLite text-to-SQL system for a process database.
Return exactly one read-only SQLite SELECT query and nothing else.
The SQL result shape is mandatory: {contract}
Use the exact event, entity, population, and conditions requested. For object-level probabilities, count distinct objects where needed; do not count joined rows unless events are the stated denominator.
For event ordering or time windows, construct timestamps from the correct event_<type> tables joined by ocel_id. event.ocel_time does not exist. Join temporal events through the same object/case where the question describes a lifecycle or sequence.
For What-If questions, apply the stated hypothetical condition to the relevant population or event before calculating the probability; do not return the baseline probability unchanged unless the condition has no effect.
Use only tables, columns, and mapped values in this schema.

Schema:
{schema}

Question:
{question}
{repair}
'''
    return _clean_query(LLM.invoke(prompt))


def _build_structured_relational_query(
    information: str,
    schema: str,
    previous_query: Optional[str] = None,
    feedback: Optional[str] = None,
) -> Optional[str]:
    """Generate scalar/set SQL for workbook tasks without free-form reclassification."""
    if not information.startswith("TASK=RELATIONAL_QUERY"):
        return None
    category_match = re.search(r"(?m)^CATEGORY=(scalar|set)$", information)
    request_match = re.search(r"(?m)^REQUEST_JSON=(.*)$", information)
    if not category_match or not request_match:
        raise ValueError("Structured relational request is missing CATEGORY or REQUEST_JSON")
    category = category_match.group(1)
    try:
        request = json.loads(request_match.group(1))
    except (ValueError, TypeError) as exc:
        raise ValueError("Structured relational request is not valid JSON") from exc
    question = request.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Structured relational request has no question")

    repair = ""
    if previous_query:
        repair = f"""

Previous SQL:
{previous_query}

Validation feedback:
{feedback or "The previous query did not satisfy the requested result shape or semantics."}

Repair the reported issue and return a complete replacement query.
"""
    if category == "set":
        result_contract = (
            "Return exactly the requested columns and rows. Preserve filters and "
            "ordering when requested; do not aggregate unless the question asks for it. "
            "Use ORDER BY ... LIMIT K if a Top-K or Bottom-K is requested. "
            "Do not use SELECT * unless all columns are explicitly requested."
        )
    else:
        result_contract = (
            "Return exactly one row and one column containing the requested scalar value. "
            "Use the requested aggregation (COUNT, AVG, MAX, SUM, etc.); do not return the underlying matching records."
        )

    prompt = f'''You are a SQLite text-to-SQL system for a process database.
    Return exactly one executable, read-only SQLite SELECT query and nothing else.
    Task category: {category}.
    Result contract: {result_contract}

    CRITICAL RULES:
    1. Keep the query simple, but use the correct OCEL relationships. Never join event.ocel_id directly to object.ocel_id. Use event_object for event-to-object links and object_object for object-to-object links.
    2. Object subtype tables can contain multiple snapshot rows for the same ocel_id. For entity counts and lists, use COUNT(DISTINCT ocel_id) or SELECT DISTINCT. For a current attribute value, select the latest non-null value per object ordered by that subtype table's ocel_time; do not multiply entities by snapshot rows.
    3. Do not treat event_object as a generic package/item containment table. Inspect its qualifier and use object_object with the relevant qualifier/direction for relationships among objects.
    4. Event IDs and object IDs are distinct: use event.ocel_type to select event kinds and event_object.ocel_object_id to identify associated entities. Event subtype tables provide timestamps; event.ocel_time does not exist.
    5. Cast text-valued numeric attributes before arithmetic. Exclude NULL or text sentinels ('null', 'none', '') only when they represent missing values. Never add a > 0 or >= 0 filter unless requested; zero can be a valid value.
    6. For activity names shown in a result, prefer event_map_type.ocel_type (human-readable raw label) over the compact table suffix or ocel_type_map if the latter removes spaces.
    7. Compare event times for the same entity/case when the question describes a sequence or temporal relation, using each event type's timestamp table.
    8. Interpret quantities carefully: a dot followed by groups of three digits in a count/quantity (e.g. 2.000) is a thousands separator; use 2000. Decimal fractions such as 0.5 remain decimals.

    Database schema:
    {schema}

    Question:
    {question}
    
    {repair}
    '''

    return _clean_query(LLM.invoke(prompt))


# Prompts: one for a first generation, one for a correction round

def get_generation_prompt():
    return ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are a Text-to-SQL component specialized in next-activity
            prediction from process event logs.

            Your task is to translate the extracted information into ONE valid
            SQLite query, using exclusively the provided database schema.

            GENERAL RULES:

            - Return only executable SQLite SQL.
            - Do not include explanations, comments, Markdown, or code fences.
            - Use only tables and columns present in the provided schema.
            - Do not invent tables, columns, or database values.
            - The query must answer the user's request, not a different question.
            - If the schema contains a simplified event table named
              container_trace with case_id, activity, and step columns,
              you MUST use that table for trace-based next-activity prediction.
            - For the original OCEL SQLite schema, the event table does not
              contain event timestamps. Never assume event.ocel_time exists.
              Timestamps are stored in each event_<type-map> table's ocel_time
              column and must be joined to event using ocel_id. Reconstruct
              object traces through event_object and object; the case key is
              event_object.ocel_object_id, not event.ocel_id. Use the OCEL
              schema details supplied with the database schema to identify
              the available per-event-type timestamp tables.

            NEXT-ACTIVITY PREDICTION RULES:

            First determine the requested objective from the extracted information:
            - If CASE_ID and OBJECT_TYPE are supplied and the objective is the
              next event in that case, filter to that exact object and return
              the event at position k+1. Do not aggregate other cases.
            - If no case/object is specified and the objective is prediction
              from history, aggregate next activities across matching cases
              and return the most frequent one.

            When the task is NEXT_ACTIVITY_PREDICTION, match the complete
            extracted prefix as a consecutive sequence in the trace. The
            prefix can be the beginning of a trace or a later observed window.

            For a prefix containing k activities:

            1. Reconstruct the ordered events of each case using ROW_NUMBER()
               OVER (PARTITION BY case_id ORDER BY step), or an equivalent
               valid SQLite approach.

            2. Identify candidate cases and start positions where all k
               prefix activities match exactly and consecutively.

            3. Preserve the order and every repetition in the prefix.
               Do not use COUNT(DISTINCT activity) to identify a matching prefix.
               Do not select cases merely because they contain the same activities.

            4. Retrieve the activity immediately after the matched prefix.

            5. For CASE_SPECIFIC_NEXT_EVENT, filter to CASE_ID and return that
               case's next activity without aggregation. For
               MOST_FREQUENT_NEXT_ACTIVITY only, group by next activity and
               order by count descending, then activity ascending.

            6. If no training trace matches the prefix or no next activity
               exists, the query should return no rows. Never invent a prediction.

            7. The query must not use information from development or test cases.
               Use only the data available in the provided database.

            RECOMMENDED STRUCTURE FOR MOST_FREQUENT_NEXT_ACTIVITY:

            WITH ordered_events AS (
                SELECT
                    case_id,
                    activity,
                    ROW_NUMBER() OVER (
                        PARTITION BY case_id
                        ORDER BY step
                    ) AS pos
                FROM container_trace
            ),
            matching_cases AS (
                SELECT case_id
                FROM ordered_events
                GROUP BY case_id
                HAVING
                    MAX(CASE WHEN pos = 1 THEN activity END) = 'FIRST_ACTIVITY'
                    AND
                    MAX(CASE WHEN pos = 2 THEN activity END) = 'SECOND_ACTIVITY'
            ),
            next_activities AS (
                SELECT e.activity
                FROM ordered_events AS e
                JOIN matching_cases AS m
                    ON e.case_id = m.case_id
                WHERE e.pos = PREFIX_LENGTH_PLUS_ONE
            )
            SELECT activity
            FROM next_activities
            GROUP BY activity
            ORDER BY COUNT(*) DESC, activity ASC
            LIMIT 1;

            The query structure above is illustrative. Replace the example
            activities and positions with the actual complete prefix.
            Generate as many positional conditions as the prefix requires.
            Do not leave placeholders in the final SQL.

            Escape string literals correctly for SQLite.
            Always generate a query consistent with the actual schema.
            """
        ),
        (
            "human",
            """
            Database schema:
            {schema}

            Information extracted from the user's request:
            {extracted_information}

            Generate the SQL query for the extracted task.
            """
        )
    ])


def get_correction_prompt():
    return ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are a SQLite SQL query repair component specialized in
            next-activity prediction from process event logs.

            Your task is to repair the previous SQL query using:
            - the database schema;
            - the extracted user request;
            - the previous SQL query;
            - the validation feedback.

            Return ONLY one complete, executable SQLite query.
            Do not include explanations, Markdown, or code fences.

            MANDATORY REPAIR RULES:

            1. Read the validation feedback carefully and identify the exact
               issue that must be fixed.

            2. You MUST address the reported issue. Do not ignore the feedback.

            3. If the feedback identifies a genuine error, modify the SQL
               logic that causes that error. Do not merely reformat the query
               or return the same query unchanged.

            4. Do not make unrelated changes that could introduce new errors.

            5. If the feedback is contradictory, or claims that a repeated
               activity is invalid merely because it already appears in the
               prefix, follow the original task semantics:
               repeated activities are valid, and the next activity may equal
               an activity already present in the prefix.

            6. Use only the tables and columns in the supplied schema.

            NEXT-ACTIVITY PREDICTION:

            Respect the objective in the extracted information: when CASE_ID
            and OBJECT_TYPE identify a specific object, return that object's
            next event and do not aggregate other objects. Aggregate only for
            a history-based prediction without a specified object.

            For the original OCEL schema (when container_trace is absent):
            - Never use e.ocel_time on the event table.
            - Build event timestamps by UNION ALL over the listed
              event_<type-map> tables, then join the result to event by ocel_id.
            - Join event_object and object and partition trace positions by
              ocel_object_id; event IDs are not case IDs.
            - Use only table and column names shown in the supplied schema and
              OCEL schema details.

            For a prefix of length k:
            - Match the complete prefix exactly, in order, and consecutively
              at any start position in the case trace.
            - Preserve the order and all repeated activities.
            - Retrieve the activity immediately after the matching sequence.
            - Aggregate and rank next activities only for the history-based
              objective; break ties deterministically using activity ASC.
            - Do not use COUNT(DISTINCT activity) as a substitute for
              matching the ordered prefix.
            - Do not exclude a next activity merely because it appears
              in the prefix.
            - Do not use development or test cases to make predictions.

            Before returning the SQL, verify internally that:
            - every prefix activity is matched at its correct position;
            - the next activity is retrieved from position k+1;
            - aggregation is used only for the history-based objective;
            - the query is valid SQLite.

            If the previous query already implements the request correctly,
            do not introduce arbitrary changes simply to produce a different
            query. However, if the feedback identifies a concrete error,
            you must fix that error.

            Return only the final SQL query.
            """
        ),
        (
            "human",
            """
            Database schema:
            {schema}

            Extracted information:
            {extracted_information}

            Previous SQL query:
            {previous_query}

            Validation feedback:
            {feedback}

            Repair the query according to the rules above.
            """
        )
    ])



# Main query-generation entry point exposed by this module.
def generate_query(
    extracted_information: str,
    database_schema: str,
    previous_query: Optional[str] = None,
    feedback: Optional[str] = None,
) -> str:
    """
    Generates a SQL query from the extracted information. If previous_query
    is given (i.e. the pipeline is asking for a revision after Agent 1
    rejected a previous attempt), the previous query and Agent 1's feedback
    are used to produce a corrected query instead of generating from scratch.
    """
    boolean_query = _build_structured_boolean_query(
        extracted_information, database_schema, previous_query, feedback,
    )
    if boolean_query is not None:
        print("Structured boolean SQL generated with a one-cell SI/NO contract.")
        return boolean_query

    probability_query = _build_structured_probability_query(
        extracted_information, database_schema, previous_query, feedback,
    )
    if probability_query is not None:
        print("Structured probability SQL generated with the workbook result shape.")
        return probability_query

    relational_query = _build_structured_relational_query(
        extracted_information, database_schema, previous_query, feedback,
    )
    if relational_query is not None:
        print("Structured scalar/set SQL generated from workbook task category.")
        return relational_query

    structured_query = _build_structured_nsp_query(extracted_information, database_schema)
    if structured_query is not None:
        print("Structured NSP query generated from workbook task fields.")
        return structured_query

    print("\n========== SQL CORRECTION DEBUG ==========")
    print("Previous query:")
    print(previous_query)

    print("\nFeedback received by Agent 2:")
    print(feedback)

    print("==========================================\n")

    if previous_query is None:
        output_grezzo = CHAIN_GENERAZIONE.invoke(
            {
                "schema": database_schema,
                "extracted_information": extracted_information,
            }
        )
    else:
        output_grezzo = CHAIN_CORREZIONE.invoke(
            {
                "schema": database_schema,
                "extracted_information": extracted_information,
                "previous_query": previous_query,
                "feedback": feedback or "",
            }
        )

    # Debug: show the raw output returned by the LLM
    print("\n========== SQL GENERATION DEBUG ==========")
    print("Raw LLM output:")
    print(output_grezzo)

    query_finale = _clean_query(output_grezzo)

    # Debug: show the final query after cleaning
    print("\nCleaned SQL query:")
    print(query_finale)
    print("==========================================\n")

    print("\n========== SQL CORRECTION RESULT ==========")
    print("Corrected query:")
    print(query_finale)

    print(
        "Query unchanged:",
        query_finale.strip() == (previous_query or "").strip()
    )
    print("==========================================\n")

    return query_finale


# Initialization of model, prompts and chains
LLM = create_llm(MODEL_NAME, load_in_4bit=True)

PROMPT_GENERAZIONE = get_generation_prompt()
CHAIN_GENERAZIONE = PROMPT_GENERAZIONE | LLM | StrOutputParser()

PROMPT_CORREZIONE = get_correction_prompt()
CHAIN_CORREZIONE = PROMPT_CORREZIONE | LLM | StrOutputParser()
