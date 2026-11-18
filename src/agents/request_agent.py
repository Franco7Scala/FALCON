# Agent 1 - Extracts relevant information from the user's request.
from typing import Optional, Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.utils.utils import create_llm

# Text-only instruction model; compatible with the current causal-LM loader.
MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"


def _extract_after_label(text: str, label: str, until_label: Optional[str] = None) -> Optional[str]:
    """
    Returns the text following "<label>:" up to the next occurrence of
    "<until_label>:" (if given), or to the end of the string otherwise. Returns
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


def _parse_extraction(text: str) -> Tuple[str, Optional[str]]:
    """
    Parse the extraction prompt output, translating its English labels into
    the existing internal classification values.
    """
    clean_text = text.strip()

    if "OUT_OF_CONTEXT" in clean_text.upper():
        return "OUT_OF_CONTEXT", None

    information = _extract_after_label(clean_text, "INFORMATION")
    return "IN_CONTEXT", information or clean_text


# Classifies the request and extracts relevant information.
def get_extraction_prompt() -> ChatPromptTemplate:
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

            1. Classify a request as IN_CONTEXT if it can be answered using
               the tables and columns in the provided schema.

            2. Classify a request as OUT_OF_CONTEXT only if it cannot be answered
               using the available schema, or if it concerns information that is
               fundamentally unrelated to the database.

            3. A request about predicting the next activity in a process trace
               is IN_CONTEXT if the schema provides the information needed to
               reconstruct event sequences, such as case identifiers, activity
               names, and event ordering.
               
            4. In next-activity prediction tasks, the activity names supplied
               by the user are VALUES of an activity column. They do not need
               to be column names or table names.

            5. Never classify a request as OUT_OF_CONTEXT merely because:
               - the request contains a sequence of activities;
               - an activity appears multiple times;
               - the prefix is long;
               - the request asks for a prediction;
               - the activity names are not explicitly listed in the schema,
                 provided they can be queried as values of an available column.

            INFORMATION EXTRACTION RULES:

            If the request is IN_CONTEXT, extract the information required
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

            CLASSIFICATION: IN_CONTEXT
            INFORMATION: TASK=NEXT_ACTIVITY_PREDICTION;
            PREFIX=[activity_1, activity_2, ..., activity_k];
            CASE_ID=<explicit case/object id, or NONE>;
            OBJECT_TYPE=<explicit object type, or NONE>;
            OBJECTIVE=<case-specific next event, or most-frequent next activity
            across matching traces, according to the request>.

            For other in-context requests, extract the relevant entities,
            filters, conditions, requested fields, and aggregation requirements.

            If the request is out of context, return exactly:

            CLASSIFICATION: OUT_OF_CONTEXT

            Do not add explanations before or after the required output.
            Do not invent information that is absent from the user's request.
            """
        ),
            ("human", "Request: \"{request}\""),
        ]
    )


def extract_information(user_request: str, database_schema: str) -> Tuple[str, Optional[str]]:
    """
    Run Agent 1's extraction step. Return the internal classification and
    extracted information; the latter is None for out-of-context requests.
    """
    raw_output = CHAIN_ESTRAZIONE.invoke(
        {
            "schema": database_schema,
            "request": user_request,
        }
    )
    return _parse_extraction(raw_output)


# Initialization of model, prompt and chain
LLM = create_llm(MODEL_NAME, load_in_4bit=True)

PROMPT_ESTRAZIONE = get_extraction_prompt()
CHAIN_ESTRAZIONE = PROMPT_ESTRAZIONE | LLM | StrOutputParser()
