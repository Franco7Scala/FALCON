# Agent 2 - Generates a SQL query from the information extracted by Agent 1.
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.utils import crea_llm

# "Qwen/Qwen3-Coder-30B-A3B-Instruct"
MODEL_NAME = "Qwen/Qwen3-Coder-30B-A3B-Instruct"


# Utility: cleaning up the model output
def _pulisci_query(testo: str) -> str:
    """
    Removes a markdown code block (```sql ... ```) from the raw LLM output,
    if present, since the model might return one despite the prompt
    instructions.
    """
    testo = testo.strip()
    if "```" in testo:
        righe = testo.split("\n")
        righe = righe[1:] if len(righe) > 1 else righe
        testo = "\n".join(righe)
        testo = testo.rsplit("```", 1)[0]
    return testo.strip()


# Prompts: one for a first generation, one for a correction round

def get_prompt_generazione():
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
              use that table for trace-based next-activity prediction.
            
            NEXT-ACTIVITY PREDICTION RULES:
            
            When the task is NEXT_ACTIVITY_PREDICTION, the extracted prefix
            represents the beginning of a trace and must be matched exactly.
            
            For a prefix containing k activities:
            
            1. Reconstruct the ordered events of each case using ROW_NUMBER()
               OVER (PARTITION BY case_id ORDER BY step), or an equivalent
               valid SQLite approach.
            
            2. Identify candidate cases whose first k activities match ALL
               prefix activities exactly, position by position.
            
            3. Preserve the order and every repetition in the prefix.
               Do not use COUNT(DISTINCT activity) to identify a matching prefix.
               Do not select cases merely because they contain the same activities.
            
            4. Retrieve the activity at position k+1 for each candidate case.
            
            5. Return the activity occurring most frequently at position k+1,
               using GROUP BY activity and ORDER BY COUNT(*) DESC.
               Use activity ASC as a deterministic tie-breaker.
            
            6. If no training trace matches the prefix or no next activity
               exists, the query should return no rows. Never invent a prediction.
            
            7. The query must not use information from development or test cases.
               Use only the data available in the provided database.
            
            RECOMMENDED QUERY STRUCTURE:
            
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
            {informazioni_estratte}
            
            Generate the SQL query for the extracted task.
            """
        )
    ])


def get_prompt_correzione():
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
            
            For a prefix of length k:
            - Match the first k events of each candidate case exactly,
              position by position.
            - Preserve the order and all repeated activities.
            - Retrieve the activity at position k+1.
            - Aggregate those next activities and return the most frequent.
            - Break ties deterministically using activity ASC.
            - Do not use COUNT(DISTINCT activity) as a substitute for
              matching the ordered prefix.
            - Do not exclude a next activity merely because it appears
              in the prefix.
            - Do not use development or test cases to make predictions.
            
            Before returning the SQL, verify internally that:
            - every prefix activity is matched at its correct position;
            - the next activity is retrieved from position k+1;
            - only the next activities of matching training traces
              are aggregated;
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
            {informazioni_estratte}
            
            Previous SQL query:
            {query_precedente}
            
            Validation feedback:
            {feedback}
            
            Repair the query according to the rules above.
            """
        )
    ])



# genera_query: the only function this module exposes
def genera_query(
    informazioni_estratte: str,
    schema_database: str,
    query_precedente: Optional[str] = None,
    feedback: Optional[str] = None,
) -> str:
    """
    Generates a SQL query from the extracted information. If query_precedente
    is given (i.e. the pipeline is asking for a revision after Agent 1
    rejected a previous attempt), the previous query and Agent 1's feedback
    are used to produce a corrected query instead of generating from scratch.
    """
    print("\n========== SQL CORRECTION DEBUG ==========")
    print("Previous query:")
    print(query_precedente)
    
    print("\nFeedback received by Agent 2:")
    print(feedback)
    
    print("==========================================\n")
    
    if query_precedente is None:
        output_grezzo = CHAIN_GENERAZIONE.invoke(
            {
                "schema": schema_database,
                "informazioni_estratte": informazioni_estratte,
            }
        )
    else:
        output_grezzo = CHAIN_CORREZIONE.invoke(
            {
                "schema": schema_database,
                "informazioni_estratte": informazioni_estratte,
                "query_precedente": query_precedente,
                "feedback": feedback or "",
            }
        )

    # Debug: show the raw output returned by the LLM
    print("\n========== SQL GENERATION DEBUG ==========")
    print("Raw LLM output:")
    print(output_grezzo)

    query_finale = _pulisci_query(output_grezzo)

    # Debug: show the final query after cleaning
    print("\nCleaned SQL query:")
    print(query_finale)
    print("==========================================\n")

    print("\n========== SQL CORRECTION RESULT ==========")
    print("Corrected query:")
    print(query_finale)
    
    print(
        "Query unchanged:",
        query_finale.strip() == (query_precedente or "").strip()
    )
    print("==========================================\n")

    return query_finale


# Initialization of model, prompts and chains
LLM = crea_llm(MODEL_NAME, carica_in_4bit=True)

PROMPT_GENERAZIONE = get_prompt_generazione()
CHAIN_GENERAZIONE = PROMPT_GENERAZIONE | LLM | StrOutputParser()

PROMPT_CORREZIONE = get_prompt_correzione()
CHAIN_CORREZIONE = PROMPT_CORREZIONE | LLM | StrOutputParser()