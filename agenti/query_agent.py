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
def get_prompt_generazione() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are a module that translates information extracted from a "
             "natural language request into valid SQL queries, based "
             "exclusively on the provided database schema. "
             "Respond ONLY with the SQL query, without comments, explanations, "
             "or Markdown code blocks."),
            ("human",
             "Database schema:\n{schema}\n\n"
             "Information extracted from the user's request:\n{informazioni_estratte}\n\n"
             "Generate the corresponding SQL query."),
        ]
    )


def get_prompt_correzione() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are a module that corrects an SQL query that is not yet "
             "correct, based on the received feedback, the updated extracted "
             "information, and the database schema. "
             "Respond ONLY with the corrected SQL query, without comments, "
             "explanations, or Markdown code blocks."),
            ("human",
             "Database schema:\n{schema}\n\n"
             "Information extracted from the user's request:\n{informazioni_estratte}\n\n"
             "Previous query:\n{query_precedente}\n\n"
             "Problem encountered:\n{feedback}\n\n"
             "Provide the corrected query."),
        ]
    )


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

    return query_finale


# Initialization of model, prompts and chains
LLM = crea_llm(MODEL_NAME, carica_in_4bit=True)

PROMPT_GENERAZIONE = get_prompt_generazione()
CHAIN_GENERAZIONE = PROMPT_GENERAZIONE | LLM | StrOutputParser()

PROMPT_CORREZIONE = get_prompt_correzione()
CHAIN_CORREZIONE = PROMPT_CORREZIONE | LLM | StrOutputParser()