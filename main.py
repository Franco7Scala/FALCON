# Main
from agenti.request_agent import extract_information
from agenti.query_agent import generate_query
from agenti.response_agent import execute_and_evaluate
from orchestratore.orchestratore import PIPELINE, PipelineState
from utils.utils import DB


def demo_agent_1() -> None:
    """Runs Agent 1's two functions directly, outside the pipeline."""
    schema = DB.get_table_info()

    sample_requests = \
    [
        "Mostrami i clienti che vanno di nome Mario",
        "Mostrami tutti i prodotti disponibili",
        "Che tempo fa oggi a Roma?",
    ]

    for request in sample_requests:
        classification, information = extract_information(request, schema)
        print(f"\nRequest: {request}")
        print(f"Classification: {classification}")
        print(f"Extracted information: {information}")


def demo_agent_2() -> None:
    """Runs Agent 2's function directly, outside the pipeline."""
    schema = DB.get_table_info()
    information = (
        "- entita': clienti\n"
        "- campi richiesti: nome, cognome\n"
        "- filtro: hanno un ordine con totale > 100"
    )

    query = generate_query(information, schema)

    print(f"Extracted information: {information}")
    print(f"Generated query: {query}")


def demo_agent_3() -> None:
    """Runs Agent 3's function directly, outside the pipeline."""
    request = "Mostrami i clienti che hanno un ordine con totale superiore a 100"
    query = (
        "SELECT c.nome, c.cognome FROM clienti c "
        "JOIN ordini o ON o.cliente_id = c.id "
        "WHERE o.totale > 100"
    )

    result = execute_and_evaluate(request, query)

    print(f"Raw results: {result['risultati_query']}")
    print(f"Final answer: {result['risposta_finale']}")


def run_interactive_pipeline() -> None:
    """Runs the full pipeline (the only LangGraph graph) interactively from the terminal."""

    print("Pipeline ready. Enter a request (or 'exit' to quit).")

    while True:
        request = input("\n> ").strip()
        if request.lower() in {"exit", "quit"}:
            break

        initial_state: PipelineState = \
        {
            "user_request": request,
            "database_schema": DB.get_table_info(),
            "classification": None,
            "extracted_information": None,
            "current_query": None,
            "evaluation_feedback": None,
            "evaluation": None,
            "improvement_attempts": 0,
            "query_results": None,
            "final_answer": None,
            "outcome": None,
            "error_message": None,
        }

        result = PIPELINE.invoke(initial_state)

        if result["outcome"] == "ERROR":
            print(f"\n{result['error_message']}")
        else:
            print(f"\n{result['final_answer']}")


if __name__ == "__main__":
    run_interactive_pipeline()
