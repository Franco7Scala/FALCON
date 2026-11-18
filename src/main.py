# Main
from src.agents.request_agent import extract_information
from src.agents.query_agent import generate_query
from src.agents.response_agent import execute_and_evaluate
from src.orchestrator.orchestrator import PIPELINE, PipelineState, create_initial_state
from src.utils.utils import DB


def demo_agent_1() -> None:
    """Runs Agent 1's extraction function directly, outside the pipeline."""
    schema = DB.get_table_info()

    sample_requests = [
        "How many events are in the log?",
        "Show all distinct object types in the database",
        "What is the weather like today in Rome?",
    ]

    for request in sample_requests:
        classification, information = extract_information(request, schema)
        print(f"\nRequest: {request}")
        print(f"Classification: {classification}")
        print(f"Extracted information: {information}")


def demo_agent_2() -> None:
    """Runs Agent 2's query generation function directly, outside the pipeline."""
    schema = DB.get_table_info()
    information = (
        "- entity: event\n"
        "- requested fields: COUNT(*)\n"
        "- filter: ocel_type = 'Depart'"
    )

    query = generate_query(information, schema)

    print(f"Extracted information: {information}")
    print(f"Generated query: {query}")


def demo_agent_3() -> None:
    """Runs Agent 3's execution and evaluation function directly, outside the pipeline."""
    request = "How many Depart events are recorded in the log?"
    query = "SELECT COUNT(*) FROM event WHERE ocel_type = 'Depart'"

    result = execute_and_evaluate(request, query, database_schema=DB.get_table_info())

    print(f"Raw results: {result['query_results']}")
    print(f"Final answer: {result['final_answer']}")


def run_interactive_pipeline() -> None:
    """Runs the full pipeline (the only LangGraph graph) interactively from the terminal."""

    print("Pipeline ready. Enter a request (or 'exit' to quit).")

    while True:
        request = input("\n> ").strip()
        if request.lower() in {"exit", "quit"}:
            break

        initial_state: PipelineState = create_initial_state(request)
        result = PIPELINE.invoke(initial_state)

        if result["outcome"] == "ERROR":
            print(f"\n{result['error_message']}")
        else:
            print(f"\n{result['final_answer']}")


if __name__ == "__main__":
    run_interactive_pipeline()
