"""Interactive CLI for querying OCEL 2.0 SQLite databases with FALCON."""
from __future__ import annotations

import argparse
from pathlib import Path

from langchain_community.utilities import SQLDatabase

import src.agents.response_agent as response_module
import src.orchestrator.orchestrator as orchestrator_module
import src.utils.utils as utils_module
from src.dataset.ocel import orchestrator_schema_guidance

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "benchmark" / "databases" / "container_logistics.sqlite"


def _configure_database(db_path: Path) -> str:
    """Bind the pipeline to the selected OCEL SQLite database and return schema guidance."""
    resolved = db_path.resolve()
    db = SQLDatabase.from_uri(f"sqlite:///{resolved.as_posix()}")
    utils_module.DB = db
    response_module.DB = db
    orchestrator_module.DB = db
    return orchestrator_schema_guidance(resolved)


def ask_falcon(question: str, schema_guidance: str, show_sql: bool = False) -> None:
    """Run a single natural-language query through the FALCON pipeline and print the answer."""
    state = orchestrator_module.create_initial_state(question)
    state["database_schema"] += schema_guidance

    result = orchestrator_module.PIPELINE.invoke(state)

    if show_sql and result.get("current_query"):
        print(f"\n[SQL]\n{result['current_query']}")

    if result.get("outcome") == "ERROR":
        print(f"\n{result.get('final_answer') or result.get('error_message')}")
    else:
        print(f"\n{result.get('final_answer')}")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FALCON: Query OCEL 2.0 SQLite databases in natural language."
    )
    parser.add_argument(
        "--database", "--db",
        type=Path,
        default=DEFAULT_DB,
        help="Path to the OCEL 2.0 SQLite database (default: benchmark/databases/container_logistics.sqlite)",
    )
    parser.add_argument(
        "-q", "--query",
        type=str,
        default=None,
        help="Run a single natural-language query and exit (omit for interactive mode).",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=6,
        help="Maximum SQL generation/repair attempts per query (default: 6).",
    )
    parser.add_argument(
        "--show-sql",
        action="store_true",
        help="Print the generated SQL query before the final answer.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    db_path = args.database if args.database.is_absolute() else ROOT / args.database
    if not db_path.is_file():
        raise SystemExit(f"OCEL database not found: {db_path}")
    if args.max_attempts < 1:
        raise SystemExit("--max-attempts must be at least 1")

    orchestrator_module.MAX_ATTEMPTS = args.max_attempts
    schema_guidance = _configure_database(db_path)

    if args.query:
        ask_falcon(args.query.strip(), schema_guidance, show_sql=args.show_sql)
        return 0

    print(f"FALCON ready. Connected to: {db_path}")
    print("Enter a natural-language query about your OCEL log (or 'exit' to quit).")

    while True:
        try:
            request = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not request:
            continue
        if request.lower() in {"exit", "quit"}:
            break
        ask_falcon(request, schema_guidance, show_sql=args.show_sql)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
