"""Execution adapters for the repository orchestrator, PMAgent, and PM4Py LLM."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import ast
import json
import re
import time

from .datasets import QueryCase
from .generic_sql import run_generic_sql
from .ocel import object_types, TypeFlattenings, orchestrator_schema_guidance
from .pm4py_data import event_object_dataframe


def _metadata_value(case: QueryCase, *names: str) -> Any:
    normalize = lambda value: re.sub(r"[^a-z0-9]", "", str(value or "").casefold())
    wanted = {normalize(name) for name in names}
    for key, value in case.metadata.items():
        if normalize(key) in wanted and value is not None and str(value).strip():
            return value
    return None


def _base(case: QueryCase, system: str) -> dict[str, Any]:
    return {"query_id": case.query_id, "system": system, "category": case.category,
            "question": case.question, "object_type": _metadata_value(case, "Object Type"),
            "expected": case.expected, "result": None, "sql": None, "seconds": None,
            "status": "execution_error", "correct": None, "error": None, "note": None,
            "expected_source": case.expected_source, "pipeline_status": None,
            "agent_decision": None, "agent_error": None, "attempts": None,
            "source_file": case.source_file, "sheet": case.sheet, "source_row": case.source_row}


def _pipeline_fields(out: dict[str, Any], result: dict[str, Any]) -> None:
    """Keep query execution separate from the pipeline's semantic acceptance decision."""
    rows = result.get("risultati_query")
    classification = str(result.get("classificazione") or "").upper()
    decision = result.get("valutazione")
    out.update(result=rows, sql=result.get("query_corrente"), pipeline_status=result.get("esito"),
               agent_decision=decision, agent_error=result.get("messaggio_errore"),
               attempts=result.get("tentativi_miglioramento"))
    if rows is not None:
        out["status"] = "success"
        if result.get("esito") != "successo":
            if str(result.get("informazioni_estratte") or "").startswith("TASK=NEXT_ACTIVITY_PREDICTION"):
                out["note"] = "Query eseguita ma senza un'attività successiva corrispondente; valutata contro l'atteso come risultato vuoto."
            else:
                out["note"] = "La query ha restituito un risultato, ma il pipeline non l'ha accettato come risposta finale; valutata comunque contro l'atteso."
    elif classification == "FUORI_CONTESTO":
        out["status"] = "out_of_context"
        out["error"] = result.get("messaggio_errore")
    else:
        out["status"] = "execution_error"
        out["error"] = result.get("messaggio_errore")


def run_orchestrator(case: QueryCase, database: Path | None = None, max_attempts: int = 3) -> dict[str, Any]:
    out = _base(case, "orchestrator")
    started = time.perf_counter()
    try:
        from langchain_community.utilities import SQLDatabase
        import utils.utils as utils_module
        import agenti.response_agent as response_module
        import orchestratore.orchestratore as pipeline_module

        pipeline_module.MAX_TENTATIVI = max_attempts

        database = database or Path(__file__).resolve().parents[1] / "databases" / "container_logistics.sqlite"
        db = SQLDatabase.from_uri(f"sqlite:///{database.resolve().as_posix()}")
        # Keep every imported DB reference in the existing pipeline on the selected original OCEL DB.
        utils_module.DB = db
        response_module.DB = db
        pipeline_module.DB = db
        state = pipeline_module.crea_stato_iniziale(case.question)
        state["schema_database"] += orchestrator_schema_guidance(database)
        specification = _nsp_specification(case)
        if not specification:
            specification = _relational_specification(case)
        if specification:
            state["classificazione"] = "IN_CONTESTO"
            state["informazioni_estratte"] = specification
        # Graph execution API and state keys are the repository's actual interface.
        result = pipeline_module.PIPELINE.invoke(state, config={"recursion_limit": 30})
        _pipeline_fields(out, result)
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    out["seconds"] = time.perf_counter() - started
    return out


def _parse_prefix(case: QueryCase) -> list[str] | None:
    raw = _metadata_value(case, "Traccia / Prefisso (Case-Based)", "Trace / Prefix", "Trace Prefix")
    if not raw:
        return None
    match = re.search(r"Trace Prefix:\s*\[(.*?)\]", str(raw), re.I)
    if not match:
        match = re.search(r"prefisso.*?\[(.*?)\]", str(raw), re.I)
    if not match:
        match = re.search(r"(?:prefix|prefisso)\s*:\s*\[(.*?)\]", str(raw), re.I)
    return [part.strip() for part in re.split(r"\s*(?:->|>)\s*", match.group(1)) if part.strip()] if match else None


def _nsp_specification(case: QueryCase) -> str | None:
    """Serialize workbook NSP fields so systems use the same complete prefix and objective."""
    if not case.category.startswith("next_step"):
        return None
    prefix = _parse_prefix(case)
    object_type = _metadata_value(case, "Object Type")
    if not prefix or not object_type:
        return None
    case_id = _metadata_value(case, "Case ID / Target", "Case ID", "Target") if case.category == "next_step_case" else None
    objective = "CASE_SPECIFIC_NEXT_EVENT" if case.category == "next_step_case" else "MOST_FREQUENT_NEXT_ACTIVITY"
    target = str(_metadata_value(case, "Case ID / Target", "Case ID", "Target") or "").casefold()
    match_mode = "CURRENT_ACTIVITY" if case.category == "next_step_case" or target.startswith("pattern") else "FULL_PREFIX"
    return "\n".join((
        "TASK=NEXT_ACTIVITY_PREDICTION",
        f"OBJECTIVE={objective}",
        f"MATCH_MODE={match_mode}",
        f"CASE_ID={case_id or 'NONE'}",
        f"OBJECT_TYPE={object_type}",
        f"PREFIX_JSON={json.dumps(prefix, ensure_ascii=False)}",
    ))


def _relational_specification(case: QueryCase) -> str | None:
    """Pass workbook-declared relational tasks without unreliable reclassification."""
    category = case.category.strip().casefold()
    if category not in {"boolean", "scalar", "set"}:
        return None
    request = json.dumps({"question": case.question}, ensure_ascii=False)
    task = "BOOLEAN_QUERY" if category == "boolean" else "RELATIONAL_QUERY"
    return f"TASK={task}\nCATEGORY={category}\nREQUEST_JSON={request}"


def _parse_expected(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    return value


class PMAgentAdapter:
    """Reuse `next_step_data` type-aware trace extraction and the existing LangGraph pipeline."""
    def __init__(self, database: Path, timeout: float = 300, max_attempts: int = 3):
        self.database = database
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.flattenings = TypeFlattenings(database)
        self.valid_types = object_types(database)
        self._modules = None

    def _load(self):
        if self._modules is not None:
            return self._modules
        # Imports initialize the locally configured HuggingFace models, as in next_step.py.
        from langchain_community.utilities import SQLDatabase
        import agenti.response_agent as response_module
        import orchestratore.orchestratore as pipeline_module
        from experiments.next_step import DBConTimeout
        self._modules = (SQLDatabase, response_module, pipeline_module, DBConTimeout)
        return self._modules

    def run(self, case: QueryCase) -> dict[str, Any]:
        out = _base(case, "PMAgent")
        if case.category in {"scalar", "set", "boolean"}:
            return run_generic_sql(case, self.database, "PMAgent", self.max_attempts, self.timeout)
        if not case.category.startswith("next_step"):
            out.update(status="out_of_scope", note=f"PMAgent adapter has no implementation for category {case.category!r}.")
            return out
        object_type = _metadata_value(case, "Object Type")
        prefix = _parse_prefix(case)
        if not object_type or not prefix:
            out.update(status="unsupported", note="next_step.py supports Next-Step Prediction; this row lacks a supported NSP category, Object Type, or parseable trace prefix.")
            return out
        if str(object_type) not in self.valid_types:
            out.update(status="unsupported", note=f"Object Type {object_type!r} is not a case-object type in object_map_type and cannot be flattened as an object trace.")
            return out
        started = time.perf_counter()
        try:
            flat_db = self.flattenings.for_type(str(object_type))
            SQLDatabase, response_module, pipeline_module, DBConTimeout = self._load()
            pipeline_module.MAX_TENTATIVI = self.max_attempts
            real_db = SQLDatabase.from_uri(f"sqlite:///{flat_db.as_posix()}")
            db = DBConTimeout(real_db, flat_db, int(self.timeout))
            response_module.DB = db
            pipeline_module.DB = db
            state = {
                "richiesta_utente": case.question, "schema_database": db.get_table_info(),
                "classificazione": None, "informazioni_estratte": None, "query_corrente": None,
                "valutazione": None, "feedback_valutazione": None, "tentativi_miglioramento": 0,
                "risultati_query": None, "risposta_finale": None, "esito": None, "messaggio_errore": None,
            }
            specification = _nsp_specification(case)
            if specification:
                state["classificazione"] = "IN_CONTESTO"
                state["informazioni_estratte"] = specification
            result = pipeline_module.PIPELINE.invoke(state, config={"recursion_limit": 30})
            _pipeline_fields(out, result)
        except Exception as exc:
            out["error"] = f"{type(exc).__name__}: {exc}"
        out["seconds"] = time.perf_counter() - started
        return out

    def close(self):
        self.flattenings.close()


class PM4PyLLMAdapter:
    """PM4Py nlp_to_log_query is used only for compatible NSP rows on an event-object DataFrame."""
    def __init__(self, database: Path, max_attempts: int = 3):
        self.database = database
        self.max_attempts = max_attempts
        self._pm4py = None
        self._executor_llm = None
        self._frame_cache: dict[str, Any] = {}
        self._last_sql: str | None = None
        self.valid_types = object_types(database)

    def _load(self):
        if self._pm4py is None:
            import pm4py
            self._pm4py = pm4py
        return self._pm4py

    def _local_executor(self, prompt: str, **kwargs) -> str:
        """Use the project's quantized local Qwen coder as PM4Py's prompt executor."""
        if self._executor_llm is None:
            # This is the same local model configured by agenti.query_agent.LLM.
            from agenti.query_agent import LLM
            self._executor_llm = LLM
        response = self._executor_llm.invoke(prompt)
        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        return str(content)

    def _frame(self, object_type: str):
        if object_type in self._frame_cache:
            return self._frame_cache[object_type]
        frame = event_object_dataframe(self.database, object_type)
        import pandas as pd
        for column in ("event_id", "case:concept:name", "concept:name", "object_id", "object_type", "qualifier"):
            if column in frame:
                frame[column] = frame[column].astype("string")
        if "time:timestamp" in frame:
            frame["time:timestamp"] = pd.to_datetime(frame["time:timestamp"], errors="coerce", utc=True)
        frame = frame.sort_values(["case:concept:name", "time:timestamp", "event_id"], kind="stable").reset_index(drop=True)
        self._frame_cache[object_type] = frame
        return frame

    def run(self, case: QueryCase) -> dict[str, Any]:
        out = _base(case, "PM4Py LLM")
        if case.category in {"scalar", "set", "boolean"}:
            return run_generic_sql(case, self.database, "PM4Py LLM", max_attempts=self.max_attempts)
        if not case.category.startswith("next_step"):
            out.update(status="out_of_scope", note=f"PM4Py adapter has no implementation for category {case.category!r}.")
            return out
        object_type = _metadata_value(case, "Object Type")
        prefix = _parse_prefix(case)
        if not object_type or not prefix:
            out.update(status="unsupported", note="PM4Py requires a supported NSP row with Object Type and a parseable trace prefix.")
            return out
        if str(object_type) not in self.valid_types:
            out.update(status="unsupported", note=f"Object Type {object_type!r} is not an OCEL object type available in the database.")
            return out
        started = time.perf_counter()
        try:
            pm4py = self._load()
            frame = self._frame(str(object_type))
            spec = _nsp_specification(case)
            objective = "case-specific next event" if case.category == "next_step_case" else "most frequent next activity across matching traces"
            case_id = _metadata_value(case, "Case ID / Target", "Case ID", "Target") if case.category == "next_step_case" else None
            match_mode = "CURRENT_ACTIVITY" if "MATCH_MODE=CURRENT_ACTIVITY" in (spec or "") else "FULL_PREFIX"
            match_instruction = (
                "Use only the final/current activity in the prefix as the match point; do not require earlier prefix activities. "
                "Return the event immediately after that current activity."
                if match_mode == "CURRENT_ACTIVITY" else
                "Match the complete prefix consecutively, preserving order and repetitions, then return the immediately following event."
            )
            history_instruction = (
                "For a historical task, group next activities after matching the current activity and return the most frequent."
                if match_mode == "CURRENT_ACTIVITY" else
                "For a historical task, group next activities after exact full-prefix matches and return the most frequent."
            )
            question = (
                f"Task: return the {objective}.\n"
                f"Object type: {object_type}.\n"
                f"Exact case ID: {case_id or 'not specified'}.\n"
                f"Complete ordered trace prefix, including its final/current activity: {json.dumps(prefix, ensure_ascii=False)}.\n"
                f"{match_instruction} Return only the requested activity. "
                "For a case-specific task, filter to that exact case and do not aggregate. "
                f"{history_instruction} "
                f"Use only the provided event-log columns.\nStructured fields:\n{spec}\nOriginal question: " + case.question
            )
            columns = ", ".join(repr(column) for column in frame.columns)
            dtypes = ", ".join(f"{column}={dtype}" for column, dtype in frame.dtypes.items())
            # execute_query=True uses PM4Py's own SQL safety validation and DuckDB path.
            original_executor = self._local_executor
            self._last_sql = None
            data = None
            last_error = None
            last_failed_sql = None
            for attempt in range(1, self.max_attempts + 1):
                out["attempts"] = attempt
                self._last_sql = None

                def duckdb_executor(prompt: str, **kwargs) -> str:
                    prompt += (f"\nActual DataFrame columns: {columns}. Actual dtypes: {dtypes}. "
                               "SQL dialect: DuckDB. Use only those exact columns; do not invent org:resource or any other attribute. "
                               "Quote column names containing colons with double quotes. "
                               "Build an ordered_events CTE with ROW_NUMBER() PARTITION BY case ID and ORDER BY timestamp, event ID. "
                               "Do not put window functions in WHERE or HAVING; filter their aliases in an outer CTE. "
                               "Match every prefix activity at consecutive positions using self-joins on case ID and position. "
                               "For an exact case, filter to that case and return its next activity without aggregation. "
                               "For history, group only the next activity after a full prefix match and order by count descending, activity ascending. "
                               "Return one next-activity column. Do not use LISTAGG/LIST_AGG or SQL comments; return one read-only query.")
                    if last_error:
                        prompt += (f"\nThe previous SQL was:\n{last_failed_sql}\n"
                                   f"DuckDB rejected it with: {last_error}\n"
                                   "Correct that specific error. Do not repeat the previous query unchanged.")
                    generated = original_executor(prompt, **kwargs)
                    self._last_sql = generated.strip()
                    return generated

                try:
                    data = pm4py.llm.nlp_to_log_query(
                        frame, question, executor=duckdb_executor,
                        obtain_query=True, execute_query=True,
                    )
                    if not hasattr(data, "to_dict"):
                        raise ValueError("PM4Py did not return the documented DataFrame query result")
                    if len(data.columns) != 1:
                        raise ValueError("Next-step query must return exactly one activity column")
                    last_error = None
                    break
                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    last_failed_sql = self._last_sql
                    data = None
            if data is None:
                raise RuntimeError(
                    f"PM4Py/DuckDB query failed after {self.max_attempts} attempts: {last_error}"
                )
            out.update(result=data.to_dict(orient="records"), sql=self._last_sql, status="success")
            out["note"] = "SQL generation used the project's local Qwen model; PM4Py validated and executed it on the DataFrame. PM4Py returns the result DataFrame, not the generated SQL string."
        except Exception as exc:
            out["error"] = f"{type(exc).__name__}: {exc}"
            out["sql"] = self._last_sql
            out["note"] = "Local model, PM4Py query generation, validation, or execution failed; see error and local model configuration."
        out["seconds"] = time.perf_counter() - started
        return out
