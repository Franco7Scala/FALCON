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
    rows = result.get("query_results")
    classification = str(result.get("classification") or "").upper()
    decision = result.get("evaluation")
    out.update(result=rows, sql=result.get("current_query"), pipeline_status=result.get("outcome"),
               agent_decision=decision, agent_error=result.get("error_message"),
               attempts=result.get("improvement_attempts"))
    if rows is not None:
        if isinstance(rows, str) and not rows.strip() and not (result.get("current_query") or "").strip():
            out["status"] = "execution_error"
            out["error"] = result.get("error_message") or "The pipeline produced no SQL query or result."
            out["note"] = "The pipeline exhausted its attempts without producing an executable SQL query."
            return
        out["status"] = "success"
        if result.get("outcome") != "SUCCESS":
            if str(result.get("extracted_information") or "").startswith("TASK=NEXT_ACTIVITY_PREDICTION"):
                out["note"] = "The query ran but found no matching next activity; its empty result was compared with the expected result."
            else:
                out["note"] = "The query returned a result, but the pipeline did not accept it as the final answer; it was still compared with the expected result."
    elif classification == "OUT_OF_CONTEXT":
        out["status"] = "out_of_context"
        out["error"] = result.get("error_message")
    else:
        out["status"] = "execution_error"
        out["error"] = result.get("error_message")


def run_orchestrator(case: QueryCase, database: Path | None = None, max_attempts: int = 3) -> dict[str, Any]:
    out = _base(case, "orchestrator")
    started = time.perf_counter()
    try:
        from langchain_community.utilities import SQLDatabase
        import utils.utils as utils_module
        import agenti.response_agent as response_module
        import orchestratore.orchestratore as pipeline_module

        pipeline_module.MAX_ATTEMPTS = max_attempts

        database = database or Path(__file__).resolve().parents[1] / "databases" / "container_logistics.sqlite"
        db = SQLDatabase.from_uri(f"sqlite:///{database.resolve().as_posix()}")
        # Keep every imported DB reference in the existing pipeline on the selected original OCEL DB.
        utils_module.DB = db
        response_module.DB = db
        pipeline_module.DB = db
        state = pipeline_module.create_initial_state(case.question)
        state["database_schema"] += orchestrator_schema_guidance(database)
        specification = _nsp_specification(case)
        if not specification:
            specification = _relational_specification(case)
        if specification:
            state["classification"] = "IN_CONTEXT"
            state["extracted_information"] = specification
        # Graph execution API and state keys are the repository's actual interface.
        result = pipeline_module.PIPELINE.invoke(state, config={"recursion_limit": 30})
        _pipeline_fields(out, result)
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    out["seconds"] = time.perf_counter() - started
    return out


def _parse_prefix(case: QueryCase) -> list[str] | None:
    raw = _metadata_value(case, "Traccia / Prefisso (Case-Based)", "Trace / Prefix", "Trace Prefix")
    text = str(raw or case.question)
    activities = (
        "place order", "confirm order", "pay order", "payment reminder",
        "create package", "send package", "package delivered", "failed delivery",
        "item out of stock", "reorder item", "pick item",
        "register customer order", "order empty containers", "pick up empty container",
        "collect goods", "weigh", "place in stock", "bring to loading bay",
        "load to vehicle", "load truck", "create transport document", "depart",
        "drive to terminal", "book vehicles", "reschedule container",
    )
    match = re.search(r"\[([^\]]+(?:->|→)[^\]]+)\]", text)
    if not match:
        match = re.search(r"(?:Trace Prefix|prefix|prefisso)\s*:\s*\[([^\]]+)\]", text, re.I)
    if match:
        parts = [part.strip(" '’\"") for part in re.split(r"\s*(?:->|→|>)\s*", match.group(1)) if part.strip()]
        preceding = text[:match.start()].casefold()
        described = []
        for activity in activities:
            for occurrence in re.finditer(re.escape(activity), preceding):
                context = preceding[max(0, occurrence.start() - 70):occurrence.end()]
                if re.search(r"\b(?:experienced|completed|triggered|executed|received|after)\b", context):
                    described.append((occurrence.start(), activity))
        if described:
            last_activity = max(described)[1]
            if not parts or parts[-1].casefold() != last_activity:
                parts.append(last_activity)
        return parts or None

    # This workbook has no dedicated prefix column. Read activity labels from
    # the natural-language prompt, but only accept labels from the supported
    # event vocabulary so names and object IDs cannot become trace activities.
    found = []
    question = case.question.casefold()
    for activity in activities:
        for match in re.finditer(re.escape(activity), question):
            found.append((match.start(), activity))
    found.sort()
    if not found:
        return None
    if case.category == "next_step_what_if" and re.search(r"\bconfirmed order\b", question):
        return ["confirm order"]
    immediately_after = re.search(r"immediately after", question)
    if immediately_after:
        before = [(pos, name) for pos, name in found if pos < immediately_after.start()]
        after = [(pos, name) for pos, name in found if pos >= immediately_after.end()]
        if before and after:
            return [after[0][1], before[-1][1]]
    repeated = re.search(r"\btwo consecutive\s+['\"]([^'\"]+)['\"]", question, re.I)
    if repeated:
        activity = repeated.group(1).casefold()
        if activity in activities:
            return [activity, activity]
    # For What-If prompts, the first activity is the observed event; the later
    # activity is the nominal transition being excluded by the hypothetical.
    if case.category in {"next_step_what_if", "next_step_probability"}:
        return [found[0][1]]
    return [found[-1][1]]


def _nsp_specification(case: QueryCase) -> str | None:
    """Serialize NSP intent from explicit workbook fields or its question text."""
    supported = {"next_step_case", "next_step_history", "next_step_probability", "next_step_what_if"}
    if case.category not in supported:
        return None
    if case.category == "next_step_probability":
        question = case.question.casefold()
        is_transition_distribution = bool(
            re.search(r"probability distribution.{0,120}(?:next|following|immediately following).{0,80}activity", question)
            or re.search(r"probability.{0,80}next event.{0,120}\bversus\b", question)
        )
        # NSP-PROB also includes lifecycle, temporal, attribute, and What-If
        # probabilities. Those are scalar/set SQL tasks, not transition
        # distributions, so leave them to the relational SQL adapters.
        if not is_transition_distribution:
            return None
    prefix = _parse_prefix(case)
    object_type = _metadata_value(case, "Object Type")
    case_id = _metadata_value(case, "Case ID / Target", "Case ID", "Target")
    if not object_type:
        explicit_type = re.search(r"\b(?:a|an|the)\s+(customer|package|order|item)\s*\(['\"]([^'\"]+)['\"]\)", case.question, re.I)
        if explicit_type:
            object_type = explicit_type.group(2)
        if case_id:
            object_type = object_type or {"p": "packages", "o": "orders", "i": "items"}.get(str(case_id).strip()[:1].casefold())
        if not object_type:
            match = re.search(r"\b(?:object type|type)\s*['\"]([^'\"]+)['\"]", case.question, re.I)
            if match:
                object_type = match.group(1)
        if not object_type:
            question = case.question.casefold()
            type_mentions = [
                (match.start(), plural)
                for singular, plural in (("package", "packages"), ("order", "orders"), ("item", "items"), ("customer", "customers"))
                for match in re.finditer(rf"\b{singular}\b", question)
            ]
            if type_mentions:
                object_type = min(type_mentions)[1]
    if not prefix or not object_type:
        return None
    if case.category == "next_step_case" and not case_id:
        match = re.search(r"\b(?:package|order|item)\s+['\"]([^'\"]+)['\"]", case.question, re.I)
        if match:
            case_id = match.group(1)
    if case.category == "next_step_case" and not case_id:
        return None

    objective = {
        "next_step_case": "CASE_SPECIFIC_NEXT_EVENT",
        "next_step_history": "MOST_FREQUENT_NEXT_ACTIVITY",
        "next_step_probability": "NEXT_ACTIVITY_DISTRIBUTION",
        "next_step_what_if": "WHAT_IF_NEXT_ACTIVITY",
    }[case.category]
    mode_match = re.search(r"(?m)^MATCH_MODE=(CURRENT_ACTIVITY|FULL_PREFIX)$", str(_metadata_value(case, "Match Mode") or ""))
    match_mode = mode_match.group(1) if mode_match else ("FULL_PREFIX" if len(prefix) > 1 else "CURRENT_ACTIVITY")

    fields = [
        "TASK=NEXT_ACTIVITY_PREDICTION",
        f"OBJECTIVE={objective}",
        f"MATCH_MODE={match_mode}",
        f"CASE_ID={case_id or 'NONE'}",
        f"OBJECT_TYPE={object_type}",
        f"PREFIX_JSON={json.dumps(prefix, ensure_ascii=False)}",
    ]
    filters = []
    numeric = re.search(r"\b(weight|price)\s*(>=|<=|>|<|=)\s*([0-9]+(?:\.[0-9]+)?)", case.question, re.I)
    if numeric:
        filters.append({"kind": "numeric", "column": numeric.group(1).casefold(),
                        "operator": numeric.group(2), "value": float(numeric.group(3))})
    relation_patterns = (
        (r"\bplaced by customer\s+['\"]([^'\"]+)['\"]", "source_is_value"),
        (r"\bhandled by employee\s+['\"]([^'\"]+)['\"]", "target_is_value"),
        (r"\bproduct type\s+['\"]([^'\"]+)['\"]", "target_is_value"),
    )
    for pattern, direction in relation_patterns:
        relation = re.search(pattern, case.question, re.I)
        if relation:
            filters.append({"kind": "relation", "direction": direction, "value": relation.group(1)})
            break
    if filters:
        fields.append(f"FILTERS_JSON={json.dumps(filters, ensure_ascii=False)}")
    if case.category == "next_step_what_if":
        current_activity = prefix[-1].casefold()
        occurrences = sorted(
            (match.start(), name)
            for name in ("place order", "confirm order", "pay order", "payment reminder", "create package", "send package", "package delivered", "failed delivery", "item out of stock", "reorder item", "pick item")
            for match in re.finditer(re.escape(name), case.question.casefold())
        )
        excluded = next((name for _, name in occurrences if name != current_activity), "")
        if excluded:
            fields.append(f"EXCLUDE_ACTIVITY={excluded}")
    return "\n".join(fields)


def _spec_value(specification: str | None, key: str) -> str | None:
    if not specification:
        return None
    match = re.search(rf"(?m)^{re.escape(key)}=(.*)$", specification)
    return match.group(1).strip() if match else None


def _relational_specification(case: QueryCase) -> str | None:
    """Pass workbook-declared relational tasks without unreliable reclassification."""
    category = case.category.strip().casefold()
    if category == "next_step_probability":
        expected = case.expected
        if isinstance(expected, str):
            try:
                expected = ast.literal_eval(expected)
            except (ValueError, SyntaxError):
                expected = None
        rows = expected if isinstance(expected, (list, tuple)) else []
        shape = "distribution" if rows and isinstance(rows[0], (list, tuple)) and rows[0] and isinstance(rows[0][0], str) else "numeric"
        request = json.dumps({"question": case.question}, ensure_ascii=False)
        return f"TASK=PROBABILITY_QUERY\nSHAPE={shape}\nREQUEST_JSON={request}"
    sql_category = {
        "next_step_case": "scalar",
        "next_step_history": "scalar",
        "next_step_what_if": "scalar",
    }.get(category, category)
    if sql_category not in {"boolean", "scalar", "set"}:
        return None
    request = json.dumps({"question": case.question}, ensure_ascii=False)
    task = "BOOLEAN_QUERY" if sql_category == "boolean" else "RELATIONAL_QUERY"
    return f"TASK={task}\nCATEGORY={sql_category}\nREQUEST_JSON={request}"


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
        from benchmark.next_step import DatabaseWithTimeout
        self._modules = (SQLDatabase, response_module, pipeline_module, DatabaseWithTimeout)
        return self._modules

    def run(self, case: QueryCase) -> dict[str, Any]:
        out = _base(case, "PMAgent")
        if case.category in {"scalar", "set", "boolean"}:
            return run_generic_sql(case, self.database, "PMAgent", self.max_attempts, self.timeout)
        if not case.category.startswith("next_step"):
            out.update(status="out_of_scope", note=f"PMAgent adapter has no implementation for category {case.category!r}.")
            return out
        specification = _nsp_specification(case)
        if specification is None:
            return run_generic_sql(case, self.database, "PMAgent", self.max_attempts, self.timeout)
        object_type = _metadata_value(case, "Object Type") or _spec_value(specification, "OBJECT_TYPE")
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
            SQLDatabase, response_module, pipeline_module, DatabaseWithTimeout = self._load()
            pipeline_module.MAX_ATTEMPTS = self.max_attempts
            real_db = SQLDatabase.from_uri(f"sqlite:///{flat_db.as_posix()}")
            db = DatabaseWithTimeout(real_db, flat_db, int(self.timeout))
            response_module.DB = db
            pipeline_module.DB = db
            state = {
                "user_request": case.question, "database_schema": db.get_table_info(),
                "classification": None, "extracted_information": None, "current_query": None,
                "evaluation": None, "evaluation_feedback": None, "improvement_attempts": 0,
                "query_results": None, "final_answer": None, "outcome": None, "error_message": None,
            }
            if specification:
                state["classification"] = "IN_CONTEXT"
                state["extracted_information"] = specification
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
        for column in ("event_id", "case:concept:name", "concept:name", "object_id", "object_type", "qualifier", "related_source_ids", "related_target_ids"):
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
        specification = _nsp_specification(case)
        if specification is None:
            return run_generic_sql(case, self.database, "PM4Py LLM", max_attempts=self.max_attempts)
        object_type = _metadata_value(case, "Object Type") or _spec_value(specification, "OBJECT_TYPE")
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
            spec = specification
            objective = {
                "next_step_case": "case-specific next event",
                "next_step_history": "most frequent next activity across matching traces",
                "next_step_probability": "the distribution of immediate next activities, with counts and percentages",
                "next_step_what_if": "the most frequent alternative next activity after excluding the nominal next activity",
            }[case.category]
            case_id = _metadata_value(case, "Case ID / Target", "Case ID", "Target") if case.category == "next_step_case" else None
            match_mode = "CURRENT_ACTIVITY" if "MATCH_MODE=CURRENT_ACTIVITY" in (spec or "") else "FULL_PREFIX"
            match_instruction = (
                "Use only the final/current activity in the prefix as the match point; do not require earlier prefix activities. "
                "Return the event immediately after that current activity."
                if match_mode == "CURRENT_ACTIVITY" else
                "Match the complete prefix consecutively, preserving order and repetitions, then return the immediately following event."
            )
            history_instruction = ""
            if case.category == "next_step_history":
                history_instruction = (
                    "For a historical task, group next activities after matching the current activity and return the most frequent."
                    if match_mode == "CURRENT_ACTIVITY" else
                    "For a historical task, group next activities after exact full-prefix matches and return the most frequent."
                )
            excluded_activity = _spec_value(specification, "EXCLUDE_ACTIVITY")
            output_instruction = (
                "For a probability task, return exactly three columns: activity, count, probability_percentage; calculate percentages over all matched immediate-next transitions."
                if case.category == "next_step_probability" else
                "For a what-if task, exclude the nominal next activity explicitly named as impossible, then rank the remaining immediate next activities by frequency. Return one activity column."
                if case.category == "next_step_what_if" else
                "Return exactly one activity column."
            )
            question = (
                f"Task: return the {objective}.\n"
                f"Object type: {object_type}.\n"
                f"Exact case ID: {case_id or 'not specified'}.\n"
                f"Complete ordered trace prefix, including its final/current activity: {json.dumps(prefix, ensure_ascii=False)}.\n"
                f"{match_instruction} "
                "For a case-specific task, filter to that exact case and do not aggregate. "
                f"{history_instruction} {output_instruction} "
                + (f"For the what-if task, exclude {excluded_activity!r} from candidate next activities. " if excluded_activity else "")
                + f"Use only the provided event-log columns.\nStructured fields:\n{spec}\nOriginal question: {case.question}"
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
                    result_contract = (
                        "Return exactly three columns: activity, count, probability_percentage; compute percentages over all matched immediate-next transitions."
                        if case.category == "next_step_probability" else
                        "Return exactly one activity column, excluding the explicitly specified nominal activity."
                        if case.category == "next_step_what_if" else
                        "Return exactly one activity column."
                    )
                    filter_instruction = (
                        "Structured numeric filters are represented by <attribute>_min and <attribute>_max columns; use _max for > or >= filters and _min for < or <= filters. "
                        "Use related_source_ids or related_target_ids for object-object filters, following the direction in FILTERS_JSON. "
                        if "FILTERS_JSON=" in (spec or "") else ""
                    )
                    prompt += (f"\nActual DataFrame columns: {columns}. Actual dtypes: {dtypes}. "
                               "SQL dialect: DuckDB. Use only those exact columns; do not invent org:resource or any other attribute. "
                               "Quote column names containing colons with double quotes. "
                               "Build an ordered_events CTE with ROW_NUMBER() PARTITION BY case ID and ORDER BY timestamp, event ID. "
                               "Do not put window functions in WHERE or HAVING; filter their aliases in an outer CTE. "
                               "Match every prefix activity at consecutive positions using self-joins on case ID and position. "
                               f"{filter_instruction}"
                               "For an exact case, filter to that case and return its next activity without aggregation. "
                               "For history, group only the next activity after a full prefix match and order by count descending, activity ascending. "
                               f"{result_contract} Do not use LISTAGG/LIST_AGG or SQL comments; return one read-only query.")
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
                    expected_columns = 3 if case.category == "next_step_probability" else 1
                    if len(data.columns) != expected_columns:
                        raise ValueError(f"Next-step query must return exactly {expected_columns} column(s) for {case.category}")
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
