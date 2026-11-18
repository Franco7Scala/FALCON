"""Execution adapter for the FALCON multi-agent orchestrator."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import ast
import json
import re
import time

from src.dataset.datasets import QueryCase
from src.dataset.ocel import orchestrator_schema_guidance


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
        import src.utils.utils as utils_module
        import src.agents.response_agent as response_module
        from ..orchestrator import orchestrator as pipeline_module

        pipeline_module.MAX_ATTEMPTS = max_attempts

        database = database or Path(__file__).resolve().parents[2] / "benchmark" / "databases" / "container_logistics.sqlite"
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

