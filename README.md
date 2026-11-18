# FALCON: Flexible Agentic Lineup for Comprehending Object Network logs

[![Paper](https://img.shields.io/badge/Paper-Under%20Review-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the code and benchmark resources for **FALCON** (*Flexible Agentic Lineup for Comprehending Object Network logs*), presented in the research paper *"FALCON: Flexible Agentic Lineup for Comprehending Object Network logs"*. FALCON is a Multi-Agent System (MAS) for autonomous, conversational Object-Centric Process Mining (OCPM) that answers expressive natural-language queries against Object-Centric Event Log (OCEL 2.0) relational databases. By orchestrating three specialized agents over a directed cyclic graph—an intent contextualization agent (`ContextAgent`), a deterministic SQL coding agent (`CodeAgent`), and a Chain-of-Thought validation and reasoning agent (`ReasoningAgent`)—FALCON delegates exact computations to the database engine and grounds reasoning strictly in extracted facts to prevent hallucinations.

## Key Features

* **Multi-Agent Cyclic Orchestration:** Models the analytical workflow as a directed cyclic graph (`LangGraph`) across `ContextAgent`, `CodeAgent`, and `ReasoningAgent`, enabling iterative self-correction and backward routing whenever syntactic SQL errors or semantic discrepancies arise.
* **Direct Relational OCEL 2.0 Querying:** Operates directly on the relational schema of OCEL 2.0 databases (`event`, `object`, `event_object`, `object_object`, and per-type attribute/timestamp tables) without lossy trace flattening or task-specific fine-tuning, supporting `Scalar`, `Boolean`, `Set`, and multi-object `Next-Step Prediction` queries.
* **Deterministic Computation & Anti-Hallucination:** Pushes mathematical aggregations, global probability estimations, and multi-hop relational traversals to SQLite, while `ReasoningAgent` validates both structural output contracts and logical consistency before synthesizing the final answer.
* **Reproducible Benchmarking Suite:** Includes full evaluation pipelines and curated query workbooks to compare FALCON against `PMAgent` and `pm4py.llm` baselines on standard OCEL 2.0 logs (Container Logistics and Order Management), measuring accuracy, failure rate, response time, and FLOPs.

## Repository Structure

* **`src/main.py`**: Interactive CLI entry point (`run_interactive_pipeline`) and standalone demos for each agent (`demo_agent_1`, `demo_agent_2`, `demo_agent_3`).
* **`src/`**: Core source code directory:
  * `agents/`: Core agent implementations powered by local quantized HuggingFace models:
    * `request_agent.py`: **ContextAgent** (`Qwen2.5-14B-Instruct`) for schema understanding, in-context/out-of-context classification, and intent extraction;
    * `query_agent.py`: **CodeAgent** (`Qwen3-Coder-30B-A3B-Instruct`) for SQLite query synthesis and feedback-driven SQL repair;
    * `response_agent.py`: **ReasoningAgent** (`Qwen2.5-14B-Instruct`) for SQL execution, static/semantic validation, and natural-language answer synthesis;
  * `orchestrator/`: Defines `PipelineState` and builds the cyclic `LangGraph` workflow with conditional routing and retry budgets (`orchestrator.py`);
  * `utils/`: Shared LLM loader (`create_llm` with 4-bit NF4 quantization) and SQLite database connection (`utils.py`);
  * `dataset/`: Workbook loading (`datasets.py`) and read-only OCEL SQLite schema/flattening utilities (`ocel.py`);
  * `benchmark/`: Comparative evaluation framework (`run.py`, `adapters.py`, `evaluation.py`, `generic_sql.py`, `pm4py_data.py`), Next-Step Prediction experiments (`next_step.py`, `next_step_data.py`), and computational cost/latency scripts (`measure_flops.py`, `measure_latency.py`, `cost_common.py`).
* **`benchmark/`**: Evaluation data and databases for benchmarking the system:
  * `databases/`: OCEL 2.0 SQLite databases (`container_logistics.sqlite` and `order_management.sqlite`);
  * `queries/`: Curated benchmark workbooks (`container_logistics_benchmark.xlsx` and `order_management_benchmark.xlsx`).
* **`requirements.txt`**: List of project and benchmark dependencies.

## Requirements

To run the code, ensure you have the libraries specified in `requirements.txt` installed:

```bash
pip install -r requirements.txt
```

Main dependencies include:

* Python (>=3.9)
* PyTorch, Transformers, Accelerate, BitsAndBytes (for 4-bit quantized local LLM inference)
* LangChain & LangGraph (for multi-agent orchestration)
* FLOPpy (`floppy-tracker`, for computational cost monitoring)
* Pandas, OpenPyXL, DuckDB, and PM4Py (for benchmark data handling and baseline execution)

## Usage

Run the interactive multi-agent pipeline from the repository root:

```bash
python main.py
```

Run the comparative benchmark over the OCEL query workbooks:

```bash
python -m src.benchmark.run --systems orchestrator,PMAgent,pm4py
```

## Citation

```bibtex
  Coming soon...
```
