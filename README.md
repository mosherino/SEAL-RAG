Seal‑RAG (ICLR Supplementary Material)

This repository contains the supplementary materials for the Seal‑RAG system evaluated in our ICLR submission. It includes the full retrieval‑augmented generation (RAG) pipeline (LangGraph), indexing notebook, experiment notebook, confidence/statistics notebook, and the CSV outputs for full experiments and ablations.

Contents
- Quick start (environment, installation, and running)
- .env template (API keys and optional telemetry)
- Notebooks: indexing, experiments, and confidence/statistics
- Programmatic usage (run the graph from Python)
- Where results live (CSV locations for full experiments and ablations)
- FAQ for reviewers


Quick start (uv)
1) Prerequisites
- Python 3.11 (managed by uv; macOS/Linux/Windows supported)
- Pinecone account (for the vector index) and OpenAI API access

2) Install and sync dependencies
```bash
# From the repo root
uv python install 3.11
uv sync
uv run python -V   # should print 3.11.x
```

4) Configure environment variables
Create a file named .env in the repository root with the following keys. Only the OpenAI and Pinecone keys are required to run; LangSmith is optional for logging/telemetry.
```ini
# Required
OPENAI_API_KEY=<your_openai_key>
PINECONE_API_KEY=<your_pinecone_key>

# Optional: LangSmith tracing/experiment logging
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=<your_langsmith_key>
LANGCHAIN_PROJECT=seal-rag
```


Notebooks
All notebooks auto‑read the .env file. Open them in Jupyter, VS Code, or your preferred environment.

- Indexing notebook (build/update the Pinecone index)
  - Path: indexing.ipynb
  - Purpose: Ingest documents and build the Pinecone index used by Seal‑RAG. When running the graph, select the target index via configuration (`pinecone_index_name`).

- Experiment notebook (run the main evaluation)
  - Path: experiment.ipynb
  - Purpose: Runs the main experimental pipeline over a set of questions and writes CSVs.

- Confidence/statistics notebook
  - Path: src/files/experiment_comarison/statistical_confidence.ipynb
  - Purpose: Aggregates and visualizes results (confidence, accuracy, comparison across settings and k values).


Programmatic usage (run the Seal‑RAG graph)
You can run the graph directly from Python using LangGraph. The graph expects a State with a user_query and returns a final_answer (plus intermediate fields).
```python
from src.workflow_manager import build_seal_rag_graph

# Build compiled graph (recursion limit is set inside)
graph = build_seal_rag_graph()

# Minimal run with defaults (ensure your .env is set)
result_state = graph.invoke({
    "user_query": "What is the key contribution of the referenced method?"
})

print(result_state.get("final_answer"))

# Optional: override configuration at runtime via LangGraph configurable
# (only if you need to change defaults like retriever_k or index name)
custom = {
    "configurable": {
        "retriever_k": 1,
        "pinecone_index_name": "<your_pinecone_index_name>",  # set to the index you created
        "temperature": 0.0
    }
}
result_state_custom = graph.invoke({"user_query": "…"}, config=custom)
```


Configuration reference (src/modules/configuration.py)
The graph exposes a small set of configurable parameters via LangGraph’s configurable interface. You can override them per‑run using the config argument shown above.

- Model (model)
  - Default: openai:gpt-4o
  - Options: openai:gpt-4o, openai:gpt-4o-mini, openai:gpt-4.1, openai:gpt-4.1-mini, openai:gpt-3.5-turbo
  - Effect: Selects the chat model provider/version used across agents.

- Temperature (temperature)
  - Default: 0.0
  - Range: 0.0–1.0
  - Effect: Sampling temperature for generation steps (lower = more deterministic).

- Extraction temperature (extraction_temperature)
  - Default: 0.2
  - Range: 0.0–1.0
  - Effect: Dedicated temperature for entity/relationship extraction steps.

- Max retries (max_retries)
  - Default: 3
  - Effect: LLM call retry policy for transient failures.

- Retriever k (retriever_k)
  - Default: 1
  - Range: 1–7
  - Effect: Number of documents retrieved per call. For paper reproduction, keep k=1.

- Embedding model (embed_model)
  - Default: text-embedding-3-small
  - Options: text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002
  - Effect: OpenAI embedding model used for Pinecone vectorization.

- Retriever backend (retriever)
  - Default: pinecone (only supported option)
  - Effect: Retrieval backend selection.

- Pinecone index (pinecone_index_name)
  - Example: seal-v2-hard
  - Effect: Target Pinecone index for retrieval. Must exist and contain your corpus.

- Pinecone namespace (pinecone_namespace)
  - Default: empty/None
  - Effect: Optional namespace scoping within the chosen index.

Override pattern
```python
graph.invoke(
  {"user_query": "…"},
  config={
    "configurable": {
      "model": "openai:gpt-4o",           # swap model
      "temperature": 0.0,                   # deterministic
      "retriever_k": 1,                     # keep 1 to reproduce paper results
      "pinecone_index_name": "seal-v2-hard",
      "pinecone_namespace": None
    }
  }
)
```


Where to find results (CSV files)
- Full experiments (k variants) are under:
  - src/files/experiment_comarison/k_1/
  - src/files/experiment_comarison/k_3/
  - These folders include multiple CSVs used in the paper, e.g.:
    - seal_k_1_model_4_1.csv, seal_k_1_model_4_o.csv (and their _mini variants)
    - seal_k_3_model_4_1.csv, seal_k_3_model_4_o.csv (and their _mini variants)

- Ablation experiments are under:
  - src/files/ablations/
  - Example files:
    - Seal-RAG-Ablations - k_1_model_4_1.csv
    - Seal-RAG-Ablations - k_1_model_4_o.csv
    - (and matching _mini variants)


FAQ for reviewers
- Which API keys are required?
  - OPENAI_API_KEY and PINECONE_API_KEY are required. LANGCHAIN_API_KEY is optional (only for LangSmith logging/tracing).

- Do I need to build a new index?
  - If you want to run against your own corpus, use indexing.ipynb to build it. If you already have an index, you can skip indexing and point the configuration to your existing index by setting `pinecone_index_name`.

- Where is the main pipeline?
  - The LangGraph wiring lives in src/workflow_manager.py (function build_seal_rag_graph). Agent nodes are in src/modules/agents/. The state schema is in src/states.py.

- How do I install and run without activating a venv?
  - Use uv for everything:
    ```bash
    uv python install 3.11
    uv sync
    uv run python -V
    uv run langgraph dev
    ```

- How do I export a requirements.txt for pip users?
  ```bash
  uv export --format requirements-txt > requirements.txt
  ```


Run in LangGraph Studio
The repository already includes a langgraph.json that points to all available graphs, including the Seal‑RAG flow.

1) Ensure your .env is present (see above) and your Pinecone index exists.

2) Start the local Studio dev server:
```bash
uv run langgraph dev
```

3) In the Studio UI:
- Select a graph (e.g., seal-rag → src/workflow_manager.py:build_seal_rag_graph).
- Provide an input with the user_query field.
- Optionally set overrides under Config (e.g., retriever_k, pinecone_index_name).

Notes
- The mapping of graph names to modules/functions is defined in langgraph.json. You can add/remove entries as needed.
- The LangGraph CLI is installed via pyproject with uv, so `uv run langgraph dev` works cross‑platform without activating a venv.
