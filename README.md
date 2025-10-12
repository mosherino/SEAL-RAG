# SEAL-RAG

**A multi-hop retrieval-augmented generation system with entity-based reasoning**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.3+-green.svg)](https://github.com/langchain-ai/langgraph)

> 🔬 Research paper submitted to ICLR 2025. See [Citation](#citation) for details.

---

## Features

- 🔗 **Multi-hop question answering** with entity extraction and repair loops
- 📊 **Configurable retrieval** (Pinecone vector store + OpenAI embeddings)
- 🎨 **Interactive UI** via LangGraph Studio or programmatic Python API
- 📈 **Reproducible experiments** with included notebooks and evaluation pipelines
- ⚙️ **Flexible configuration** (models, retrieval settings, temperature)

---

## Quick Start

**Prerequisites:** 
- Python 3.11+
- OpenAI API key (for LLM and embeddings)
- **Vector store with documents** (default: Pinecone; extensible to any LangChain-compatible store)

### 1. Install dependencies

```bash
uv python install 3.11
uv sync
```

### 2. Set up environment

```bash
cp env.example .env
# Edit .env and add your API keys:
# OPENAI_API_KEY=your_key_here
# PINECONE_API_KEY=your_key_here
```

> ⚠️ **Important:** SEAL-RAG requires a vector store with indexed documents for retrieval.
> - **Default:** Pinecone (set `PINECONE_API_KEY` and configure `pinecone_index_name`)
> - **Custom:** Modify `src/modules/configuration.py` to use another LangChain vector store (Chroma, Weaviate, etc.)
> 
> See [Indexing](#notebooks) to build a Pinecone index, or configure an existing one.

### 3. Run the graph

**Option A: Interactive UI (LangGraph Studio)**
```bash
uv run langgraph dev
# Open http://localhost:2024 in your browser
```

**Option B: Python API**
```python
from src.workflow_manager import build_seal_rag_graph

graph = build_seal_rag_graph()
result = graph.invoke({"user_query": "Your question here"})
print(result["final_answer"])
```

---

## Usage

### LangGraph Studio (Interactive UI)

Start the development server:

```bash
uv run langgraph dev
```

In the UI:
1. Select the `seal-rag` graph
2. Provide an input with `user_query`
3. Configure settings (optional): `retriever_k`, `pinecone_index_name`, `temperature`

### Python API

```python
from src.workflow_manager import build_seal_rag_graph

# Build the graph
graph = build_seal_rag_graph()

# Run with defaults
result_state = graph.invoke({
    "user_query": "What is the key contribution of the referenced method?"
})
print(result_state.get("final_answer"))

# Run with custom configuration
custom_config = {
    "configurable": {
        "retriever_k": 1,
        "pinecone_index_name": "<your_pinecone_index_name>",
        "temperature": 0.0,
        "model": "openai:gpt-4o"
    }
}
result_state = graph.invoke({"user_query": "..."}, config=custom_config)
```

### Notebooks

All notebooks auto-read the `.env` file. Run with:

```bash
uv sync --extra notebooks
uv run jupyter notebook
```

**Available notebooks:**

| Notebook | Path | Purpose |
|----------|------|---------|
| **Indexing** | `indexing.ipynb` | **Required first step:** Build/update the Pinecone index with your document corpus |
| **Experiments** | `experiment.ipynb` | Run evaluation pipeline over questions and generate CSVs |
| **Statistics** | `src/files/experiment_comarison/statistical_confidence.ipynb` | Aggregate results and compute confidence intervals |

---

## Configuration

The graph exposes configurable parameters via LangGraph's `configurable` interface. Override them at runtime:

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| `model` | `openai:gpt-4o` | `openai:gpt-4o`, `openai:gpt-4o-mini`, `openai:gpt-4.1`, `openai:gpt-4.1-mini`, `openai:gpt-3.5-turbo` | Chat model provider/version |
| `temperature` | `0.0` | `0.0–1.0` | Sampling temperature (lower = more deterministic) |
| `extraction_temperature` | `0.2` | `0.0–1.0` | Temperature for entity extraction |
| `retriever_k` | `1` | `1–7` | Number of documents to retrieve per call |
| `embed_model` | `text-embedding-3-small` | `text-embedding-3-small`, `text-embedding-3-large`, `text-embedding-ada-002` | OpenAI embedding model |
| `pinecone_index_name` | `seal-v3-hard` | Any valid index | Target Pinecone index for retrieval |
| `pinecone_namespace` | `None` | Any string or `None` | Optional namespace scoping |
| `repair_loop_limit` | `5` | `0–50` | Max micro-query iterations before forcing answer |

**Full configuration reference:** [`src/modules/configuration.py`](src/modules/configuration.py)

---

## Results

Experimental results (CSVs) are included in this repository:

```
src/files/
├── experiment_comarison/
│   ├── k_1/                    # Experiments with retriever_k=1
│   │   └── seal_k_1_model_*.csv
│   └── k_3/                    # Experiments with retriever_k=3
│       └── seal_k_3_model_*.csv
└── ablations/                  # Ablation study results
    └── Seal-RAG-Ablations_*.csv
```

---

## Project Structure

```
.
├── src/
│   ├── modules/
│   │   ├── agents/            # Agent nodes (retrieve, repair, rank, generate)
│   │   └── configuration.py   # Configurable parameters
│   ├── other_rags/            # Baseline implementations (e.g., self-rag)
│   ├── states.py              # State schema
│   └── workflow_manager.py    # LangGraph graph builder
├── app.py                     # Graph entrypoint
├── indexing.ipynb             # Index building notebook
├── experiment.ipynb           # Evaluation pipeline
├── statistical_confidence.ipynb
├── langgraph.json             # LangGraph Studio config
├── pyproject.toml             # Dependencies (uv)
└── README.md
```

---

## FAQ

### Which API keys are required?

- **Required:** `OPENAI_API_KEY`, `PINECONE_API_KEY`
- **Optional:** `LANGCHAIN_API_KEY` (for LangSmith tracing/logging)

### Can I run this without a vector store?

No. SEAL-RAG requires a populated vector store for document retrieval. You must either:
1. Use Pinecone: run `indexing.ipynb` to build an index, or point to an existing one via `pinecone_index_name`
2. Use another vector store: modify `src/modules/configuration.py` to integrate any LangChain-compatible store (Chroma, Weaviate, FAISS, etc.)

### Can I use a vector store other than Pinecone?

Yes. Pinecone is the default, but you can extend `src/modules/configuration.py` to support any LangChain vector store:
1. Update the `build_retriever()` method to initialize your chosen store
2. Add any necessary configuration parameters (e.g., `chroma_collection_name`)
3. Install the required package (e.g., `uv add langchain-chroma`)

See [LangChain's VectorStore docs](https://python.langchain.com/docs/integrations/vectorstores/) for supported options.

### How do I install without activating a venv?

Use `uv` for everything:
```bash
uv python install 3.11
uv sync
uv run langgraph dev
```

### How do I export a requirements.txt for pip users?

```bash
uv export --format requirements-txt > requirements.txt
```

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Citation

If you use SEAL-RAG in your research, please cite:

```bibtex
@misc{seal-rag-2025,
  title={SEAL-RAG: Entity-Driven Multi-Hop Retrieval-Augmented Generation},
  author={[Authors]},
  year={2025},
  note={Submitted to ICLR 2025}
}
```

*(BibTeX will be updated upon acceptance)*
