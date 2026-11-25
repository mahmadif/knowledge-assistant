### Knowledge Assistant

A modular Retrieval-Augmented Generation (RAG) assistant built with LangChain, ChromaDB and Gradio:
- Two ingestion sources: Confluence (via API) or local `.md` / `.txt` directories.
- Local embeddings: HuggingFace `sentence-transformers/all-MiniLM-L6-v2` on CPU, normalized embeddings, cosine distance in ChromaDB.
- Switchable LLM backend: local models through Ollama or Azure OpenAI.
- Grounded answers: numbered source labels and a fixed abstention behaviour when the context is insufficient.
- Gradio chat UI showing the answer and its sources.
- Evaluation: Hit@k, MRR, faithfulness, correctness and abstention rate.

#### Getting started

Requires Python 3.10+ (developed on 3.11).

For local generation, e.g. `ollama pull qwen2.5:3b`.
`
git clone https://github.com/mahmadif/knowledge-assistant.git
`
`make install` or
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

#### Usage

```bash
python ingest.py --source local --path data/handbook --reset
# or: make index

python app.py
# or make run
```

#### Evaluation

`make eval` on `eval/golden_dataset.json`: 
- `answerable` (direct lookup/multi-hop questions)
- `unanswerable`

Latest run (`qwen2.5:3b`, chunk size 1000 / overlap 150, k = 3):
- Retrieval
  - Multi-source `Hit@3`: 100.0%
  - `MRR`: 0.97
- Generation
  - Faithfulness: 94.4%
  - Answer correctness: 94.4%
  - Abstention rate: 66.7%

#### Future Directions
- Replacing full re-indexing (`--reset`): e.g., incremental indexing with page-level upserts, deduplication, etc.
- Systematic benchmarking of parameters (chunk size, overlap, `k`, distance metric).
- A separate, stronger LLM judge instead of using the same small local model.
- Hybrid retrieval with reranking.

#### Agentic RAG – Ongoing Development

Beyond improving the existing RAG pipeline, the project is being extended with a self-reflective agentic workflow to improve retrieval quality and answer reliability.