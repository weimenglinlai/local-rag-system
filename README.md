# Local RAG System

A self-hosted retrieval augmented generation system with document ingestion, vector indexing, reranking and a browser UI.

## Structure

```
.gitignore
config/settings.yaml
config/sources.yaml
index.html
rag_core/crawler.py
rag_core/database.py
rag_core/embedder.py
rag_core/init__.py
rag_core/llm.py
rag_core/rag_engine.py
rag_core/vector_store.py
rag_core/wordcloud.py
requirements.txt
static/app.js
static/style.css
tests/test_rag.py
webui.py
```

## Setup

Install dependencies from `requirements.txt` when present:

```bash
pip install -r requirements.txt
```

Data files, model weights and generated outputs are not included; place them under the paths referenced by the configuration before running.
