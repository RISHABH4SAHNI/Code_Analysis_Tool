# Setup Guide

This guide takes you from a fresh clone to a first analysis, and covers each model provider and common problems. For what the tool does and how it works, see the [README](../README.md).

## Prerequisites

| Requirement | Why |
| --- | --- |
| Python 3.10+ | Runs the tool |
| OpenSearch 2.x with the k-NN plugin | Stores function embeddings. Both approaches need it |
| Docker (optional) | Easiest way to run OpenSearch locally |
| An OpenAI API key, or Ollama | Embeddings and analysis |

## 1. Start OpenSearch

### With Docker (recommended for local use)

```bash
docker run -d --name opensearch -p 9200:9200 \
  -e discovery.type=single-node \
  -e DISABLE_SECURITY_PLUGIN=true \
  -e DISABLE_INSTALL_DEMO_CONFIG=true \
  -e "OPENSEARCH_JAVA_OPTS=-Xms512m -Xmx512m" \
  opensearchproject/opensearch:2.13.0
```

This disables authentication and TLS. Use it only on your own machine.

The node takes 20 to 60 seconds to start. Check it with:

```bash
curl http://localhost:9200
```

The reply should include `"number" : "2.13.0"`.

### Existing or secured cluster

Point the tool at it in `.env`:

```bash
OPENSEARCH_HOST=search.example.com
OPENSEARCH_PORT=9200
OPENSEARCH_USER=admin
OPENSEARCH_PASSWORD=your-password
OPENSEARCH_USE_SSL=true
OPENSEARCH_VERIFY_CERTS=true
```

The cluster must have the k-NN plugin, which the official OpenSearch distribution includes. Amazon OpenSearch Service domains also support k-NN.

## 2. Install the tool

```bash
git clone https://github.com/RISHABH4SAHNI/Code_Analysis_Tool.git
cd Code_Analysis_Tool
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

## 3. Choose a model provider

You pick an **embedding service** and an **LLM service** separately when you start the tool. They can be different providers.

### OpenAI

Add your key to `.env`:

```bash
OPENAI_API_KEY=sk-...
```

The defaults are `text-embedding-3-large` (3072 dimensions) for embeddings and `gpt-4o` for analysis. Override them with `OPENAI_EMBEDDING_MODEL`, `OPENAI_EMBEDDING_DIMENSIONS` and `OPENAI_CHAT_MODEL`.

### Ollama (local, no API key)

Install [Ollama](https://ollama.com), then pull the default models:

```bash
ollama pull nomic-embed-text   # embeddings, 768 dimensions
ollama pull codellama:7b       # analysis
```

If Ollama is not on `http://localhost:11434`, set `OLLAMA_BASE_URL`. To use other models, set `OLLAMA_CHAT_MODEL`, `OLLAMA_EMBEDDING_MODEL` and `OLLAMA_EMBEDDING_DIMENSIONS`. The dimensions must match the embedding model.

Small local models follow the structured answer format less reliably than hosted ones, so confidence scores and follow-up context requests work best with larger models.

### Perplexity

Perplexity can be used as the LLM service:

```bash
PERPLEXITY_API_KEY=pplx-...
PERPLEXITY_CHAT_MODEL=sonar      # default
```

To use Perplexity for embeddings as well, also set `PERPLEXITY_EMBEDDING_MODEL` and `PERPLEXITY_EMBEDDING_DIMENSIONS` to a model your account can use. Without them, choosing Perplexity embeddings fails with an explanatory error.

## 4. Verify the setup

```bash
python main_structured.py
```

Choose **8. System Diagnostics**. It shows installed packages, which keys are set and whether OpenSearch is reachable.

To check the code itself:

```bash
pip install -r requirements-dev.txt
pytest
```

The tests mock all external services, so they need no keys and no OpenSearch.

## 5. First analysis

### CLI

1. Run `python main_structured.py`.
2. Choose **1. Process Repository**. Pick an approach (`function_lookup_table` or `call_graph`), the embedding and LLM services, and the repository path.
3. Choose **2. Analyze Issue** and describe the problem, for example: `Login fails with a 500 error when the password is empty`.

### Web UI

1. Run `streamlit run ui/streamlit_app.py` and open http://localhost:8501.
2. In the sidebar, pick the approach and services, then click **Initialize Services**.
3. On **Repository Processing**, enter the path and click **Process Repository**.
4. On **Issue Analysis**, describe the issue and click **Analyze Issue**.

Processing time is dominated by embedding calls: one per function. A repository with a few hundred functions takes a few minutes with OpenAI.

## Tuning

These constants are in [config/settings.py](../config/settings.py):

| Setting | Default | Effect |
| --- | --- | --- |
| `CONFIDENCE_THRESHOLD` | `0.8` | Below this, the tool fetches functions the LLM asks for and re-analyses |
| `MAX_ANALYSIS_ITERATIONS` | `3` | Maximum re-analysis rounds |
| `DEFAULT_FUNCTION_LIMIT` | `5` | Functions retrieved per question |
| `MAX_CALL_GRAPH_DEPTH` | `3` | Default call depth for the call graph approach (UI and CLI allow 1-5) |
| `EXCLUDED_DIRS` / `TEST_DIRS` | see file | Directories skipped when scanning a repository |

Test files are skipped by default. Enable **Include test files** in the UI or answer `y` in the CLI to index them.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `ConnectionError` or "OpenSearch connection failed" | OpenSearch is not running or not reachable. Check `curl http://localhost:9200` and the `OPENSEARCH_*` settings |
| "OpenAI API key is required" | `OPENAI_API_KEY` is missing from `.env` or the environment, and OpenAI is selected for embeddings or the LLM |
| Index is recreated with a warning about dimensions | You changed the embedding model. This is expected; the index is rebuilt for the new size |
| "No relevant functions found" | The repository has not been processed with the current approach in this session. Each approach has its own index |
| "No Python functions found" | The path has no `.py` files outside skipped directories such as `tests/`, `venv/` and `build/` |
| Files skipped during processing | Those files have syntax errors or are not UTF-8. The log lists each file and the error |
| Ollama requests time out | The model is still loading or is too large for the machine. Pull a smaller model or raise `timeout` in `OLLAMA_CONFIG` |
| Low confidence on every answer | Rephrase with concrete details: error messages, function or file names. Try the call graph approach for bugs that span several functions |
