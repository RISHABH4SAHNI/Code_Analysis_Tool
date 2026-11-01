# API Reference

How to drive Code Analysis Bot from Python. Most callers need only `WorkflowEngine`; the lower-level classes are listed for extending or testing the tool.

Run code from the project root, or put it on `PYTHONPATH`, since modules are imported as `core.…`, `services.…` and so on. Call `dotenv.load_dotenv()` first if your keys are in `.env`.

## WorkflowEngine

`core/workflow_engine.py`. Builds the services for one approach and provider combination, processes a repository and analyses issues.

```python
from core.workflow_engine import WorkflowEngine

engine = WorkflowEngine(
    embedding_service_type="openai",           # "openai" | "ollama" | "perplexity"
    llm_service_type="openai",                 # "openai" | "ollama" | "perplexity"
    analysis_approach="function_lookup_table", # or "call_graph"
    history_manager=None,                      # optional HistoryManager; default writes analysis_history.json
)
```

Raises `ValueError` for an unknown approach or provider, or when a required API key is missing. Creating the engine does not contact OpenSearch.

### `process_repository(repo_path, include_tests=False, max_functions=None) -> dict`

Parses the repository, embeds every function and stores it in the approach's OpenSearch index. Earlier documents for the same repository are deleted first, so re-processing replaces them. Later searches are limited to this repository.

Raises `ValueError` if the path does not exist, is not a directory or contains no Python functions. OpenSearch errors propagate.

Result for `function_lookup_table`:

```python
{
    "total_functions": 233,
    "embedded_functions": 233,
    "approach": "function_lookup_table",
    "repository_name": "my-app",
    "skipped_files": 0,          # files that could not be parsed
    "summary": {
        "total_functions": 233,
        "async_functions": 4,
        "functions_with_error_handling": 61,
        "unique_modules": 27,
        "unique_repositories": 1
    }
}
```

`call_graph` returns the same top-level keys, plus `graph_statistics`:

```python
"graph_statistics": {
    "total_nodes": 233, "total_edges": 451,
    "root_nodes": 27,             # functions nothing calls
    "leaf_nodes": 109,            # functions that call nothing in the repo
    "max_depth": 8,               # longest shortest-path from a root
    "avg_calls_per_function": 1.94,
    "stored_functions": 233, "total_functions": 233, "failed_functions": 0
}
```

Its `summary` holds the function counts above together with the graph counts.

### `analyze_user_issue(user_query, context=None) -> dict`

Rewrites the query, retrieves functions, runs the LLM analysis and saves the result to history.

`context` keys, all optional:

| Key | Type | Effect |
| --- | --- | --- |
| `repo_name` | `str` | Added to the query-rewriting prompt |
| `tech_stack` | `list[str]` | Added to the query-rewriting prompt |
| `priority` | `str` | Added to the query-rewriting prompt |
| `call_graph_depth` | `int` 1-5 | Call graph approach: how many call levels to include |

Result:

```python
{
    "query": "Login fails with a 500 error when the password is empty",
    "enhanced_query": "...",
    "functions_found": 5,          # retrieved by search
    "functions_analyzed": 6,       # including any the LLM asked for
    "analysis": "CONFIDENCE_SCORE: 0.85\nANALYSIS: ...",  # full LLM reply
    "confidence_score": 0.85,
    "suggested_fixes": [{"description": "...", "type": "code_change"}],
    "search_results": [SearchResult, ...],
    "approach": "function_lookup_table",
    "graph_context": {"insights": "..."}   # call graph approach only, else None
}
```

When nothing is found, or the query is empty, the result is `{"error": "...", ...}` instead.

### Other methods

| Method | Description |
| --- | --- |
| `switch_analysis_approach(name) -> bool` | Rebuilds the services for another approach. Returns `True` if a repository had been processed, meaning it must be processed again |
| `set_llm_service(name)` | Switches the query-rewriting and analysis LLM. The index is unaffected |
| `export_function_lookup_table(path, format="json") -> str` | Writes the function table as `json` or `csv` and returns the absolute path. Lookup table approach only, after processing |
| `validate_services() -> dict[str, bool]` | Live check of `embedding_service`, `search_service`, `query_enhancer` and `analysis_service`. Makes one embedding call and one LLM call |
| `get_recent_analyses(limit=10) -> list[dict]` | History entries, newest first |
| `get_analysis_history_stats() -> dict` | Totals, average confidence and usage per provider pair |
| `get_supported_approaches() -> list[dict]` | `key`, `name` and `description` of each approach |

## Models

`models/function_model.py` and `models/call_graph_model.py`.

### FunctionMetadata

One extracted function. `name`, `lookup_id`, `file_path`, `repository_name` and `module_name` are required; everything else has a default.

| Field | Meaning |
| --- | --- |
| `lookup_id` | Stable UUID derived from repository, file, qualified name and line, so it is the same on every run |
| `start_line`, `end_line`, `code`, `code_with_line_numbers` | Source, including decorators |
| `class_context` | Enclosing class name, or `None` |
| `calls` | Names called inside the function |
| `qualified_calls` | Calls with their receiver kept, e.g. `self.save`, `helper`, `client.get` |
| `nested_call_ids` | `lookup_id`s of the repository functions those calls resolve to |
| `imports`, `decorators`, `is_async` | As extracted from the source |
| `error_handling` | `has_try_catch`, `error_types` and `try_catch_blocks` |

### SearchResult

`function_metadata`, `relevance_score`, `search_method`, `match_type`, `approach_used` and `nested_functions`.

`nested_functions` depends on the approach:

```python
# function_lookup_table: callees keyed by lookup_id
{"<lookup_id>": {"function_name": "...", "file_path": "...", "code": "...", "start_line": 1, "end_line": 9}}

# call_graph
{
    "call_graph_context": {
        "primary_function": "login", "dependency_count": 3, "dependency_names": [...],
        "graph_depth": 2, "total_functions": 4, "call_paths_count": 1,
        "files_involved": [...], "modules_involved": [...],
        "has_error_handling": True, "async_functions": 0, "relevance_score": 1.0
    },
    "dependency_functions": ["<lookup_id>", ...],
    "call_paths": [["<root id>", "...", "<this id>"]]
}
```

Results the LLM requested during re-analysis have `match_type == "additional_context"`.

### AnalysisRequest and AnalysisResult

`AnalysisRequest` holds `query`, `enhanced_query`, `functions` (a list of `SearchResult`), `context`, `approach` and `call_graph_context`.

`AnalysisResult` holds `analysis`, `confidence_score`, `functions_analyzed`, `additional_context_needed` (lookup IDs), `suggested_fixes`, `approach_used` and `graph_context`.

### CallGraph, CallGraphNode, CallGraphSearchResult

- `CallGraph` has `nodes` (`lookup_id` → `CallGraphNode`), `root_nodes`, `leaf_nodes`, `max_depth` and an `edges` property listing `(caller_id, callee_id)` pairs.
  - `get_dependencies(node_id, max_depth)` returns `{lookup_id: distance}` for every callee within `max_depth`.
- `CallGraphNode` has `function_metadata`, `dependencies` (callee IDs), `dependents` (caller IDs) and `depth_level`.
- `CallGraphSearchResult` has `primary_node`, `dependency_context` (callee nodes, excluding the primary node), `graph_depth`, `total_context_functions` and `call_paths`.
  - Its methods are `get_all_functions()` and `get_context_summary()`.

## Lower-level components

### RepositoryProcessor

`core/repository_processor.py`.

```python
from core.repository_processor import RepositoryProcessor

processor = RepositoryProcessor()
functions = processor.process_repository("/path/to/repo", include_tests=False, max_functions=None)
processor.skipped_files   # [{"file": "...", "error": "..."}] for files that failed to parse
```

### CallResolver

`core/call_resolver.py`. Maps calls to likely targets; used for both `nested_call_ids` and call graph edges.

- `self.x()` and `cls.x()` resolve to `x` in the caller's class.
- A bare `x()` resolves to `x` in the same module, then to `x` in a module the caller imports, then to any module-level `x`.
- `obj.x()` on any other receiver links every function named `x`.
- Built-ins such as `len` and `print` are ignored.

```python
from core.call_resolver import CallResolver

target_ids = CallResolver(functions).resolve(functions[0])
```

### CallGraphProcessor

`core/call_graph_processor.py`.

| Method | Returns |
| --- | --- |
| `build_call_graph(functions)` | `CallGraph` |
| `get_function_context_with_dependencies(lookup_id, max_depth=3)` | `CallGraphSearchResult`, or `None` for an unknown ID |
| `get_call_paths_to_function(lookup_id, max_depth=3, max_paths=20)` | Lists of IDs from a root function to the target |
| `find_functions_by_name(name)` | `CallGraphNode`s with that exact name |
| `get_graph_statistics()` | The `graph_statistics` dict shown above, without the storage counts |

### FunctionLookupTable

`core/function_lookup.py`. A pandas table of function metadata.

- `build_lookup_table(functions)`
- `get_function_by_id(lookup_id)`
- `get_functions_by_name(name)`
- `get_nested_functions(lookup_id)`
- `get_most_relevant_functions(query, limit=5)`: keyword scoring, without embeddings
- `export_to_json(path)` and `export_to_csv(path)`
- `get_table_summary()`

### Search services

`services/vector_search_service.py` and `services/call_graph_search_service.py`. Both extend `OpenSearchIndexService` (`services/opensearch_base.py`), which provides:

- `setup_index()`: creates the index for the embedding service's dimensions, and recreates it if they changed
- `delete_repository(name)`
- `refresh()`
- `health_check()` and `is_healthy()`
- `get_index_stats()`
- `repository_name`: when set, searches are limited to that repository

| Service | Main methods |
| --- | --- |
| `VectorSearchService(embedding_service)` | `store_function`, `store_functions_bulk`, `search_functions(query, limit)`, `get_function_by_id`, `search_by_filters(filters, limit)` |
| `CallGraphSearchService(embedding_service, call_graph_processor=None)` | `build_and_store_call_graph(functions)`, `search_functions_with_context(query, limit, max_depth)`, `get_function_by_id`, `get_function_with_graph_context(lookup_id, max_depth)` |

### Provider services

| Base class | Implementations | Contract |
| --- | --- | --- |
| `EmbeddingService` | `OpenAIEmbeddingService`, `OllamaEmbeddingService`, `PerplexityEmbeddingService` | `create_embedding`, `create_query_embedding`, `create_call_graph_embedding`, `get_embedding_dimensions` |
| `QueryEnhancementService` | OpenAI, Perplexity, Ollama | `enhance_query(query, context) -> str`; returns the original query on failure |
| `AnalysisService` | OpenAI, Perplexity, Ollama | Implement `_complete(system_prompt, user_prompt) -> str`; prompting and parsing are shared |

To add a provider, subclass these and register it in `factories/embedding_factory.py` and `factories/llm_factory.py`.

The analysis reply is parsed into `CONFIDENCE_SCORE`, `ANALYSIS`, `GRAPH_INSIGHTS`, `ISSUES_FOUND`, `SUGGESTED_FIXES` and `ADDITIONAL_CONTEXT_NEEDED` sections. Multi-line sections and markdown headers such as `**ANALYSIS:**` are accepted. A confidence of `85` or `85%` is read as `0.85`.

### ServiceFactory

`factories/service_factory.py`.

- `get_services_for_approach(approach, embedding_type, llm_type)` returns a dict with these keys:
  - `embedding_service`, `query_enhancer`, `analysis_service` and `repository_processor`
  - `search_service`
  - `lookup_table` or `call_graph_processor`, depending on the approach
- One embedding service is shared by all of them.
- An unknown approach raises `ValueError`.

### HistoryManager

`utils/history_manager.py`. A JSON file of past analyses, thread-safe and size-capped, with backups.

- `HistoryManager(path="analysis_history.json")`
- `add_analysis(...)`
- `get_recent_analyses(limit)`: newest first
- `get_all_analyses()`
- `get_statistics()`
- `export_history(path) -> str`
- `clear_history()`

## Configuration

Environment variables are listed in the [README](../README.md#configuration) and tuning constants in [SETUP.md](SETUP.md#tuning). Index mappings and provider defaults are in [config/settings.py](../config/settings.py).
