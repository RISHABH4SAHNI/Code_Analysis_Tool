import os
from typing import Dict, Any, List

def get_opensearch_config():
    return {
        "hosts": [{
            "host": os.getenv('OPENSEARCH_HOST', 'localhost'),
            "port": int(os.getenv('OPENSEARCH_PORT', 9200))
        }],
        "http_auth": (
            os.getenv('OPENSEARCH_USER', 'admin'),
            os.getenv('OPENSEARCH_PASSWORD', 'admin')
        ),
        "use_ssl": os.getenv('OPENSEARCH_USE_SSL', 'false').lower() == 'true',
        "verify_certs": os.getenv('OPENSEARCH_VERIFY_CERTS', 'false').lower() == 'true',
        "timeout": int(os.getenv('OPENSEARCH_TIMEOUT', '30')),
        "max_retries": int(os.getenv('OPENSEARCH_MAX_RETRIES', '3')),
        "retry_on_timeout": True
    }

OPENAI_CONFIG = {
    "model": os.getenv('OPENAI_EMBEDDING_MODEL', "text-embedding-3-large"),
    "embedding_dimensions": int(os.getenv('OPENAI_EMBEDDING_DIMENSIONS', '3072')),
    # text-embedding-3-large accepts up to 8191 input tokens
    "embedding_max_tokens": 8000,
    "chat_model": os.getenv('OPENAI_CHAT_MODEL', "gpt-4o"),
    "max_tokens": 1500,
    "enhancement_max_tokens": 200,
    "temperature": 0.1,
    "timeout": 60,
    "max_retries": 3
}

PERPLEXITY_CONFIG = {
    # Perplexity's chat API does not serve the OpenAI embedding models, so the
    # embedding model must be set explicitly to use Perplexity for embeddings
    "embedding_model": os.getenv('PERPLEXITY_EMBEDDING_MODEL', ''),
    "embedding_dimensions": int(os.getenv('PERPLEXITY_EMBEDDING_DIMENSIONS', '1024')),
    "chat_model": os.getenv('PERPLEXITY_CHAT_MODEL', "sonar"),
    "base_url": "https://api.perplexity.ai",
    "max_tokens": 1500,
    "enhancement_max_tokens": 200,
    "temperature": 0.1,
    "timeout": 60,
    "max_retries": 2
}

OLLAMA_CONFIG = {
    "base_url": os.getenv('OLLAMA_BASE_URL', "http://localhost:11434"),
    "embedding_model": os.getenv('OLLAMA_EMBEDDING_MODEL', "nomic-embed-text"),
    "chat_model": os.getenv('OLLAMA_CHAT_MODEL', "codellama:7b"),
    "embedding_dimensions": int(os.getenv('OLLAMA_EMBEDDING_DIMENSIONS', '768')),
    "max_tokens": 1500,
    "enhancement_max_tokens": 200,
    "temperature": 0.1,
    "timeout": 120,
    "max_retries": 2
}

# The knn_vector dimension is a placeholder: the search services set it from the
# active embedding service when they create the index
INDEX_CONFIG = {
    "name": "code_embeddings",
    "settings": {
        "index": {"knn": True},
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "embedding_knn": {"type": "knn_vector", "dimension": 3072},
            "func_name": {"type": "keyword"},
            "lookup_id": {"type": "keyword"},
            "file_path": {"type": "keyword"},
            "file_name": {"type": "keyword"},
            "repository_name": {"type": "keyword"},
            "module_name": {"type": "keyword"},
            "code": {"type": "text"},
            "code_with_line_numbers": {"type": "text"},
            "calls": {"type": "keyword"},
            "is_async": {"type": "boolean"},
            "start_line": {"type": "integer"},
            "end_line": {"type": "integer"},
            "class_context": {"type": "keyword"},
            "nested_call_ids": {"type": "keyword"},
            "decorators": {"type": "keyword"},
            "error_handling": {"type": "object", "enabled": False},
            "imports": {"type": "keyword"},
            "line_numbers": {"type": "integer"}
        }
    }
}

CALL_GRAPH_CONFIG = {
    "index_name": "call_graph_embeddings",
    "max_depth": 5,
    "settings": {
        "index": {"knn": True},
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "embedding_knn": {"type": "knn_vector", "dimension": 3072},
            "func_name": {"type": "keyword"},
            "lookup_id": {"type": "keyword"},
            "file_path": {"type": "keyword"},
            "file_name": {"type": "keyword"},
            "repository_name": {"type": "keyword"},
            "module_name": {"type": "keyword"},
            "code": {"type": "text"},
            "code_with_line_numbers": {"type": "text"},
            "calls": {"type": "keyword"},
            "nested_call_ids": {"type": "keyword"},
            "start_line": {"type": "integer"},
            "end_line": {"type": "integer"},
            "is_async": {"type": "boolean"},
            "class_context": {"type": "keyword"},
            "has_error_handling": {"type": "boolean"},
            "depth_level": {"type": "integer"},
            "dependencies_count": {"type": "integer"},
            "dependents_count": {"type": "integer"},
            "decorators": {"type": "keyword"}
        }
    }
}

ANALYSIS_APPROACH_CONFIG = {
    "function_lookup_table": {
        "name": "Function Lookup Table",
        "description": "Uses function metadata and nested call relationships",
        "key": "function_lookup_table",
        "max_nested_depth": 3,
        "include_full_metadata": True,
        "prompt_type": "lookup_table"
    },
    "call_graph": {
        "name": "Call Graph",
        "description": "Uses function call graph for dependency analysis",
        "key": "call_graph",
        "max_graph_depth": 3,
        "include_minimal_metadata": True,
        "traverse_method": "bfs"
    }
}

CONFIDENCE_THRESHOLD = 0.8
MAX_ANALYSIS_ITERATIONS = 3
DEFAULT_FUNCTION_LIMIT = 5
DEFAULT_ANALYSIS_APPROACH = "function_lookup_table"
SUPPORTED_APPROACHES = ["function_lookup_table", "call_graph"]
MAX_CALL_GRAPH_DEPTH = 3

MAX_QUERY_LENGTH = 1000
MAX_CODE_LENGTH = 50000
MAX_FUNCTIONS_PER_SEARCH = 20

MAX_HISTORY_ENTRIES = 1000
MAX_HISTORY_FILE_SIZE_MB = 10
HISTORY_BACKUP_COUNT = 3

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Directory names skipped when scanning a repository for Python files
EXCLUDED_DIRS = {
    '.git', '.hg', '.svn', '__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache', '.tox', '.nox',
    'venv', '.venv', 'env', '.env', 'virtualenv', 'site-packages', 'node_modules', 'build', 'dist', '.eggs'
}
TEST_DIRS = {'test', 'tests', 'testing'}
