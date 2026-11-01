from typing import List, Dict, Any, Optional
from opensearchpy.exceptions import ConnectionError, RequestError, NotFoundError
from opensearchpy.helpers import bulk
from models.function_model import FunctionMetadata, SearchResult, AnalysisApproach
from services.embedding_service import EmbeddingService
from services.opensearch_base import OpenSearchIndexService
from config.settings import INDEX_CONFIG, MAX_FUNCTIONS_PER_SEARCH

class VectorSearchService(OpenSearchIndexService):
    def __init__(self, embedding_service: EmbeddingService):
        super().__init__(embedding_service, INDEX_CONFIG['name'], INDEX_CONFIG)

    def _build_document(self, function: FunctionMetadata, embedding: List[float]) -> Dict[str, Any]:
        return {
            "embedding_knn": embedding,
            "func_name": function.name,
            "lookup_id": function.lookup_id,
            "file_path": function.file_path,
            "file_name": function.file_name or function.file_path.split('/')[-1],
            "repository_name": function.repository_name,
            "module_name": function.module_name,
            "code": function.code,
            "code_with_line_numbers": function.code_with_line_numbers,
            "calls": function.calls,
            "is_async": function.is_async,
            "start_line": function.start_line,
            "end_line": function.end_line,
            "class_context": function.class_context,
            "nested_call_ids": function.nested_call_ids,
            "decorators": function.decorators,
            "error_handling": function.error_handling,
            "imports": function.imports,
            "line_numbers": function.line_numbers
        }

    def _is_storable(self, function: FunctionMetadata, embedding: Optional[List[float]]) -> bool:
        if not embedding or len(embedding) != self.embedding_dimensions:
            self.logger.warning(f"Invalid embedding dimensions for function {function.name}. Expected: {self.embedding_dimensions}, Got: {len(embedding) if embedding else 0}")
            return False
        if not function.lookup_id or not function.name:
            self.logger.warning(f"Invalid function metadata: {function.name!r}")
            return False
        return True

    def store_function(self, function: FunctionMetadata, embedding: List[float]) -> bool:
        """Store a function with its embedding in OpenSearch"""
        if not self._is_storable(function, embedding):
            return False

        try:
            self.client.index(index=self.index_name, id=function.lookup_id, body=self._build_document(function, embedding))
            self.logger.debug(f"Stored function: {function.name} ({function.lookup_id})")
            return True
        except (RequestError, ConnectionError) as e:
            self.logger.error(f"OpenSearch error storing function {function.name}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to store function {function.name}: {e}")
            return False

    def store_functions_bulk(self, functions_with_embeddings: List[tuple]) -> Dict[str, Any]:
        """Store multiple functions efficiently using bulk operations"""
        actions = [
            {"_index": self.index_name, "_id": function.lookup_id, "_source": self._build_document(function, embedding)}
            for function, embedding in functions_with_embeddings
            if self._is_storable(function, embedding)
        ]

        if not actions:
            return {'success': 0, 'failed': 0, 'errors': ['No valid functions to store']}

        try:
            success, errors = bulk(self.client, actions, raise_on_error=False)
            self.logger.info(f"Bulk operation completed: {success} successful, {len(errors)} failed")
            return {'success': success, 'failed': len(errors), 'errors': [str(error) for error in errors]}
        except Exception as e:
            self.logger.error(f"Bulk store operation failed: {e}")
            return {'success': 0, 'failed': len(actions), 'errors': [str(e)]}

    def search_functions(self, query: str, limit: int = 5) -> List[SearchResult]:
        """Search for functions using vector similarity"""
        if not query or not query.strip():
            self.logger.warning("Empty query provided")
            return []

        limit = min(max(1, limit), MAX_FUNCTIONS_PER_SEARCH)

        query_embedding = self.embedding_service.create_query_embedding(query)
        if not query_embedding:
            self.logger.warning("Failed to create query embedding")
            return []

        if len(query_embedding) != self.embedding_dimensions:
            self.logger.error(f"Invalid query embedding dimensions: {len(query_embedding)}, expected: {self.embedding_dimensions}")
            return []

        try:
            response = self.client.search(index=self.index_name, body=self.knn_query(query_embedding, limit))
        except NotFoundError:
            self.logger.error(f"Index {self.index_name} not found")
            return []
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []

        results = []
        for hit in response.get('hits', {}).get('hits', []):
            function_metadata = self._build_function_metadata(hit['_source'])
            if function_metadata:
                results.append(SearchResult(
                    function_metadata=function_metadata,
                    relevance_score=hit['_score'],
                    search_method="vector_similarity",
                    match_type="semantic",
                    approach_used=AnalysisApproach.FUNCTION_LOOKUP_TABLE
                ))

        self.logger.info(f"Found {len(results)} search results for query")
        return results

    def get_function_by_id(self, lookup_id: str) -> Optional[FunctionMetadata]:
        """Retrieve a specific function by its lookup ID"""
        if not lookup_id:
            return None

        try:
            response = self.client.get(
                index=self.index_name,
                id=lookup_id,
                _source_excludes=["embedding_knn"]
            )
            return self._build_function_metadata(response['_source'])
        except NotFoundError:
            self.logger.debug(f"Function not found by ID {lookup_id}")
            return None
        except Exception as e:
            self.logger.error(f"Error retrieving function by ID {lookup_id}: {e}")
            return None

    def _build_function_metadata(self, source: Dict[str, Any]) -> Optional[FunctionMetadata]:
        """Build FunctionMetadata object from OpenSearch source data"""
        try:
            return FunctionMetadata(
                name=source.get('func_name', ''),
                lookup_id=source.get('lookup_id', ''),
                file_path=source.get('file_path', ''),
                repository_name=source.get('repository_name', ''),
                module_name=source.get('module_name', ''),
                nested_call_ids=source.get('nested_call_ids') or [],
                start_line=source.get('start_line', 0),
                end_line=source.get('end_line', 0),
                code=source.get('code', ''),
                is_async=source.get('is_async', False),
                class_context=source.get('class_context'),
                calls=source.get('calls') or [],
                imports=source.get('imports') or [],
                decorators=source.get('decorators') or [],
                error_handling=source.get('error_handling') or {},
                line_numbers=source.get('line_numbers') or [],
                file_name=source.get('file_name', ''),
                code_with_line_numbers=source.get('code_with_line_numbers', '')
            )
        except Exception as e:
            self.logger.error(f"Failed to build function metadata: {e}")
            return None

    def search_by_filters(self, filters: Dict[str, Any], limit: int = 10) -> List[SearchResult]:
        """Search functions by specific filters (file_path, repository_name, etc.)"""
        filter_conditions = []
        for key, value in (filters or {}).items():
            if key in ['func_name', 'file_path', 'file_name', 'repository_name', 'module_name', 'class_context']:
                filter_conditions.append({"term": {key: value}})
            elif key == 'is_async':
                filter_conditions.append({"term": {key: bool(value)}})
            elif key in ['start_line', 'end_line']:
                filter_conditions.append({"range": {key: {"gte": value}}})

        if not filter_conditions:
            return []

        search_body = {
            "query": {"bool": {"filter": filter_conditions}},
            "size": min(limit, MAX_FUNCTIONS_PER_SEARCH),
            "_source": {"excludes": ["embedding_knn"]}
        }

        try:
            response = self.client.search(index=self.index_name, body=search_body)
        except Exception as e:
            self.logger.error(f"Filter search failed: {e}")
            return []

        results = []
        for hit in response.get('hits', {}).get('hits', []):
            function_metadata = self._build_function_metadata(hit['_source'])
            if function_metadata:
                results.append(SearchResult(
                    function_metadata=function_metadata,
                    relevance_score=hit['_score'],
                    search_method="filter_search",
                    match_type="exact",
                    approach_used=AnalysisApproach.FUNCTION_LOOKUP_TABLE
                ))
        return results
