import copy
import logging
from typing import Any, Dict, List, Optional
from opensearchpy import OpenSearch
from opensearchpy.connection import RequestsHttpConnection
from opensearchpy.exceptions import NotFoundError
from config.settings import get_opensearch_config
from services.embedding_service import EmbeddingService

class OpenSearchIndexService:
    """Shared OpenSearch client and index management for the search services"""

    def __init__(self, embedding_service: EmbeddingService, index_name: str, index_config: Dict[str, Any]):
        self.logger = logging.getLogger(self.__class__.__name__)
        config = get_opensearch_config()
        # Creating the client does not connect; connection errors surface on first use
        self.client = OpenSearch(
            config['hosts'],
            http_auth=config['http_auth'],
            use_ssl=config['use_ssl'],
            verify_certs=config.get('verify_certs', False),
            connection_class=RequestsHttpConnection,
            timeout=config.get('timeout', 30),
            max_retries=config.get('max_retries', 3),
            retry_on_timeout=config.get('retry_on_timeout', True)
        )
        self.embedding_service = embedding_service
        self.embedding_dimensions = embedding_service.get_embedding_dimensions()
        self.index_name = index_name
        self.index_config = index_config
        self.repository_name: Optional[str] = None

    def setup_index(self):
        """Create the index for the active embedding size; recreate it if it was built for another size"""
        if self.client.indices.exists(index=self.index_name):
            existing_dims = self._existing_dimensions()
            if existing_dims == self.embedding_dimensions:
                self.logger.debug(f"Index {self.index_name} already exists")
                return
            self.logger.warning(
                f"Index {self.index_name} holds {existing_dims}-dimension vectors but the embedding service "
                f"produces {self.embedding_dimensions}; recreating it"
            )
            self.client.indices.delete(index=self.index_name)

        mappings = copy.deepcopy(self.index_config['mappings'])
        mappings['properties']['embedding_knn']['dimension'] = self.embedding_dimensions
        body = {"settings": self.index_config['settings'], "mappings": mappings}
        self.client.indices.create(index=self.index_name, body=body)
        self.logger.info(f"Created index {self.index_name} with {self.embedding_dimensions} dimensions")

    def _existing_dimensions(self) -> Optional[int]:
        try:
            mapping = self.client.indices.get_mapping(index=self.index_name)
            properties = mapping[self.index_name]['mappings'].get('properties', {})
            return properties.get('embedding_knn', {}).get('dimension')
        except Exception as e:
            self.logger.warning(f"Could not read mapping for {self.index_name}: {e}")
            return None

    def delete_repository(self, repository_name: str) -> int:
        """Remove a repository's documents so re-processing does not leave stale functions behind"""
        try:
            response = self.client.delete_by_query(
                index=self.index_name,
                body={"query": {"term": {"repository_name": repository_name}}},
                refresh=True,
                conflicts='proceed'
            )
            return response.get('deleted', 0)
        except NotFoundError:
            return 0

    def refresh(self):
        self.client.indices.refresh(index=self.index_name)

    def knn_query(self, query_embedding: List[float], limit: int) -> Dict[str, Any]:
        """kNN query, restricted to the processed repository when one is set"""
        knn = {"knn": {"embedding_knn": {"vector": query_embedding, "k": min(limit * 4, 100)}}}
        if self.repository_name:
            query = {"bool": {"must": [knn], "filter": [{"term": {"repository_name": self.repository_name}}]}}
        else:
            query = knn
        return {
            "query": query,
            "size": limit,
            "_source": {"excludes": ["embedding_knn"]}
        }

    def health_check(self) -> Dict[str, Any]:
        """Check the health of the OpenSearch connection and index"""
        try:
            cluster_health = self.client.cluster.health(timeout='5s')
            return {
                'cluster_status': cluster_health.get('status'),
                'index_exists': self.client.indices.exists(index=self.index_name),
                'embedding_dimensions': self.embedding_dimensions
            }
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {'error': str(e)}

    def is_healthy(self) -> bool:
        return self.health_check().get('cluster_status') in ('green', 'yellow')

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the search index"""
        try:
            if not self.client.indices.exists(index=self.index_name):
                return {'error': 'Index does not exist', 'index_name': self.index_name}
            stats = self.client.indices.stats(index=self.index_name)
            totals = stats['indices'][self.index_name]['total']
            return {
                'document_count': totals['docs']['count'],
                'store_size': totals['store']['size_in_bytes'],
                'index_name': self.index_name,
                'embedding_dimensions': self.embedding_dimensions
            }
        except Exception as e:
            self.logger.error(f"Failed to get index stats: {e}")
            return {'error': str(e), 'index_name': self.index_name}
