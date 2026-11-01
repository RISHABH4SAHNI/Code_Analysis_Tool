from factories.embedding_factory import EmbeddingFactory
from factories.llm_factory import LLMFactory
from services.vector_search_service import VectorSearchService
from services.call_graph_search_service import CallGraphSearchService
from core.call_graph_processor import CallGraphProcessor
from core.function_lookup import FunctionLookupTable
from core.repository_processor import RepositoryProcessor
from config.settings import SUPPORTED_APPROACHES, ANALYSIS_APPROACH_CONFIG

class ServiceFactory:
    def get_embedding_service(self, service_type: str = "openai"):
        return EmbeddingFactory.create_service(service_type)

    def get_search_service(self, embedding_service_type: str = "openai", embedding_service=None):
        return VectorSearchService(embedding_service or self.get_embedding_service(embedding_service_type))
    
    def get_call_graph_search_service(self, embedding_service_type: str = "openai", embedding_service=None,
                                      call_graph_processor=None):
        return CallGraphSearchService(embedding_service or self.get_embedding_service(embedding_service_type),
                                      call_graph_processor)
    
    def get_call_graph_processor(self):
        return CallGraphProcessor()
    
    def get_query_enhancer(self, service_type: str = "openai"):
        return LLMFactory.create_query_enhancer(service_type)

    def get_analysis_service(self, service_type: str = "openai"):
        return LLMFactory.create_analysis_service(service_type)

    def get_repository_processor(self):
        return RepositoryProcessor()

    def get_lookup_table(self):
        return FunctionLookupTable()
    
    def get_services_for_approach(self, approach: str, embedding_service_type: str = "openai", llm_service_type: str = "openai"):
        """Get all required services for a specific analysis approach"""
        if not self.validate_approach(approach):
            raise ValueError(f"Unsupported analysis approach: {approach}. Supported: {SUPPORTED_APPROACHES}")
        
        embedding_service = self.get_embedding_service(embedding_service_type)
        services = {
            'embedding_service': embedding_service,
            'query_enhancer': self.get_query_enhancer(llm_service_type),
            'analysis_service': self.get_analysis_service(llm_service_type),
            'repository_processor': self.get_repository_processor(),
        }
        
        if approach == "call_graph":
            # The search service and the engine share one processor, so the graph it builds is the one searched
            call_graph_processor = self.get_call_graph_processor()
            services.update({
                'search_service': self.get_call_graph_search_service(
                    embedding_service=embedding_service, call_graph_processor=call_graph_processor),
                'call_graph_processor': call_graph_processor
            })
        else:
            services.update({
                'search_service': self.get_search_service(embedding_service=embedding_service),
                'lookup_table': self.get_lookup_table()
            })
        
        return services
    
    def validate_approach(self, approach: str) -> bool:
        """Validate if the analysis approach is supported"""
        return approach in SUPPORTED_APPROACHES
    
    def get_supported_approaches(self):
        """Get list of supported analysis approaches"""
        return [
            {
                'key': key,
                'name': ANALYSIS_APPROACH_CONFIG[key]['name'],
                'description': ANALYSIS_APPROACH_CONFIG[key]['description']
            }
            for key in SUPPORTED_APPROACHES
        ]
