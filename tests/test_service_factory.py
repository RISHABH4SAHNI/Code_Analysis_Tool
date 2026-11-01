import pytest
from unittest.mock import Mock, patch
from factories.service_factory import ServiceFactory
from config.settings import SUPPORTED_APPROACHES

class TestServiceFactory:
    
    def setup_method(self):
        """Setup test fixtures"""
        self.factory = ServiceFactory()
    
    def test_factory_initialization(self):
        """Test service factory initializes correctly"""
        assert self.factory is not None
        assert hasattr(self.factory, 'get_services_for_approach')
    
    def test_validate_approach_valid(self):
        """Test validation of valid analysis approaches"""
        for approach in SUPPORTED_APPROACHES:
            assert self.factory.validate_approach(approach) == True
    
    def test_validate_approach_invalid(self):
        """Test validation of invalid analysis approaches"""
        invalid_approaches = ["invalid", "unknown", "test"]
        for approach in invalid_approaches:
            assert self.factory.validate_approach(approach) == False
    
    @patch('factories.embedding_factory.EmbeddingFactory.create_service')
    @patch('factories.llm_factory.LLMFactory.create_analysis_service')
    @patch('factories.llm_factory.LLMFactory.create_query_enhancer')
    def test_get_services_for_call_graph_approach(self, 
                                                  mock_query_enhancer,
                                                  mock_analysis_service, 
                                                  mock_embedding_service):
        """Test service creation for call_graph approach"""
        # Setup mocks
        mock_embedding_service.return_value = Mock()
        mock_analysis_service.return_value = Mock()
        mock_query_enhancer.return_value = Mock()
        
        services = self.factory.get_services_for_approach(
            "call_graph", "openai", "openai"
        )
        
        # Check all required services are present
        required_services = [
            'repository_processor', 'embedding_service', 'search_service',
            'query_enhancer', 'analysis_service', 'call_graph_processor'
        ]
        
        for service_name in required_services:
            assert service_name in services
            assert services[service_name] is not None
        
        # One embedding service, shared by the engine and the search service
        mock_embedding_service.assert_called_once_with("openai")
        assert services['search_service'].embedding_service is services['embedding_service']
        # The search service must search the same graph the engine exposes
        assert services['search_service'].call_graph_processor is services['call_graph_processor']
        mock_analysis_service.assert_called_once_with("openai")
        mock_query_enhancer.assert_called_once_with("openai")
    
    @patch('factories.embedding_factory.EmbeddingFactory.create_service')
    @patch('factories.llm_factory.LLMFactory.create_analysis_service')
    @patch('factories.llm_factory.LLMFactory.create_query_enhancer')
    def test_get_services_for_lookup_table_approach(self,
                                                     mock_query_enhancer,
                                                     mock_analysis_service,
                                                     mock_embedding_service):
        """Test service creation for function_lookup_table approach"""
        # Setup mocks
        mock_embedding_service.return_value = Mock()
        mock_analysis_service.return_value = Mock()
        mock_query_enhancer.return_value = Mock()
        
        services = self.factory.get_services_for_approach(
            "function_lookup_table", "openai", "openai"
        )
        
        # Check required services for the lookup table approach
        required_services = [
            'repository_processor', 'embedding_service', 'search_service',
            'query_enhancer', 'analysis_service', 'lookup_table'
        ]
        
        for service_name in required_services:
            assert service_name in services
            assert services[service_name] is not None
        
        # Should not have call_graph_processor for the lookup table approach
        assert 'call_graph_processor' not in services
    
    def test_get_supported_approaches(self):
        """Test getting supported approaches with descriptions"""
        approaches = self.factory.get_supported_approaches()
        
        assert isinstance(approaches, list)
        assert len(approaches) == len(SUPPORTED_APPROACHES)
        
        # Check each approach has required fields
        for approach in approaches:
            assert 'key' in approach
            assert 'name' in approach
            assert 'description' in approach
            assert approach['key'] in SUPPORTED_APPROACHES
    
    def test_unsupported_approach_error(self):
        """Test error handling for unsupported approach"""
        with pytest.raises(ValueError, match="Unsupported analysis approach"):
            self.factory.get_services_for_approach(
                "invalid_approach", "openai", "openai"
            )
    
    @patch('factories.embedding_factory.EmbeddingFactory.create_service')
    def test_different_embedding_services(self, mock_embedding_service):
        """Test creation with different embedding services"""
        mock_embedding_service.return_value = Mock()
        
        embedding_services = ["openai", "ollama", "perplexity"]
        
        for service in embedding_services:
            services = self.factory.get_services_for_approach(
                "call_graph", service, "openai"
            )
            assert services is not None
            assert 'embedding_service' in services
    
    @patch('factories.llm_factory.LLMFactory.create_analysis_service')
    def test_different_llm_services(self, mock_analysis_service):
        """Test creation with different LLM services"""
        mock_analysis_service.return_value = Mock()
        
        llm_services = ["openai", "perplexity", "ollama"]
        
        for service in llm_services:
            services = self.factory.get_services_for_approach(
                "call_graph", "openai", service
            )
            assert services is not None
            assert 'analysis_service' in services
