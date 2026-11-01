from services.analysis_service import (
    AnalysisService,
    OpenAIAnalysisService,
    PerplexityAnalysisService, 
    OllamaAnalysisService
)
from services.query_enhancement_service import (
    QueryEnhancementService,
    OpenAIQueryEnhancementService,
    PerplexityQueryEnhancementService,
    OllamaQueryEnhancementService
)

class LLMFactory:
    @staticmethod
    def create_analysis_service(service_type: str, **kwargs) -> AnalysisService:
        service_type = service_type.lower()
        
        if service_type == "openai":
            return OpenAIAnalysisService(**kwargs)
        elif service_type == "perplexity":
            return PerplexityAnalysisService(**kwargs)
        elif service_type == "ollama":
            return OllamaAnalysisService(**kwargs)
        else:
            raise ValueError(f"Unknown LLM service type: {service_type}")
    
    @staticmethod
    def create_query_enhancer(service_type: str, **kwargs) -> QueryEnhancementService:
        service_type = service_type.lower()
        
        if service_type == "openai":
            return OpenAIQueryEnhancementService(**kwargs)
        elif service_type == "perplexity":
            return PerplexityQueryEnhancementService(**kwargs)
        elif service_type == "ollama":
            return OllamaQueryEnhancementService(**kwargs)
        else:
            raise ValueError(f"Unknown query enhancement service type: {service_type}")
    
    # Kept for callers of the older name
    create_query_enhancement_service = create_query_enhancer

    @staticmethod
    def get_available_services():
        return ["openai", "ollama", "perplexity"]
    
    @staticmethod
    def get_service_info(service_type: str) -> dict:
        service_info = {
            "openai": {"name": "OpenAI", "description": "High-quality analysis with OpenAI chat models"},
            "perplexity": {"name": "Perplexity AI", "description": "Analysis with Perplexity Sonar models"},
            "ollama": {"name": "Ollama Local", "description": "Local analysis with open-source models"}
        }
        return service_info.get(service_type.lower(), {
            "name": "Unknown", 
            "description": "Unknown service type"
        })
