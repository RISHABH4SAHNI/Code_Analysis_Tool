import logging
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import openai
import requests
import os
from models.function_model import FunctionMetadata
from config.settings import OPENAI_CONFIG, OLLAMA_CONFIG, PERPLEXITY_CONFIG, MAX_CODE_LENGTH

class EmbeddingService(ABC):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_retries = 3
        self.retry_delay = 1.0
        
    @abstractmethod
    def create_embedding(self, function: FunctionMetadata, include_metadata: bool = True) -> Optional[List[float]]:
        pass

    @abstractmethod
    def create_query_embedding(self, query: str) -> Optional[List[float]]:
        pass

    @abstractmethod
    def create_call_graph_embedding(self, function: FunctionMetadata) -> Optional[List[float]]:
        pass
    
    @abstractmethod
    def get_embedding_dimensions(self) -> int:
        pass
    
    def validate_embedding(self, embedding: Optional[List[float]]) -> bool:
        if not embedding:
            return False
        expected_dims = self.get_embedding_dimensions()
        return len(embedding) == expected_dims

    def _truncate_content(self, content: str, max_length: int = None) -> str:
        if max_length is None:
            max_length = MAX_CODE_LENGTH
        
        if len(content) <= max_length:
            return content
            
        truncated = content[:max_length]
        last_newline = truncated.rfind('\n')
        if last_newline > max_length * 0.8:
            truncated = truncated[:last_newline]
            
        return truncated + "\n... (truncated)"

class OpenAIEmbeddingService(EmbeddingService):
    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key is required")
        self.client = openai.OpenAI(api_key=api_key)
        self.model = OPENAI_CONFIG['model']
        self.max_tokens = OPENAI_CONFIG['embedding_max_tokens']
        self.timeout = OPENAI_CONFIG.get('timeout', 60)
        self.embedding_dimensions = OPENAI_CONFIG['embedding_dimensions']
    
    def get_embedding_dimensions(self) -> int:
        return self.embedding_dimensions

    def create_embedding(self, function: FunctionMetadata, include_metadata: bool = True) -> Optional[List[float]]:
        if not function or not function.code:
            self.logger.warning("Invalid function metadata provided")
            return None
            
        context = self._build_context(function, include_metadata)
        # Code averages roughly 3 characters per token; stay under the model's input limit
        context = self._truncate_content(context, self.max_tokens * 3)
        return self._embed_text(context, f"function {function.name}")

    def create_query_embedding(self, query: str) -> Optional[List[float]]:
        if not query or not query.strip():
            self.logger.warning("Empty query provided for embedding")
            return None
        return self._embed_text(query.strip(), "query")

    def create_call_graph_embedding(self, function: FunctionMetadata) -> Optional[List[float]]:
        if not function or not function.code:
            self.logger.warning("Invalid function metadata provided")
            return None
        context = self._build_call_graph_context(function)
        return self._embed_text(context, f"call_graph {function.name}")

    def _build_context(self, function: FunctionMetadata, include_metadata: bool) -> str:
        context_parts = [f"Function: {function.name}"]
        
        if include_metadata:
            context_parts.extend([
                f"File: {function.file_name}",
                f"Repository: {function.repository_name}",
                f"Module: {function.module_name}"
            ])
            
            if function.class_context:
                context_parts.append(f"Class: {function.class_context}")
            if function.is_async:
                context_parts.append("Type: async function")
            if function.decorators:
                context_parts.append(f"Decorators: {', '.join(function.decorators)}")
            if function.calls:
                context_parts.append(f"Calls: {', '.join(function.calls[:10])}")
                
        context_parts.extend([
            "Code:",
            function.code_with_line_numbers or function.code
        ])
        
        return '\n'.join(context_parts)

    def _build_call_graph_context(self, function: FunctionMetadata) -> str:
        context_parts = [
            f"Function: {function.name}",
            f"Module: {function.module_name}"
        ]
        
        if function.class_context:
            context_parts.append(f"Class: {function.class_context}")
        if function.calls:
            context_parts.append(f"Dependencies: {', '.join(function.calls[:5])}")
            
        if function.code:
            lines = function.code.split('\n')
            essential_lines = []
            for line in lines[:20]:
                stripped = line.strip()
                if (stripped.startswith(('def ', 'async def ', 'class ', 'import ', 'from ')) or
                    any(call in line for call in function.calls[:5]) or
                    'return' in stripped):
                    essential_lines.append(line)
            
            context_parts.extend(["Code:", '\n'.join(essential_lines)])
        
        return '\n'.join(context_parts)

    def _embed_text(self, text: str, context_description: str = "text") -> Optional[List[float]]:
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model, 
                    input=text,
                    timeout=self.timeout
                )
                embedding = response.data[0].embedding
                
                if self.validate_embedding(embedding):
                    return embedding
                else:
                    self.logger.error(f"Invalid embedding dimensions for {context_description}")
                    return None
                    
            except openai.RateLimitError:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    self.logger.warning(f"Rate limit hit for {context_description}, retrying in {wait_time}s")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"Rate limit exceeded for {context_description}")
                    return None
            except Exception as e:
                self.logger.error(f"Failed to create embedding for {context_description}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    return None
        return None

class OllamaEmbeddingService(EmbeddingService):
    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        super().__init__()
        self.base_url = base_url or OLLAMA_CONFIG['base_url']
        self.model = model or OLLAMA_CONFIG['embedding_model']
        self.timeout = OLLAMA_CONFIG.get('timeout', 60)
        self.embedding_dimensions = OLLAMA_CONFIG['embedding_dimensions']
    
    def get_embedding_dimensions(self) -> int:
        return self.embedding_dimensions

    def create_embedding(self, function: FunctionMetadata, include_metadata: bool = True) -> Optional[List[float]]:
        if not function or not function.code:
            self.logger.warning("Invalid function metadata provided")
            return None
            
        context = self._build_context(function, include_metadata)
        context = self._truncate_content(context)
        return self._embed_text(context, f"function {function.name}")

    def create_query_embedding(self, query: str) -> Optional[List[float]]:
        if not query or not query.strip():
            self.logger.warning("Empty query provided for embedding")
            return None
        return self._embed_text(query.strip(), "query")

    def create_call_graph_embedding(self, function: FunctionMetadata) -> Optional[List[float]]:
        return self.create_embedding(function, include_metadata=False)

    def _build_context(self, function: FunctionMetadata, include_metadata: bool) -> str:
        context_parts = [f"Function: {function.name}"]
        
        if include_metadata:
            context_parts.extend([
                f"File: {function.file_name}",
                f"Repository: {function.repository_name}",
                f"Module: {function.module_name}"
            ])
            
        context_parts.extend([
            "Code:",
            function.code_with_line_numbers or function.code
        ])
        return '\n'.join(context_parts)

    def _embed_text(self, text: str, context_description: str = "text") -> Optional[List[float]]:
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                    timeout=self.timeout
                )
                if response.status_code == 200:
                    embedding = response.json().get("embedding")
                    if self.validate_embedding(embedding):
                        return embedding
                    else:
                        self.logger.error(f"Invalid Ollama embedding dimensions for {context_description}")
                        return None
                else:
                    self.logger.warning(f"Ollama API returned {response.status_code} for {context_description}")
            except requests.exceptions.Timeout:
                self.logger.warning(f"Ollama API timeout for {context_description} (attempt {attempt + 1})")
            except Exception as e:
                self.logger.error(f"Ollama embedding failed for {context_description}: {e}")
                
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay * (attempt + 1))
        return None

class PerplexityEmbeddingService(EmbeddingService):
    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        api_key = api_key or os.getenv('PERPLEXITY_API_KEY')
        if not api_key:
            raise ValueError("Perplexity API key is required")
        if not PERPLEXITY_CONFIG['embedding_model']:
            raise ValueError(
                "Perplexity embeddings need PERPLEXITY_EMBEDDING_MODEL (and PERPLEXITY_EMBEDDING_DIMENSIONS) "
                "set to a model your Perplexity account can use; otherwise choose openai or ollama embeddings"
            )
        self.api_key = api_key
        self.base_url = PERPLEXITY_CONFIG['base_url']
        self.model = PERPLEXITY_CONFIG['embedding_model']
        self.timeout = PERPLEXITY_CONFIG.get('timeout', 30)
        self.embedding_dimensions = PERPLEXITY_CONFIG['embedding_dimensions']
    
    def get_embedding_dimensions(self) -> int:
        return self.embedding_dimensions

    def create_embedding(self, function: FunctionMetadata, include_metadata: bool = True) -> Optional[List[float]]:
        if not function or not function.code:
            self.logger.warning("Invalid function metadata provided")
            return None
            
        context = self._build_context(function, include_metadata)
        context = self._truncate_content(context)
        return self._embed_text(context, f"function {function.name}")

    def create_query_embedding(self, query: str) -> Optional[List[float]]:
        if not query or not query.strip():
            self.logger.warning("Empty query provided for embedding")
            return None
        return self._embed_text(query.strip(), "query")

    def create_call_graph_embedding(self, function: FunctionMetadata) -> Optional[List[float]]:
        return self.create_embedding(function, include_metadata=False)

    def _build_context(self, function: FunctionMetadata, include_metadata: bool) -> str:
        if include_metadata:
            return f"File: {function.file_name}\nFunction: {function.name}\nCode:\n{function.code_with_line_numbers or function.code}"
        else:
            return f"Function: {function.name}\nCode:\n{function.code_with_line_numbers or function.code}"

    def _embed_text(self, text: str, context_description: str = "text") -> Optional[List[float]]:
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={"model": self.model, "input": text},
                    timeout=self.timeout
                )
                if response.status_code == 200:
                    data = response.json()
                    if 'data' in data and len(data['data']) > 0:
                        embedding = data['data'][0]['embedding']
                        if self.validate_embedding(embedding):
                            return embedding
                        else:
                            self.logger.error(f"Invalid Perplexity embedding dimensions for {context_description}")
                            return None
                    else:
                        self.logger.error(f"Invalid Perplexity API response format for {context_description}")
                elif response.status_code == 429:
                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_delay * (2 ** attempt)
                        self.logger.warning(f"Perplexity rate limit for {context_description}, retrying in {wait_time}s")
                        time.sleep(wait_time)
                    else:
                        self.logger.error(f"Perplexity rate limit exceeded for {context_description}")
                        return None
                else:
                    self.logger.warning(f"Perplexity API returned {response.status_code} for {context_description}")
            except requests.exceptions.Timeout:
                self.logger.warning(f"Perplexity API timeout for {context_description} (attempt {attempt + 1})")
            except Exception as e:
                self.logger.error(f"Perplexity embedding failed for {context_description}: {e}")
                
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay)
        return None
