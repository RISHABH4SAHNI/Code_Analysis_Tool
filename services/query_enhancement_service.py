from abc import ABC, abstractmethod
import logging
import time
from typing import Dict, Any, Optional
import openai
import requests
import os
import re
from config.settings import OPENAI_CONFIG, PERPLEXITY_CONFIG, OLLAMA_CONFIG, MAX_QUERY_LENGTH

class QueryEnhancementService(ABC):
    @abstractmethod
    def enhance_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        pass

    @staticmethod
    def _clean_enhanced_query(enhanced: str, original: str) -> str:
        """Strip labels and quotes models often wrap around the rewritten query"""
        enhanced = re.sub(r'^\s*\**\s*enhanced query\s*:?\**\s*', '', enhanced or '', flags=re.IGNORECASE)
        enhanced = enhanced.strip().strip('"').strip()
        if not enhanced:
            return original
        return enhanced[:MAX_QUERY_LENGTH * 2]

class OpenAIQueryEnhancementService(QueryEnhancementService):
    def __init__(self, api_key: Optional[str] = None):
        if not api_key and not os.getenv('OPENAI_API_KEY'):
            raise ValueError("OpenAI API key is required")
        self.client = openai.OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        self.logger = logging.getLogger(__name__)
        self.config = OPENAI_CONFIG
        self.max_retries = self.config.get('max_retries', 3)
        self.timeout = self.config.get('timeout', 60)

    def enhance_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        if not query or not query.strip():
            return query
            
        query = query.strip()
        
        if len(query) > MAX_QUERY_LENGTH:
            self.logger.warning(f"Query too long ({len(query)} chars), truncating")
            query = query[:MAX_QUERY_LENGTH]
            
        prompt = self._build_enhancement_prompt(query, context)
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.config['chat_model'],
                    messages=[
                        {"role": "system", "content": "Enhance technical queries with relevant context while preserving original intent."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.config['enhancement_max_tokens'],
                    temperature=self.config.get('temperature', 0.1),
                    timeout=self.timeout
                )
                return self._clean_enhanced_query(response.choices[0].message.content, query)
            except openai.RateLimitError:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    self.logger.warning(f"Rate limit hit, waiting {wait_time}s")
                    time.sleep(wait_time)
                    continue
                self.logger.warning("Rate limit exceeded, returning original query")
                return query
            except openai.AuthenticationError as e:
                self.logger.error(f"Query enhancement failed, check OPENAI_API_KEY: {e}")
                return query
            except Exception as e:
                self.logger.error(f"Query enhancement failed (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    return query
        
        return query

    def _build_enhancement_prompt(self, query: str, context: Optional[Dict[str, Any]]) -> str:
        context_info = ""
        if context:
            if context.get('repo_name'):
                context_info += f"Repository: {context['repo_name']}\n"
            if context.get('tech_stack'):
                context_info += f"Tech Stack: {', '.join(context['tech_stack'])}\n"
            if context.get('priority'):
                context_info += f"Priority: {context['priority']}\n"
        
        return f"""
{context_info}
Original Query: {query}

Enhance this query with technical context for better code search. Preserve all original terms and add relevant technical concepts.
Reply with the enhanced query only."""

class PerplexityQueryEnhancementService(QueryEnhancementService):
    def __init__(self, api_key: Optional[str] = None):
        if not api_key and not os.getenv('PERPLEXITY_API_KEY'):
            raise ValueError("Perplexity API key is required")
        self.api_key = api_key or os.getenv('PERPLEXITY_API_KEY')
        self.config = PERPLEXITY_CONFIG
        self.logger = logging.getLogger(__name__)
        self.timeout = self.config.get('timeout', 30)

    def enhance_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        if not query or not query.strip():
            return query
            
        if len(query) > MAX_QUERY_LENGTH:
            query = query[:MAX_QUERY_LENGTH]
            
        prompt = self._build_enhancement_prompt(query, context)
        
        try:
            response = requests.post(
                f"{self.config['base_url']}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.config['chat_model'],
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": self.config['enhancement_max_tokens'],
                    "temperature": self.config.get('temperature', 0.1)
                },
                timeout=self.timeout
            )
            if response.status_code == 200:
                return self._clean_enhanced_query(response.json()['choices'][0]['message']['content'], query)
            self.logger.warning(f"Perplexity query enhancement returned HTTP {response.status_code}")
        except Exception as e:
            self.logger.error(f"Perplexity query enhancement failed: {e}")
        
        return query

    def _build_enhancement_prompt(self, query: str, context: Optional[Dict[str, Any]]) -> str:
        context_str = ""
        if context:
            context_items = []
            if context.get('repo_name'):
                context_items.append(f"Repository: {context['repo_name']}")
            if context.get('tech_stack'):
                context_items.append(f"Tech: {', '.join(context['tech_stack'])}")
            if context.get('priority'):
                context_items.append(f"Priority: {context['priority']}")
            context_str = " | ".join(context_items) + " | " if context_items else ""
        
        return (f"Rewrite this bug report as a code search query. Keep every original term and add relevant "
                f"technical terms. Reply with the query only.\n{context_str}{query}")

class OllamaQueryEnhancementService(QueryEnhancementService):
    def __init__(self, base_url: str = None):
        self.config = OLLAMA_CONFIG
        self.base_url = base_url or self.config['base_url']
        self.logger = logging.getLogger(__name__)
        self.timeout = self.config.get('timeout', 60)

    def enhance_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        if not query or not query.strip():
            return query
            
        if len(query) > MAX_QUERY_LENGTH:
            query = query[:MAX_QUERY_LENGTH]
            
        prompt = self._build_enhancement_prompt(query, context)
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.config['chat_model'],
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.config.get('temperature', 0.1),
                        "num_predict": self.config['enhancement_max_tokens']
                    }
                },
                timeout=self.timeout
            )
            if response.status_code == 200:
                return self._clean_enhanced_query(response.json().get('response', ''), query)
            self.logger.warning(f"Ollama query enhancement returned HTTP {response.status_code}")
        except Exception as e:
            self.logger.error(f"Ollama query enhancement failed: {e}")
        
        return query

    def _build_enhancement_prompt(self, query: str, context: Optional[Dict[str, Any]]) -> str:
        return ("Rewrite this bug report as a code search query. Keep every original term and add relevant "
                f"programming terms. Reply with the query only.\n{query}")
