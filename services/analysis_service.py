import logging
import os
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import openai
import requests
from models.function_model import AnalysisRequest, AnalysisResult, AnalysisApproach
from config.settings import OPENAI_CONFIG, PERPLEXITY_CONFIG, OLLAMA_CONFIG

SYSTEM_PROMPT = "You are a senior code expert who provides detailed technical analysis with exact line number references."

SECTION_NAMES = [
    'CONFIDENCE_SCORE', 'ANALYSIS', 'GRAPH_INSIGHTS', 'ISSUES_FOUND', 'SUGGESTED_FIXES', 'ADDITIONAL_CONTEXT_NEEDED'
]
# Matches a section header at the start of a line, tolerating markdown like "**ANALYSIS:**" or "## ANALYSIS:"
SECTION_PATTERN = re.compile(r'^[#*\s]*(' + '|'.join(SECTION_NAMES) + r')\**\s*:\**\s*', re.MULTILINE)
UUID_PATTERN = re.compile(r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}')
EMPTY_VALUES = {'', 'none', 'n/a', 'na', '-', '[]'}

class AnalysisService(ABC):
    """Builds the analysis prompt, sends it to an LLM and parses the structured reply.
    Subclasses only implement the call to their provider."""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def _complete(self, system_prompt: str, user_prompt: str) -> str:
        """Send the prompt to the LLM and return its reply text"""

    def analyze_issue(self, request: AnalysisRequest) -> AnalysisResult:
        context = self._build_analysis_context(request)
        if request.approach == AnalysisApproach.CALL_GRAPH:
            prompt = self._build_call_graph_prompt(request, context)
        else:
            prompt = self._build_lookup_table_prompt(request, context)

        try:
            response = self._complete(SYSTEM_PROMPT, prompt)
        except Exception as e:
            self.logger.error(f"LLM request failed: {e}")
            return AnalysisResult(
                analysis=f"Analysis failed: {e}",
                confidence_score=0.0,
                functions_analyzed=len(request.functions),
                additional_context_needed=[],
                suggested_fixes=[],
                approach_used=request.approach
            )
        return self._parse_analysis_response(response, request)

    def _build_lookup_table_prompt(self, request: AnalysisRequest, context: str) -> str:
        return f"""
You are a code expert analyzing an issue using FUNCTION LOOKUP TABLE approach.
Analyze the provided functions and their nested dependencies to identify the root cause.

Query: {request.enhanced_query}

Available Functions with Dependencies:
{context}

IMPORTANT: Reference exact line numbers from the code when identifying issues.

Provide your analysis in this format:
CONFIDENCE_SCORE: [a number from 0.0 to 1.0]
ANALYSIS: [Detailed analysis with exact line number references]
ISSUES_FOUND: [List specific issues with file:line format]
SUGGESTED_FIXES: [Concrete fixes with line numbers, one per line]
ADDITIONAL_CONTEXT_NEEDED: [Lookup IDs of listed nested functions whose code you still need, or None]
"""

    def _build_call_graph_prompt(self, request: AnalysisRequest, context: str) -> str:
        call_graph_info = ""
        if request.call_graph_context:
            call_graph_info = f"\nCall Graph Context:\n{request.call_graph_context}"

        return f"""
You are a code expert analyzing an issue using CALL GRAPH approach.
Use the call graph to understand function dependencies and execution flow.

Query: {request.enhanced_query}

Primary Functions and Their Call Graph Context:
{context}
{call_graph_info}

IMPORTANT: Leverage the call graph to trace execution flow and identify cascading issues.
Reference exact line numbers when identifying problems.

Provide your analysis in this format:
CONFIDENCE_SCORE: [a number from 0.0 to 1.0]
ANALYSIS: [Analysis considering call graph dependencies with line numbers]
GRAPH_INSIGHTS: [How the call graph reveals the issue]
ISSUES_FOUND: [Issues with execution flow and line references]
SUGGESTED_FIXES: [Fixes considering dependency impact, one per line]
ADDITIONAL_CONTEXT_NEEDED: [Lookup IDs of listed dependency functions whose code you still need, or None]
"""

    def _build_analysis_context(self, request: AnalysisRequest) -> str:
        context_parts = []

        for i, result in enumerate(request.functions, 1):
            func = result.function_metadata
            context_parts.append(f"""
Function {i}: {func.name}
File: {func.file_name} ({func.file_path})
Lines: {func.start_line}-{func.end_line}
Lookup ID: {func.lookup_id}
Type: {'async' if func.is_async else 'sync'}
Class: {func.class_context or 'None'}
Decorators: {', '.join(func.decorators) if func.decorators else 'None'}
Calls: {', '.join(func.calls[:10]) if func.calls else 'None'}
Error Handling: {'Yes' if func.error_handling.get('has_try_catch') else 'No'}

Code with line numbers:
{func.code_with_line_numbers}
""")

            if result.nested_functions:
                context_parts.extend(self._describe_nested_functions(request.approach, func.name, result.nested_functions))

        return '\n'.join(context_parts)

    def _describe_nested_functions(self, approach: AnalysisApproach, func_name: str,
                                   nested_functions: Dict[str, Any]) -> List[str]:
        if approach == AnalysisApproach.CALL_GRAPH:
            graph_context = nested_functions.get('call_graph_context')
            if not graph_context:
                return []
            lines = [
                f"\nCall Graph Context for {func_name}:",
                f"  - Dependencies ({graph_context.get('dependency_count', 0)}): "
                f"{', '.join(graph_context.get('dependency_names', [])) or 'None'}",
                f"  - Dependency lookup IDs: {', '.join(nested_functions.get('dependency_functions', [])) or 'None'}",
                f"  - Graph Depth: {graph_context.get('graph_depth', 0)}",
                f"  - Files Involved: {', '.join(graph_context.get('files_involved', []))}",
                f"  - Has Error Handling: {graph_context.get('has_error_handling', False)}"
            ]
            return lines

        lines = [f"\nNested Function Dependencies for {func_name} (code not shown; request by lookup ID):"]
        for nested_id, nested_info in nested_functions.items():
            if isinstance(nested_info, dict) and 'function_name' in nested_info:
                lines.append(f"  - {nested_info['function_name']} ({nested_info.get('file_path', 'Unknown')}) "
                             f"lookup ID: {nested_id}")
        return lines

    def _parse_sections(self, response: str) -> Dict[str, str]:
        sections = {}
        matches = list(SECTION_PATTERN.finditer(response))
        for i, match in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(response)
            sections[match.group(1)] = response[match.end():end].strip()
        return sections

    def _parse_confidence(self, text: str) -> float:
        match = re.search(r'\d+(?:\.\d+)?', text or '')
        if not match:
            return 0.5
        value = float(match.group())
        if value > 1:
            value = value / 100  # "85" or "85%"
        return min(max(value, 0.0), 1.0)

    def _parse_analysis_response(self, response: str, request: AnalysisRequest) -> AnalysisResult:
        sections = self._parse_sections(response)
        known_ids = {r.function_metadata.lookup_id for r in request.functions}

        additional_context_needed = []
        context_text = sections.get('ADDITIONAL_CONTEXT_NEEDED', '')
        if context_text.strip().lower().strip('.[]') not in EMPTY_VALUES:
            for lookup_id in UUID_PATTERN.findall(context_text):
                if lookup_id not in known_ids and lookup_id not in additional_context_needed:
                    additional_context_needed.append(lookup_id)

        suggested_fixes = []
        fixes_text = sections.get('SUGGESTED_FIXES', '')
        if fixes_text.strip().lower().strip('.') not in EMPTY_VALUES:
            for line in fixes_text.splitlines():
                fix = re.sub(r'^\s*(?:[-*•]|\d+[.)])\s*', '', line).strip()
                if fix:
                    suggested_fixes.append({"description": fix, "type": "code_change"})

        graph_context = None
        if request.approach == AnalysisApproach.CALL_GRAPH and sections.get('GRAPH_INSIGHTS'):
            graph_context = {'insights': sections['GRAPH_INSIGHTS']}

        return AnalysisResult(
            analysis=response,
            confidence_score=self._parse_confidence(sections.get('CONFIDENCE_SCORE', '')),
            functions_analyzed=len(request.functions),
            additional_context_needed=additional_context_needed,
            suggested_fixes=suggested_fixes,
            approach_used=request.approach,
            graph_context=graph_context
        )

class OpenAIAnalysisService(AnalysisService):
    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key is required")
        self.config = OPENAI_CONFIG
        self.client = openai.OpenAI(api_key=api_key, timeout=self.config['timeout'], max_retries=self.config['max_retries'])

    def _complete(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.config['chat_model'],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=self.config['max_tokens'],
            temperature=self.config['temperature']
        )
        return response.choices[0].message.content or ""

class PerplexityAnalysisService(AnalysisService):
    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        self.api_key = api_key or os.getenv('PERPLEXITY_API_KEY')
        if not self.api_key:
            raise ValueError("Perplexity API key is required")
        self.config = PERPLEXITY_CONFIG

    def _complete(self, system_prompt: str, user_prompt: str) -> str:
        response = requests.post(
            f"{self.config['base_url']}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.config['chat_model'],
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "max_tokens": self.config['max_tokens'],
                "temperature": self.config['temperature']
            },
            timeout=self.config['timeout']
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']

class OllamaAnalysisService(AnalysisService):
    def __init__(self, base_url: Optional[str] = None):
        super().__init__()
        self.config = OLLAMA_CONFIG
        self.base_url = base_url or self.config['base_url']

    def _complete(self, system_prompt: str, user_prompt: str) -> str:
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.config['chat_model'],
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "stream": False,
                "options": {"temperature": self.config['temperature'], "num_predict": self.config['max_tokens']}
            },
            timeout=self.config['timeout']
        )
        response.raise_for_status()
        return response.json().get('message', {}).get('content', '')
