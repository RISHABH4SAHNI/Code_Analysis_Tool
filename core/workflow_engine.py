import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from factories.service_factory import ServiceFactory
from models.function_model import AnalysisRequest, AnalysisResult, SearchResult, AnalysisApproach
from services.call_graph_search_service import CallGraphSearchService
from utils.history_manager import HistoryManager
from config.settings import (
    SUPPORTED_APPROACHES, DEFAULT_ANALYSIS_APPROACH, MAX_CALL_GRAPH_DEPTH,
    CONFIDENCE_THRESHOLD, MAX_ANALYSIS_ITERATIONS, DEFAULT_FUNCTION_LIMIT
)

class WorkflowEngine:
    def __init__(self,
                 embedding_service_type: str = "openai",
                 llm_service_type: str = "openai",
                 analysis_approach: str = DEFAULT_ANALYSIS_APPROACH,
                 history_manager: Optional[HistoryManager] = None):
        self.logger = logging.getLogger(__name__)
        self.service_factory = ServiceFactory()
        self.embedding_service_type = embedding_service_type
        self.llm_service_type = llm_service_type
        self.analysis_approach = analysis_approach
        self.processed_repo_path: Optional[str] = None

        if not self.service_factory.validate_approach(analysis_approach):
            raise ValueError(f"Unsupported analysis approach: {analysis_approach}. Supported: {SUPPORTED_APPROACHES}")

        self._load_services(analysis_approach)
        self.history_manager = history_manager or HistoryManager()

        self.logger.info(f"WorkflowEngine initialized with approach: {analysis_approach}")

    def _load_services(self, approach: str):
        self.services = self.service_factory.get_services_for_approach(
            approach, self.embedding_service_type, self.llm_service_type
        )
        self.repo_processor = self.services['repository_processor']
        self.embedding_service = self.services['embedding_service']
        self.search_service = self.services['search_service']
        self.query_enhancer = self.services['query_enhancer']
        self.analysis_service = self.services['analysis_service']
        self.call_graph_processor = self.services.get('call_graph_processor')
        self.lookup_table = self.services.get('lookup_table')

    def _approach_enum(self) -> AnalysisApproach:
        return AnalysisApproach(self.analysis_approach)

    def set_llm_service(self, llm_service_type: str):
        """Switch the LLM used for query enhancement and analysis without re-indexing"""
        self.analysis_service = self.service_factory.get_analysis_service(llm_service_type)
        self.query_enhancer = self.service_factory.get_query_enhancer(llm_service_type)
        self.llm_service_type = llm_service_type

    def process_repository(self, repo_path: str, include_tests: bool = False,
                           max_functions: Optional[int] = None) -> Dict[str, Any]:
        if not os.path.exists(repo_path):
            raise ValueError(f"Repository path does not exist: {repo_path}")

        if not os.path.isdir(repo_path):
            raise ValueError(f"Path is not a directory: {repo_path}")

        self.logger.info(f"Processing repository: {repo_path} using {self.analysis_approach} approach")

        functions = self.repo_processor.process_repository(repo_path, include_tests=include_tests,
                                                           max_functions=max_functions)
        if not functions:
            raise ValueError(f"No Python functions found in {repo_path}")

        repository_name = functions[0].repository_name
        self.search_service.setup_index()
        removed = self.search_service.delete_repository(repository_name)
        if removed:
            self.logger.info(f"Removed {removed} previously indexed functions for {repository_name}")

        if self.analysis_approach == "call_graph":
            result = self._process_with_call_graph(functions)
        else:
            result = self._process_with_lookup_table(functions)

        self.search_service.refresh()
        self.search_service.repository_name = repository_name
        self.processed_repo_path = repo_path
        result["repository_name"] = repository_name
        result["skipped_files"] = len(self.repo_processor.skipped_files)
        return result

    def _process_with_call_graph(self, functions: List) -> Dict[str, Any]:
        self.logger.info("Processing with call graph approach")

        graph_stats = self.search_service.build_and_store_call_graph(functions)
        if 'error' in graph_stats:
            raise RuntimeError(graph_stats['error'])

        self.logger.info(f"Call graph processing complete: {graph_stats}")

        return {
            "total_functions": graph_stats.get('total_functions', len(functions)),
            "embedded_functions": graph_stats.get('stored_functions', 0),
            "approach": "call_graph",
            "graph_statistics": graph_stats,
            "summary": self._get_call_graph_summary(functions, graph_stats)
        }

    def _process_with_lookup_table(self, functions: List) -> Dict[str, Any]:
        self.logger.info("Processing with function lookup table approach")

        embedded_count = 0
        for index, function in enumerate(functions, 1):
            embedding = self.embedding_service.create_embedding(function, include_metadata=True)
            if embedding:
                function.embedding = embedding
                if self.search_service.store_function(function, embedding):
                    embedded_count += 1

            if index % 10 == 0 or index == len(functions):
                self.logger.info(f"Processed {index}/{len(functions)} functions ({embedded_count} stored)")

        self.lookup_table.build_lookup_table(functions)

        self.logger.info(f"Lookup table processing complete: {embedded_count}/{len(functions)} functions embedded")

        return {
            "total_functions": len(functions),
            "embedded_functions": embedded_count,
            "approach": "function_lookup_table",
            "summary": self.lookup_table.get_table_summary()
        }

    def analyze_user_issue(self, user_query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        if not user_query or not user_query.strip():
            return {"error": "Query cannot be empty"}

        user_query = user_query.strip()
        context = context or {}

        self.logger.info(f"Analyzing user issue with {self.analysis_approach}: {user_query[:50]}...")

        try:
            enhanced_query = self.query_enhancer.enhance_query(user_query, context)
        except Exception as e:
            self.logger.warning(f"Query enhancement failed: {e}")
            enhanced_query = user_query

        if self.analysis_approach == "call_graph":
            search_results = self._search_with_call_graph(enhanced_query, context)
        else:
            search_results = self._search_with_lookup_table(enhanced_query)

        if not search_results:
            return {
                "error": "No relevant functions found for the query. Make sure the repository has been processed with this approach.",
                "query": user_query,
                "enhanced_query": enhanced_query,
                "approach": self.analysis_approach
            }

        if self.analysis_approach == "function_lookup_table":
            search_results = self._enhance_with_nested_functions(search_results)

        # Counted now: the analysis loop appends any extra functions the LLM asks for
        functions_found = len(search_results)
        analysis_request = AnalysisRequest(
            query=user_query,
            enhanced_query=enhanced_query,
            functions=search_results,
            context=context,
            approach=self._approach_enum(),
            call_graph_context=self._get_call_graph_context(search_results) if self.analysis_approach == "call_graph" else None
        )

        analysis_result = self._perform_iterative_analysis(analysis_request)

        try:
            self.history_manager.add_analysis(
                query=user_query,
                enhanced_query=enhanced_query,
                analysis_result=analysis_result.analysis,
                confidence_score=analysis_result.confidence_score,
                functions_analyzed=analysis_result.functions_analyzed,
                embedding_service=self.embedding_service_type,
                llm_service=self.llm_service_type,
                context={**context, "approach": self.analysis_approach}
            )
        except Exception as e:
            # A history write failure should not throw away a finished analysis
            self.logger.error(f"Failed to save analysis to history: {e}")

        self.logger.info(f"Analysis complete with confidence: {analysis_result.confidence_score} using {self.analysis_approach}")

        return {
            "query": user_query,
            "enhanced_query": enhanced_query,
            "functions_found": functions_found,
            "analysis": analysis_result.analysis,
            "confidence_score": analysis_result.confidence_score,
            "functions_analyzed": analysis_result.functions_analyzed,
            "suggested_fixes": analysis_result.suggested_fixes,
            "search_results": analysis_request.functions,
            "approach": self.analysis_approach,
            "graph_context": analysis_result.graph_context
        }

    def _search_with_call_graph(self, query: str, context: Dict[str, Any]) -> List[SearchResult]:
        max_depth = context.get('call_graph_depth', MAX_CALL_GRAPH_DEPTH)
        return self.search_service.search_functions_with_context(query, limit=DEFAULT_FUNCTION_LIMIT, max_depth=max_depth)

    def _search_with_lookup_table(self, query: str) -> List[SearchResult]:
        return self.search_service.search_functions(query, limit=DEFAULT_FUNCTION_LIMIT)

    def _get_call_graph_context(self, search_results: List[SearchResult]) -> Dict[str, Any]:
        if not search_results or self.analysis_approach != "call_graph":
            return {}

        context = {
            "total_functions": len(search_results),
            "functions_with_dependencies": 0,
            "max_depth": 0,
            "total_dependencies": 0
        }

        for result in search_results:
            if result.nested_functions and 'call_graph_context' in result.nested_functions:
                graph_ctx = result.nested_functions['call_graph_context']
                if graph_ctx.get('dependency_count', 0):
                    context["functions_with_dependencies"] += 1
                context["max_depth"] = max(context["max_depth"], graph_ctx.get('graph_depth', 0))
                context["total_dependencies"] += graph_ctx.get('dependency_count', 0)

        return context

    def _enhance_with_nested_functions(self, search_results: List[SearchResult]) -> List[SearchResult]:
        self.logger.info("Enhancing results with nested functions (lookup table approach)")

        for result in search_results:
            nested_functions = {}
            for nested_id in result.function_metadata.nested_call_ids:
                nested_func = self.search_service.get_function_by_id(nested_id)
                if nested_func:
                    nested_functions[nested_id] = {
                        "function_name": nested_func.name,
                        "file_path": nested_func.file_path,
                        "code": nested_func.code,
                        "start_line": nested_func.start_line,
                        "end_line": nested_func.end_line
                    }
            result.nested_functions = nested_functions
        return search_results

    def _perform_iterative_analysis(self, request: AnalysisRequest) -> AnalysisResult:
        self.logger.info(f"Starting iterative analysis with {self.analysis_approach} approach")

        analysis_result = self.analysis_service.analyze_issue(request)
        self.logger.info(f"Initial confidence: {analysis_result.confidence_score}")

        current_iteration = 0
        while (analysis_result.confidence_score < CONFIDENCE_THRESHOLD and
               analysis_result.additional_context_needed and
               current_iteration < MAX_ANALYSIS_ITERATIONS):

            self.logger.info(f"Requesting additional context (iteration {current_iteration + 1})")

            known_ids = {r.function_metadata.lookup_id for r in request.functions}
            additional_functions = []
            for function_id in analysis_result.additional_context_needed:
                if function_id in known_ids:
                    continue
                func = self.search_service.get_function_by_id(function_id)
                if not func:
                    continue
                search_result = SearchResult(
                    function_metadata=func,
                    relevance_score=1.0,
                    search_method="context_request",
                    match_type="additional_context",
                    approach_used=self._approach_enum()
                )

                if self.analysis_approach == "call_graph":
                    graph_result = self.search_service.get_function_with_graph_context(function_id)
                    if graph_result:
                        search_result.nested_functions = CallGraphSearchService.build_graph_context(graph_result)

                additional_functions.append(search_result)
                known_ids.add(function_id)

            if not additional_functions:
                self.logger.info("No new functions found for the requested context, stopping iterations")
                break

            request.functions.extend(additional_functions)
            analysis_result = self.analysis_service.analyze_issue(request)
            self.logger.info(f"Updated confidence after iteration {current_iteration + 1}: {analysis_result.confidence_score}")
            current_iteration += 1

        return analysis_result

    def _get_call_graph_summary(self, functions: List, graph_stats: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'total_functions': len(functions),
            'async_functions': sum(1 for f in functions if f.is_async),
            'functions_with_error_handling': sum(1 for f in functions if f.error_handling.get('has_try_catch')),
            'unique_modules': len({f.module_name for f in functions}),
            'total_nodes': graph_stats.get('total_nodes', 0),
            'root_nodes': graph_stats.get('root_nodes', 0),
            'leaf_nodes': graph_stats.get('leaf_nodes', 0),
            'max_depth': graph_stats.get('max_depth', 0),
            'total_edges': graph_stats.get('total_edges', 0),
            'stored_functions': graph_stats.get('stored_functions', 0),
            'approach': 'call_graph'
        }

    def switch_analysis_approach(self, new_approach: str) -> bool:
        """Switch approach. Returns True when the repository must be processed again
        (each approach keeps its own index)."""
        if not self.service_factory.validate_approach(new_approach):
            raise ValueError(f"Unsupported analysis approach: {new_approach}")

        if new_approach == self.analysis_approach:
            self.logger.info(f"Already using {new_approach} approach")
            return False

        self.logger.info(f"Switching from {self.analysis_approach} to {new_approach}")
        self.analysis_approach = new_approach
        self._load_services(new_approach)
        needs_reprocessing = self.processed_repo_path is not None
        self.processed_repo_path = None
        return needs_reprocessing

    def get_analysis_history_stats(self) -> Dict[str, Any]:
        stats = self.history_manager.get_statistics()
        stats['current_approach'] = self.analysis_approach
        return stats

    def get_recent_analyses(self, limit: int = 10) -> List[Dict[str, Any]]:
        return self.history_manager.get_recent_analyses(limit)

    def export_function_lookup_table(self, filepath: str, format: str = "json") -> str:
        if self.analysis_approach == "call_graph":
            raise ValueError("Function lookup table export is only available for the function_lookup_table approach")
        if self.lookup_table is None or self.lookup_table.lookup_df is None:
            raise ValueError("No lookup table yet; process a repository first")

        if format.lower() == "json":
            self.lookup_table.export_to_json(filepath)
        elif format.lower() == "csv":
            self.lookup_table.export_to_csv(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'csv'")

        self.logger.info(f"Function lookup table exported to {filepath}")
        return str(Path(filepath).resolve())

    def validate_services(self) -> Dict[str, bool]:
        validation_results = {
            "embedding_service": False,
            "search_service": False,
            "query_enhancer": False,
            "analysis_service": self.analysis_service is not None
        }

        try:
            test_embedding = self.embedding_service.create_query_embedding("test")
            validation_results["embedding_service"] = bool(test_embedding)
        except Exception as e:
            self.logger.warning(f"Embedding service validation failed: {e}")

        try:
            validation_results["search_service"] = self.search_service.is_healthy()
        except Exception as e:
            self.logger.warning(f"Search service validation failed: {e}")

        try:
            enhanced = self.query_enhancer.enhance_query("test")
            validation_results["query_enhancer"] = enhanced is not None
        except Exception as e:
            self.logger.warning(f"Query enhancer validation failed: {e}")

        return validation_results

    def get_supported_approaches(self):
        return self.service_factory.get_supported_approaches()
