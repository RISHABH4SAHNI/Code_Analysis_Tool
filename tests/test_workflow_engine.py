import pytest
from unittest.mock import Mock, patch
from core.workflow_engine import WorkflowEngine
from models.function_model import FunctionMetadata, SearchResult, AnalysisResult
from utils.history_manager import HistoryManager


def make_function(name, lookup_id, nested_call_ids=None):
    return FunctionMetadata(name=name, lookup_id=lookup_id, file_path=f"/repo/{name}.py",
                            repository_name="repo", module_name=name, code=f"def {name}(): pass",
                            start_line=1, end_line=1, nested_call_ids=nested_call_ids or [])


def make_analysis(confidence, needed=None):
    return AnalysisResult(analysis="ok", confidence_score=confidence, functions_analyzed=1,
                          additional_context_needed=needed or [], suggested_fixes=[])


class TestWorkflowEngine:

    def setup_method(self):
        """Engine with every external service mocked"""
        self.services = {
            'repository_processor': Mock(skipped_files=[]),
            'embedding_service': Mock(),
            'search_service': Mock(),
            'query_enhancer': Mock(),
            'analysis_service': Mock(),
            'lookup_table': Mock(),
        }
        self.history = Mock(spec=HistoryManager)
        with patch('factories.service_factory.ServiceFactory.get_services_for_approach', return_value=self.services):
            self.engine = WorkflowEngine(
                embedding_service_type="openai",
                llm_service_type="openai",
                history_manager=self.history
            )

    def test_workflow_engine_initialization(self):
        """Test workflow engine initializes correctly"""
        assert self.engine.embedding_service_type == "openai"
        assert self.engine.llm_service_type == "openai"
        assert self.engine.analysis_approach == "function_lookup_table"
        assert self.engine.search_service is self.services['search_service']
        assert self.engine.history_manager is self.history

    def test_invalid_approach(self):
        with pytest.raises(ValueError, match="Unsupported analysis approach"):
            WorkflowEngine(analysis_approach="vector_search")

    @patch('os.path.exists')
    @patch('os.path.isdir')
    def test_process_repository_invalid_path(self, mock_isdir, mock_exists):
        """Test repository processing with invalid path"""
        mock_exists.return_value = False

        with pytest.raises(ValueError, match="Repository path does not exist"):
            self.engine.process_repository("/invalid/path")

    @patch('os.path.exists')
    @patch('os.path.isdir')
    def test_process_repository_not_directory(self, mock_isdir, mock_exists):
        """Test repository processing with file instead of directory"""
        mock_exists.return_value = True
        mock_isdir.return_value = False

        with pytest.raises(ValueError, match="Path is not a directory"):
            self.engine.process_repository("/path/to/file.py")

    def test_process_repository_replaces_previous_documents(self, tmp_path):
        functions = [make_function("a", "id-a"), make_function("b", "id-b")]
        self.services['repository_processor'].process_repository.return_value = functions
        self.services['embedding_service'].create_embedding.return_value = [0.1, 0.2]
        self.services['search_service'].store_function.return_value = True
        self.services['search_service'].delete_repository.return_value = 0

        result = self.engine.process_repository(str(tmp_path), include_tests=True)

        self.services['repository_processor'].process_repository.assert_called_once_with(
            str(tmp_path), include_tests=True, max_functions=None)
        self.services['search_service'].delete_repository.assert_called_once_with("repo")
        assert self.services['search_service'].repository_name == "repo"
        assert result["embedded_functions"] == 2

    def test_analyze_user_issue_empty_query(self):
        """Test analysis with empty query"""
        result = self.engine.analyze_user_issue("")
        assert result == {"error": "Query cannot be empty"}

    def test_analysis_loop_fetches_requested_context_once(self):
        main = make_function("main", "id-main", nested_call_ids=[])
        extra = make_function("extra", "id-extra")
        self.services['query_enhancer'].enhance_query.return_value = "enhanced"
        self.services['search_service'].search_functions.return_value = [
            SearchResult(function_metadata=main, relevance_score=0.9, search_method="vector", match_type="semantic")
        ]
        self.services['search_service'].get_function_by_id.side_effect = lambda i: extra if i == "id-extra" else None
        # Low confidence asking for the same function every time: it is added once, then the loop stops
        self.services['analysis_service'].analyze_issue.return_value = make_analysis(0.4, ["id-extra"])

        result = self.engine.analyze_user_issue("bug in main")

        assert self.services['analysis_service'].analyze_issue.call_count == 2
        assert result["functions_found"] == 1
        assert [r.function_metadata.name for r in result["search_results"]] == ["main", "extra"]
        self.history.add_analysis.assert_called_once()

    def test_history_failure_does_not_lose_analysis(self):
        main = make_function("main", "id-main")
        self.services['query_enhancer'].enhance_query.return_value = "q"
        self.services['search_service'].search_functions.return_value = [
            SearchResult(function_metadata=main, relevance_score=0.9, search_method="vector", match_type="semantic")
        ]
        self.services['analysis_service'].analyze_issue.return_value = make_analysis(0.9)
        self.history.add_analysis.side_effect = OSError("disk full")

        result = self.engine.analyze_user_issue("bug")

        assert result["analysis"] == "ok"

    def test_export_requires_processed_repository(self):
        self.services['lookup_table'].lookup_df = None
        with pytest.raises(ValueError, match="process a repository first"):
            self.engine.export_function_lookup_table("out.json")

    def test_validate_services(self):
        """Test service validation"""
        self.services['embedding_service'].create_query_embedding.return_value = [0.1]
        self.services['search_service'].is_healthy.return_value = False
        self.services['query_enhancer'].enhance_query.return_value = "test"

        validation = self.engine.validate_services()

        assert validation == {
            "embedding_service": True,
            "search_service": False,
            "query_enhancer": True,
            "analysis_service": True
        }
