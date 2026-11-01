from services.analysis_service import AnalysisService
from models.function_model import AnalysisRequest, AnalysisApproach, FunctionMetadata, SearchResult

KNOWN_ID = "11111111-1111-1111-1111-111111111111"
WANTED_ID = "22222222-2222-2222-2222-222222222222"


class FakeLLM(AnalysisService):
    def __init__(self, reply):
        super().__init__()
        self.reply = reply

    def _complete(self, system_prompt, user_prompt):
        self.prompt = user_prompt
        return self.reply


def make_request(approach=AnalysisApproach.FUNCTION_LOOKUP_TABLE):
    func = FunctionMetadata(name="login", lookup_id=KNOWN_ID, file_path="/app/auth.py",
                            repository_name="app", module_name="auth", code="def login(): pass", start_line=40, end_line=40)
    result = SearchResult(function_metadata=func, relevance_score=0.9, search_method="vector", match_type="semantic",
                          nested_functions={WANTED_ID: {"function_name": "hash_password", "file_path": "/app/crypto.py",
                                                        "code": "", "start_line": 1, "end_line": 2}})
    return AnalysisRequest(query="q", enhanced_query="q", functions=[result], context={}, approach=approach)


def test_parses_multiline_markdown_sections():
    reply = """**CONFIDENCE_SCORE:** 85%
**ANALYSIS:**
`auth.py:42` calls `.strip()` on None.

**SUGGESTED_FIXES:**
1. Check for None before stripping
2. Add a test

ADDITIONAL_CONTEXT_NEEDED:
- 22222222-2222-2222-2222-222222222222 (hash_password)
- 11111111-1111-1111-1111-111111111111
"""
    result = FakeLLM(reply).analyze_issue(make_request())

    assert result.confidence_score == 0.85
    assert [f["description"] for f in result.suggested_fixes] == ["Check for None before stripping", "Add a test"]
    # IDs already in the request are not requested again
    assert result.additional_context_needed == [WANTED_ID]


def test_none_means_no_extra_context():
    result = FakeLLM("CONFIDENCE_SCORE: 0.9\nANALYSIS: fine\nADDITIONAL_CONTEXT_NEEDED: None").analyze_issue(make_request())
    assert result.confidence_score == 0.9
    assert result.additional_context_needed == []


def test_prompt_lists_nested_function_ids():
    llm = FakeLLM("CONFIDENCE_SCORE: 1")
    llm.analyze_issue(make_request())
    assert WANTED_ID in llm.prompt


def test_graph_insights_for_call_graph_approach():
    reply = "CONFIDENCE_SCORE: 0.7\nGRAPH_INSIGHTS: login calls hash_password\nANALYSIS: x"
    result = FakeLLM(reply).analyze_issue(make_request(AnalysisApproach.CALL_GRAPH))
    assert result.graph_context == {"insights": "login calls hash_password"}


def test_llm_error_returns_zero_confidence():
    class Failing(FakeLLM):
        def _complete(self, system_prompt, user_prompt):
            raise RuntimeError("rate limited")

    result = Failing("").analyze_issue(make_request())
    assert result.confidence_score == 0.0
    assert "rate limited" in result.analysis
