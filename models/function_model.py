from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

class AnalysisApproach(Enum):
    FUNCTION_LOOKUP_TABLE = "function_lookup_table"
    CALL_GRAPH = "call_graph"

@dataclass
class FunctionMetadata:
    name: str
    lookup_id: str
    file_path: str
    repository_name: str
    module_name: str
    nested_call_ids: List[str] = field(default_factory=list)
    start_line: int = 0
    end_line: int = 0
    code: str = ""
    is_async: bool = False
    class_context: Optional[str] = None
    calls: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    error_handling: Dict[str, Any] = field(default_factory=dict)
    line_numbers: List[int] = field(default_factory=list)
    file_name: str = ""
    code_with_line_numbers: str = ""
    embedding: Optional[List[float]] = None
    # Calls with their receiver kept, e.g. "self.save", "helper", "client.get"
    qualified_calls: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.file_name:
            self.file_name = self.file_path.split('/')[-1] if self.file_path else ""
        if not self.line_numbers and self.end_line >= self.start_line > 0:
            self.line_numbers = list(range(self.start_line, self.end_line + 1))
        if not self.code_with_line_numbers and self.code:
            lines = self.code.split('\n')
            numbered_lines = []
            for i, line in enumerate(lines):
                numbered_lines.append(f"{self.start_line + i:4d}: {line}")
            self.code_with_line_numbers = '\n'.join(numbered_lines)

@dataclass
class SearchResult:
    function_metadata: FunctionMetadata
    relevance_score: float
    search_method: str
    match_type: str
    approach_used: AnalysisApproach = AnalysisApproach.FUNCTION_LOOKUP_TABLE
    nested_functions: Optional[Dict[str, Any]] = None

@dataclass
class AnalysisRequest:
    query: str
    enhanced_query: str
    functions: List[SearchResult]
    context: Dict[str, Any]
    approach: AnalysisApproach = AnalysisApproach.FUNCTION_LOOKUP_TABLE
    call_graph_context: Optional[Dict[str, Any]] = None

@dataclass
class AnalysisResult:
    analysis: str
    confidence_score: float
    functions_analyzed: int
    additional_context_needed: List[str]
    suggested_fixes: List[Dict[str, str]]
    approach_used: AnalysisApproach = AnalysisApproach.FUNCTION_LOOKUP_TABLE
    graph_context: Optional[Dict[str, Any]] = None
