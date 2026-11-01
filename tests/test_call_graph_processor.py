from core.call_graph_processor import CallGraphProcessor
from models.function_model import FunctionMetadata
from models.call_graph_model import CallGraph, CallGraphSearchResult

class TestCallGraphProcessor:

    def setup_method(self):
        """Setup test fixtures"""
        self.processor = CallGraphProcessor()

        # main_function -> helper_function -> utility_function
        # main_function -> validator
        self.sample_functions = [
            FunctionMetadata(
                name="main_function",
                lookup_id="func-1",
                file_path="/src/main.py",
                repository_name="test-repo",
                module_name="src.main",
                start_line=10,
                end_line=20,
                code="def main_function(): pass",
                calls=["helper_function", "validator"]
            ),
            FunctionMetadata(
                name="helper_function",
                lookup_id="func-2",
                file_path="/src/helpers.py",
                repository_name="test-repo",
                module_name="src.helpers",
                start_line=5,
                end_line=15,
                code="def helper_function(): pass",
                calls=["utility_function"]
            ),
            FunctionMetadata(
                name="utility_function",
                lookup_id="func-3",
                file_path="/src/utils.py",
                repository_name="test-repo",
                module_name="src.utils",
                start_line=1,
                end_line=8,
                code="def utility_function(): pass",
                calls=[]
            ),
            FunctionMetadata(
                name="validator",
                lookup_id="func-4",
                file_path="/src/validators.py",
                repository_name="test-repo",
                module_name="src.validators",
                start_line=15,
                end_line=25,
                code="def validator(): pass",
                calls=[]
            )
        ]

    def _names(self, ids):
        return sorted(self.processor.call_graph.nodes[i].function_metadata.name for i in ids)

    def test_processor_initialization(self):
        """Test call graph processor initializes correctly"""
        assert self.processor.call_graph is None
        assert self.processor.function_name_to_ids == {}

    def test_build_call_graph(self):
        """Test call graph construction"""
        call_graph = self.processor.build_call_graph(self.sample_functions)

        assert isinstance(call_graph, CallGraph)
        assert len(call_graph.nodes) == 4
        assert len(call_graph.edges) == 3

        node_names = [node.function_metadata.name for node in call_graph.nodes.values()]
        for name in ["main_function", "helper_function", "utility_function", "validator"]:
            assert name in node_names

    def test_build_edges(self):
        """Test edge construction between functions"""
        self.processor.build_call_graph(self.sample_functions)

        main_node = self.processor.call_graph.nodes["func-1"]
        assert self._names(main_node.dependencies) == ["helper_function", "validator"]
        assert self._names(self.processor.call_graph.nodes["func-3"].dependents) == ["helper_function"]

    def test_identify_root_and_leaf_nodes(self):
        """Test identification of root and leaf nodes"""
        self.processor.build_call_graph(self.sample_functions)

        assert self._names(self.processor.call_graph.root_nodes) == ["main_function"]
        assert self._names(self.processor.call_graph.leaf_nodes) == ["utility_function", "validator"]

    def test_depths(self):
        """Depth is the shortest call distance from a root"""
        call_graph = self.processor.build_call_graph(self.sample_functions)

        assert call_graph.nodes["func-1"].depth_level == 0
        assert call_graph.nodes["func-2"].depth_level == 1
        assert call_graph.nodes["func-3"].depth_level == 2
        assert call_graph.max_depth == 2

    def test_get_function_context_with_dependencies(self):
        """Test getting function context with dependencies"""
        self.processor.build_call_graph(self.sample_functions)

        result = self.processor.get_function_context_with_dependencies("func-1", max_depth=3)

        assert isinstance(result, CallGraphSearchResult)
        assert result.primary_node.function_metadata.name == "main_function"
        dependency_names = sorted(n.function_metadata.name for n in result.dependency_context)
        assert dependency_names == ["helper_function", "utility_function", "validator"]
        assert result.graph_depth == 2
        assert result.total_context_functions == 4

        summary = result.get_context_summary()
        assert summary["dependency_count"] == 3

    def test_context_respects_max_depth(self):
        self.processor.build_call_graph(self.sample_functions)

        result = self.processor.get_function_context_with_dependencies("func-1", max_depth=1)

        assert sorted(n.function_metadata.name for n in result.dependency_context) == ["helper_function", "validator"]
        assert result.graph_depth == 1

    def test_context_for_unknown_function(self):
        self.processor.build_call_graph(self.sample_functions)
        assert self.processor.get_function_context_with_dependencies("missing") is None

    def test_find_functions_by_name(self):
        """Test finding functions by exact name"""
        self.processor.build_call_graph(self.sample_functions)

        results = self.processor.find_functions_by_name("helper_function")
        assert len(results) == 1
        assert results[0].function_metadata.name == "helper_function"

        assert self.processor.find_functions_by_name("nonexistent") == []

    def test_get_call_paths_to_function(self):
        """Test getting call paths to a function"""
        self.processor.build_call_graph(self.sample_functions)

        paths = self.processor.get_call_paths_to_function("func-3", max_depth=3)

        path_names = [[self.processor.call_graph.nodes[i].function_metadata.name for i in path] for path in paths]
        assert path_names == [["main_function", "helper_function", "utility_function"]]

    def test_call_paths_respect_max_depth(self):
        self.processor.build_call_graph(self.sample_functions)
        assert self.processor.get_call_paths_to_function("func-3", max_depth=1) == []

    def test_get_graph_statistics(self):
        """Test graph statistics calculation"""
        self.processor.build_call_graph(self.sample_functions)

        stats = self.processor.get_graph_statistics()

        assert stats["total_nodes"] == 4
        assert stats["total_edges"] == 3  # main->helper, main->validator, helper->utility
        assert stats["root_nodes"] == 1   # main_function
        assert stats["leaf_nodes"] == 2   # utility_function, validator
        assert stats["max_depth"] == 2
        assert stats["avg_calls_per_function"] == 0.75

    def test_empty_functions_list(self):
        """Test handling empty functions list"""
        call_graph = self.processor.build_call_graph([])

        assert isinstance(call_graph, CallGraph)
        assert len(call_graph.nodes) == 0
        assert len(call_graph.edges) == 0
        assert len(call_graph.root_nodes) == 0
        assert len(call_graph.leaf_nodes) == 0

    def test_single_function(self):
        """Test handling single function with no calls"""
        call_graph = self.processor.build_call_graph([self.sample_functions[2]])

        assert len(call_graph.nodes) == 1
        assert len(call_graph.edges) == 0
        assert len(call_graph.root_nodes) == 1
        assert len(call_graph.leaf_nodes) == 1
        assert list(call_graph.nodes.values())[0].function_metadata.name == "utility_function"

    def test_recursive_and_mutual_calls_do_not_hang(self):
        """Cycles used to recurse forever while calculating depths"""
        functions = [
            FunctionMetadata(name="entry", lookup_id="a", file_path="/m.py", repository_name="r",
                             module_name="m", calls=["ping"]),
            FunctionMetadata(name="ping", lookup_id="b", file_path="/m.py", repository_name="r",
                             module_name="m", calls=["pong", "ping"]),
            FunctionMetadata(name="pong", lookup_id="c", file_path="/m.py", repository_name="r",
                             module_name="m", calls=["ping"]),
        ]
        call_graph = self.processor.build_call_graph(functions)

        assert call_graph.nodes["c"].depth_level == 2
        result = self.processor.get_function_context_with_dependencies("a", max_depth=5)
        assert sorted(n.function_metadata.name for n in result.dependency_context) == ["ping", "pong"]
        paths = self.processor.get_call_paths_to_function("c", max_depth=5)
        assert paths == [["a", "b", "c"]]
