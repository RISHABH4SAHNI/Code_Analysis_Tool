from typing import List, Dict, Set, Optional
from models.function_model import FunctionMetadata
from models.call_graph_model import CallGraph, CallGraphNode, CallGraphSearchResult
from core.call_resolver import CallResolver

class CallGraphProcessor:
    def __init__(self):
        self.call_graph: Optional[CallGraph] = None
        self.function_name_to_ids: Dict[str, Set[str]] = {}

    def build_call_graph(self, functions: List[FunctionMetadata]) -> CallGraph:
        self.call_graph = CallGraph()
        self.function_name_to_ids = {}

        for function in functions:
            self.call_graph.add_node(function)
            self.function_name_to_ids.setdefault(function.name, set()).add(function.lookup_id)

        self._build_edges(functions)
        self._identify_root_and_leaf_nodes()
        self.call_graph.calculate_depths()

        return self.call_graph

    def _build_edges(self, functions: List[FunctionMetadata]):
        resolver = CallResolver(functions)
        for function in functions:
            for callee_id in resolver.resolve(function):
                self.call_graph.add_edge(function.lookup_id, callee_id)

    def _identify_root_and_leaf_nodes(self):
        for node_id, node in self.call_graph.nodes.items():
            if not node.dependents:
                self.call_graph.root_nodes.add(node_id)
            if not node.dependencies:
                self.call_graph.leaf_nodes.add(node_id)

    def get_function_context_with_dependencies(self,
                                             target_function_id: str,
                                             max_depth: int = 3) -> Optional[CallGraphSearchResult]:
        if not self.call_graph or target_function_id not in self.call_graph.nodes:
            return None

        primary_node = self.call_graph.nodes[target_function_id]
        distances = self.call_graph.get_dependencies(target_function_id, max_depth)
        dependency_context = [self.call_graph.nodes[dep_id] for dep_id in distances]
        call_paths = self.get_call_paths_to_function(target_function_id, max_depth)

        return CallGraphSearchResult(
            primary_node=primary_node,
            dependency_context=dependency_context,
            relevance_score=1.0,
            search_method="call_graph_bfs",
            graph_depth=max(distances.values(), default=0),
            total_context_functions=len(dependency_context) + 1,
            call_paths=call_paths
        )

    def find_functions_by_name(self, function_name: str) -> List[CallGraphNode]:
        if not self.call_graph:
            return []

        return [self.call_graph.nodes[lookup_id]
                for lookup_id in sorted(self.function_name_to_ids.get(function_name, set()))
                if lookup_id in self.call_graph.nodes]

    def get_call_paths_to_function(self, target_function_id: str, max_depth: int = 3,
                                   max_paths: int = 20) -> List[List[str]]:
        """Call chains that start at a root (uncalled) function and end at the target,
        with at most max_depth calls. Searched backwards from the target over callers."""
        if not self.call_graph or target_function_id not in self.call_graph.nodes:
            return []

        paths: List[List[str]] = []

        def walk_callers(current_id: str, reversed_path: List[str]):
            if len(paths) >= max_paths:
                return
            if current_id in self.call_graph.root_nodes:
                paths.append(list(reversed(reversed_path)))
                return
            if len(reversed_path) > max_depth:
                return
            for caller_id in sorted(self.call_graph.nodes[current_id].dependents):
                if caller_id not in reversed_path:
                    reversed_path.append(caller_id)
                    walk_callers(caller_id, reversed_path)
                    reversed_path.pop()

        walk_callers(target_function_id, [target_function_id])
        return paths

    def get_graph_statistics(self) -> Dict[str, float]:
        if not self.call_graph:
            return {}

        total_nodes = len(self.call_graph.nodes)
        total_edges = len(self.call_graph.edges)
        return {
            'total_nodes': total_nodes,
            'root_nodes': len(self.call_graph.root_nodes),
            'leaf_nodes': len(self.call_graph.leaf_nodes),
            'max_depth': self.call_graph.max_depth,
            'total_edges': total_edges,
            'avg_calls_per_function': round(total_edges / total_nodes, 2) if total_nodes else 0.0
        }
