from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any, Tuple
from models.function_model import FunctionMetadata

@dataclass
class CallGraphNode:
    function_metadata: FunctionMetadata
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    depth_level: int = 0

@dataclass
class CallGraph:
    nodes: Dict[str, CallGraphNode] = field(default_factory=dict)
    root_nodes: Set[str] = field(default_factory=set)
    leaf_nodes: Set[str] = field(default_factory=set)
    max_depth: int = 0

    @property
    def edges(self) -> List[Tuple[str, str]]:
        """All (caller_id, callee_id) pairs"""
        return [(node_id, dep_id) for node_id, node in self.nodes.items() for dep_id in node.dependencies]

    def add_node(self, function: FunctionMetadata) -> CallGraphNode:
        node = CallGraphNode(function_metadata=function)
        self.nodes[function.lookup_id] = node
        return node

    def add_edge(self, caller_id: str, callee_id: str):
        if caller_id in self.nodes and callee_id in self.nodes:
            self.nodes[caller_id].dependencies.add(callee_id)
            self.nodes[callee_id].dependents.add(caller_id)

    def get_node(self, lookup_id: str) -> Optional[CallGraphNode]:
        return self.nodes.get(lookup_id)

    def get_dependencies(self, node_id: str, max_depth: int = 3) -> Dict[str, int]:
        """Breadth-first walk over callees; returns {lookup_id: distance} for every
        function reachable within max_depth calls, excluding the start node."""
        if node_id not in self.nodes:
            return {}

        distances = {node_id: 0}
        queue = deque([node_id])
        while queue:
            current_id = queue.popleft()
            depth = distances[current_id]
            if depth >= max_depth:
                continue
            for dep_id in sorted(self.nodes[current_id].dependencies):
                if dep_id not in distances:
                    distances[dep_id] = depth + 1
                    queue.append(dep_id)

        del distances[node_id]
        return distances

    def calculate_depths(self):
        """Set each node's depth to its shortest call distance from a root node.
        Breadth-first, so call cycles cannot cause infinite recursion."""
        self.max_depth = 0
        for node in self.nodes.values():
            node.depth_level = 0

        visited = set(self.root_nodes)
        queue = deque(self.root_nodes)
        while queue:
            current_id = queue.popleft()
            node = self.nodes[current_id]
            for dep_id in node.dependencies:
                if dep_id not in visited:
                    visited.add(dep_id)
                    self.nodes[dep_id].depth_level = node.depth_level + 1
                    self.max_depth = max(self.max_depth, node.depth_level + 1)
                    queue.append(dep_id)

@dataclass
class CallGraphSearchResult:
    primary_node: CallGraphNode
    dependency_context: List[CallGraphNode]
    relevance_score: float
    search_method: str
    graph_depth: int
    total_context_functions: int
    call_paths: List[List[str]] = field(default_factory=list)

    def get_all_functions(self) -> List[FunctionMetadata]:
        functions = [self.primary_node.function_metadata]
        for dep_node in self.dependency_context:
            if dep_node.function_metadata.lookup_id != self.primary_node.function_metadata.lookup_id:
                functions.append(dep_node.function_metadata)
        return functions

    def get_context_summary(self) -> Dict[str, Any]:
        all_functions = self.get_all_functions()
        return {
            'primary_function': self.primary_node.function_metadata.name,
            'dependency_count': len(self.dependency_context),
            'dependency_names': [node.function_metadata.name for node in self.dependency_context],
            'graph_depth': self.graph_depth,
            'total_functions': self.total_context_functions,
            'call_paths_count': len(self.call_paths),
            'relevance_score': self.relevance_score,
            'files_involved': sorted(set(func.file_name for func in all_functions)),
            'modules_involved': sorted(set(func.module_name for func in all_functions)),
            'has_error_handling': any(func.error_handling.get('has_try_catch', False) for func in all_functions),
            'async_functions': sum(1 for func in all_functions if func.is_async)
        }
