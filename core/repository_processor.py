import os
import ast
import uuid
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from models.function_model import FunctionMetadata
from core.call_resolver import CallResolver
from utils.helpers import get_python_files

class RepositoryProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.skipped_files: List[Dict[str, str]] = []

    def process_repository(self, repo_path: str, include_tests: bool = False,
                           max_functions: Optional[int] = None) -> List[FunctionMetadata]:
        functions = []
        self.skipped_files = []
        python_files = get_python_files(repo_path, include_tests=include_tests)

        for file_path in python_files:
            functions.extend(self._process_file(file_path, repo_path))
            if max_functions and len(functions) >= max_functions:
                self.logger.warning(f"Reached max_functions={max_functions}; remaining files skipped")
                functions = functions[:max_functions]
                break

        if self.skipped_files:
            self.logger.warning(f"Skipped {len(self.skipped_files)} file(s) that could not be parsed")

        self._build_nested_call_relationships(functions)
        return functions

    def _process_file(self, file_path: str, repo_path: str) -> List[FunctionMetadata]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            tree = ast.parse(source_code, filename=file_path)
        except (SyntaxError, UnicodeDecodeError, OSError, ValueError) as e:
            self.logger.warning(f"Skipping {file_path}: {e}")
            self.skipped_files.append({"file": file_path, "error": str(e)})
            return []

        extractor = FunctionExtractor(file_path, repo_path, source_code)
        return extractor.extract_functions(tree)

    def _build_nested_call_relationships(self, functions: List[FunctionMetadata]):
        resolver = CallResolver(functions)
        for func in functions:
            func.nested_call_ids = resolver.resolve(func)

class FunctionExtractor(ast.NodeVisitor):
    def __init__(self, file_path: str, repo_path: str, source_code: str):
        self.file_path = file_path
        self.repo_path = repo_path
        self.source_code = source_code
        self.repo_name = Path(repo_path).resolve().name
        relative_path = os.path.relpath(file_path, repo_path)
        self.relative_path = relative_path.replace(os.sep, '/')
        self.module_name = self.relative_path.removesuffix('.py').replace('/', '.')
        self.imports: List[str] = []
        self.class_stack: List[str] = []
        self.function_stack: List[str] = []
        self.functions: List[FunctionMetadata] = []

    def extract_functions(self, tree: ast.AST) -> List[FunctionMetadata]:
        self.imports = self._collect_imports(tree)
        self.visit(tree)
        return self.functions

    def _collect_imports(self, tree: ast.AST) -> List[str]:
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}" if module else alias.name)
        return imports

    def visit_ClassDef(self, node: ast.ClassDef):
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._visit_function(node)

    def _visit_function(self, node):
        self.functions.append(self._extract_function_metadata(node))
        # Methods of a class nested inside a function still belong to that class,
        # but a function nested in a method is not itself a method
        saved_classes = self.class_stack
        self.class_stack = []
        self.function_stack.append(node.name)
        self.generic_visit(node)
        self.function_stack.pop()
        self.class_stack = saved_classes

    def _make_lookup_id(self, node) -> str:
        # Deterministic so re-processing a repository overwrites documents instead of duplicating them
        qualified_name = '.'.join(self.class_stack + self.function_stack + [node.name])
        key = f"{self.repo_name}/{self.relative_path}::{qualified_name}:{node.lineno}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, key))

    def _extract_function_metadata(self, node) -> FunctionMetadata:
        qualified_calls = self._extract_calls(node)
        start_line = node.lineno
        end_line = node.end_lineno or node.lineno
        code = ast.get_source_segment(self.source_code, node) or ""
        if node.decorator_list:
            # Include decorators so line numbers in the code match start_line
            start_line = min(d.lineno for d in node.decorator_list)
            code = '\n'.join(self.source_code.split('\n')[start_line - 1:end_line])

        return FunctionMetadata(
            name=node.name,
            lookup_id=self._make_lookup_id(node),
            file_path=self.file_path,
            repository_name=self.repo_name,
            module_name=self.module_name,
            nested_call_ids=[],
            start_line=start_line,
            end_line=end_line,
            code=code,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            class_context=self.class_stack[-1] if self.class_stack else None,
            calls=sorted({call.rpartition('.')[2] for call in qualified_calls}),
            qualified_calls=qualified_calls,
            imports=list(self.imports),
            decorators=[self._get_decorator_name(d) for d in node.decorator_list],
            error_handling=self._extract_error_handling(node),
            file_name=os.path.basename(self.file_path)
        )

    def _extract_calls(self, node) -> List[str]:
        calls = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    call = child.func.id
                elif isinstance(child.func, ast.Attribute):
                    receiver = child.func.value
                    if isinstance(receiver, ast.Name):
                        call = f"{receiver.id}.{child.func.attr}"
                    else:
                        call = f"<expr>.{child.func.attr}"
                else:
                    continue
                if call not in calls:
                    calls.append(call)
        return calls

    def _get_decorator_name(self, decorator: ast.expr) -> str:
        if isinstance(decorator, ast.Call):
            decorator = decorator.func
        if isinstance(decorator, ast.Name):
            return decorator.id
        if isinstance(decorator, ast.Attribute):
            return decorator.attr
        return "unknown"

    def _extract_error_handling(self, node) -> Dict[str, Any]:
        error_info = {'has_try_catch': False, 'error_types': [], 'try_catch_blocks': []}
        try_types = (ast.Try, ast.TryStar) if hasattr(ast, 'TryStar') else (ast.Try,)

        for child in ast.walk(node):
            if isinstance(child, try_types):
                error_info['has_try_catch'] = True
                try_catch_code = ast.get_source_segment(self.source_code, child)
                if try_catch_code:
                    error_info['try_catch_blocks'].append(try_catch_code.strip())

                for handler in child.handlers:
                    error_info['error_types'].extend(self._handler_type_names(handler.type))

        return error_info

    def _handler_type_names(self, handler_type: Optional[ast.expr]) -> List[str]:
        if handler_type is None:
            return ['BaseException']
        if isinstance(handler_type, ast.Tuple):
            return [name for elt in handler_type.elts for name in self._handler_type_names(elt)]
        if isinstance(handler_type, ast.Name):
            return [handler_type.id]
        if isinstance(handler_type, ast.Attribute):
            return [handler_type.attr]
        return []
