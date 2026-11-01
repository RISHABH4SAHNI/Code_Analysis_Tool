from collections import defaultdict
from typing import Dict, List
from models.function_model import FunctionMetadata

BUILTIN_FUNCTIONS = {
    'abs', 'all', 'any', 'ascii', 'bin', 'bool', 'breakpoint', 'bytearray', 'bytes',
    'callable', 'chr', 'classmethod', 'compile', 'complex', 'delattr', 'dict', 'dir',
    'divmod', 'enumerate', 'eval', 'exec', 'filter', 'float', 'format', 'frozenset',
    'getattr', 'globals', 'hasattr', 'hash', 'help', 'hex', 'id', 'input', 'int',
    'isinstance', 'issubclass', 'iter', 'len', 'list', 'locals', 'map', 'max',
    'memoryview', 'min', 'next', 'object', 'oct', 'open', 'ord', 'pow', 'print',
    'property', 'range', 'repr', 'reversed', 'round', 'set', 'setattr', 'slice',
    'sorted', 'staticmethod', 'str', 'sum', 'super', 'tuple', 'type', 'vars', 'zip'
}

class CallResolver:
    """Maps the calls made inside a function to the functions they most likely target.

    Python is dynamic, so this is a best-effort static resolution:
    - `self.foo()` / `cls.foo()` prefers `foo` defined in the caller's own class
    - `foo()` prefers a module-level `foo` in the same module, then one whose
      module the caller imports, then any module-level `foo` in the repository
    - `obj.foo()` on any other receiver links every function named `foo`
    """

    def __init__(self, functions: List[FunctionMetadata]):
        self.by_name: Dict[str, List[FunctionMetadata]] = defaultdict(list)
        for function in functions:
            self.by_name[function.name].append(function)

    def resolve(self, function: FunctionMetadata) -> List[str]:
        targets: List[str] = []
        for call in function.qualified_calls or function.calls:
            receiver, _, name = call.rpartition('.')
            if name in BUILTIN_FUNCTIONS and not receiver:
                continue
            candidates = [c for c in self.by_name.get(name, []) if c.lookup_id != function.lookup_id]
            for target in self._pick(function, receiver, name, candidates):
                if target.lookup_id not in targets:
                    targets.append(target.lookup_id)
        return targets

    def _pick(self, caller: FunctionMetadata, receiver: str, name: str,
              candidates: List[FunctionMetadata]) -> List[FunctionMetadata]:
        if not candidates:
            return []

        if receiver in ('self', 'cls') and caller.class_context:
            same_class = [c for c in candidates
                          if c.class_context == caller.class_context and c.module_name == caller.module_name]
            return same_class or [c for c in candidates if c.class_context]

        if not receiver:
            module_level = [c for c in candidates if c.class_context is None]
            same_module = [c for c in module_level if c.module_name == caller.module_name]
            if same_module:
                return same_module
            imported = [c for c in module_level
                        if any(imp == f"{c.module_name}.{name}" or imp.endswith(f".{c.module_name}.{name}")
                               or imp.endswith(f"{c.module_name.split('.')[-1]}.{name}")
                               for imp in caller.imports)]
            if imported:
                return imported
            # A bare name may also be a class instantiation or nested function; fall back to all
            return module_level or candidates

        return candidates
