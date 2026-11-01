from typing import List, Dict, Optional
import pandas as pd
import json
from datetime import datetime
from models.function_model import FunctionMetadata

LOOKUP_COLUMNS = [
    'function_name', 'lookup_id', 'nested_call_ids', 'file_name', 'repository_name', 'module_name',
    'start_line', 'end_line', 'is_async', 'class_context', 'calls_count', 'has_error_handling'
]

class FunctionLookupTable:
    def __init__(self):
        self.functions: Dict[str, FunctionMetadata] = {}
        self.lookup_df: Optional[pd.DataFrame] = None

    def build_lookup_table(self, functions: List[FunctionMetadata]):
        self.functions = {func.lookup_id: func for func in functions}
        
        table_data = []
        for func in functions:
            table_data.append({
                'function_name': func.name,
                'lookup_id': func.lookup_id,
                'nested_call_ids': ','.join(func.nested_call_ids),
                'file_name': func.file_name,
                'repository_name': func.repository_name,
                'module_name': func.module_name,
                'start_line': func.start_line,
                'end_line': func.end_line,
                'is_async': func.is_async,
                'class_context': func.class_context or '',
                'calls_count': len(func.calls),
                'has_error_handling': func.error_handling.get('has_try_catch', False)
            })
        
        self.lookup_df = pd.DataFrame(table_data, columns=LOOKUP_COLUMNS)

    def get_function_by_id(self, lookup_id: str) -> Optional[FunctionMetadata]:
        return self.functions.get(lookup_id)

    def get_functions_by_name(self, function_name: str) -> List[FunctionMetadata]:
        return [func for func in self.functions.values() if func.name == function_name]

    def get_most_relevant_functions(self, query: str, limit: int = 5) -> List[FunctionMetadata]:
        if self.lookup_df is None:
            return []
        
        query_lower = query.lower()
        relevance_scores = []
        
        for _, row in self.lookup_df.iterrows():
            score = 0
            if query_lower in row['function_name'].lower():
                score += 10
            if query_lower in row['module_name'].lower():
                score += 5
            if row['has_error_handling'] and any(word in query_lower for word in ['error', 'exception', 'bug', 'issue']):
                score += 3
            
            relevance_scores.append((row['lookup_id'], score))
        
        relevance_scores.sort(key=lambda x: x[1], reverse=True)
        top_ids = [item[0] for item in relevance_scores[:limit] if item[1] > 0]
        
        return [self.functions[lookup_id] for lookup_id in top_ids]

    def get_nested_functions(self, lookup_id: str) -> List[FunctionMetadata]:
        func = self.get_function_by_id(lookup_id)
        if not func:
            return []
        
        nested_functions = []
        for nested_id in func.nested_call_ids:
            nested_func = self.get_function_by_id(nested_id)
            if nested_func:
                nested_functions.append(nested_func)
        
        return nested_functions

    def export_to_csv(self, filepath: str):
        if self.lookup_df is not None:
            self.lookup_df.to_csv(filepath, index=False)

    def export_to_json(self, filepath: str):
        if self.lookup_df is None:
            return
        
        export_data = {
            "export_info": {
                "created_at": datetime.now().isoformat(),
                "version": "1.0",
                "total_functions": len(self.lookup_df)
            },
            "summary": self.get_table_summary(),
            "functions": []
        }
        
        for lookup_id, func in self.functions.items():
            func_data = {
                "function_name": func.name,
                "lookup_id": func.lookup_id,
                "nested_call_ids": func.nested_call_ids,
                "file_path": func.file_path,
                "repository_name": func.repository_name,
                "module_name": func.module_name,
                "start_line": func.start_line,
                "end_line": func.end_line,
                "is_async": func.is_async,
                "class_context": func.class_context,
                "calls": func.calls,
                "decorators": func.decorators,
                "has_error_handling": func.error_handling.get('has_try_catch', False),
                "error_types": func.error_handling.get('error_types', [])
            }
            export_data["functions"].append(func_data)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

    def get_table_summary(self) -> Dict[str, int]:
        if self.lookup_df is None:
            return {}
        
        return {
            'total_functions': len(self.lookup_df),
            'async_functions': int(self.lookup_df['is_async'].sum()),
            'functions_with_error_handling': int(self.lookup_df['has_error_handling'].sum()),
            'unique_modules': int(self.lookup_df['module_name'].nunique()),
            'unique_repositories': int(self.lookup_df['repository_name'].nunique())
        }
