import os
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from config.settings import EXCLUDED_DIRS, TEST_DIRS

def validate_repository_path(repo_path: str) -> bool:
    """Validate if the given path is a valid repository"""
    if not repo_path or not os.path.exists(repo_path):
        return False
    
    return Path(repo_path).is_dir() and bool(get_python_files(repo_path, include_tests=True))

def get_python_files(repo_path: str, include_tests: bool = False) -> List[str]:
    """Get all Python files in a repository, skipping virtualenvs, caches and (by default) tests"""
    python_files = []
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = sorted(
            d for d in dirs
            if d not in EXCLUDED_DIRS and not d.endswith('.egg-info')
            and (include_tests or d not in TEST_DIRS)
        )
        for file in sorted(files):
            if not file.endswith('.py'):
                continue
            file_path = os.path.join(root, file)
            # Relative path, so a repo that itself lives under a "test" folder is not skipped
            if include_tests or not is_test_file(os.path.relpath(file_path, repo_path)):
                python_files.append(file_path)
    
    return python_files

def extract_repo_name(repo_path: str) -> str:
    """Extract repository name from path"""
    return Path(repo_path).name

def calculate_confidence_score(analysis_text: str) -> float:
    """Calculate confidence score based on analysis text patterns"""
    confidence_indicators = {
        'high': ['definitely', 'clearly', 'obvious', 'certain', 'exactly', 'precisely'],
        'medium': ['likely', 'probably', 'appears', 'seems', 'suggests', 'indicates'],
        'low': ['might', 'could', 'possibly', 'maybe', 'uncertain', 'unclear']
    }
    
    text_lower = analysis_text.lower()
    
    high_count = sum(1 for word in confidence_indicators['high'] if word in text_lower)
    medium_count = sum(1 for word in confidence_indicators['medium'] if word in text_lower)
    low_count = sum(1 for word in confidence_indicators['low'] if word in text_lower)
    
    total_indicators = high_count + medium_count + low_count
    
    if total_indicators == 0:
        return 0.5
    
    confidence = (high_count * 1.0 + medium_count * 0.6 + low_count * 0.2) / total_indicators
    return min(max(confidence, 0.0), 1.0)

def format_code_snippet(code: str, max_lines: int = 20) -> str:
    """Format code snippet for display"""
    lines = code.split('\n')
    if len(lines) <= max_lines:
        return code
    
    return '\n'.join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"

def extract_function_signature(code: str) -> str:
    """Extract function signature from code"""
    lines = code.split('\n')
    for line in lines:
        if line.strip().startswith(('def ', 'async def ')):
            return line.strip()
    return ""

def clean_error_message(error_msg: str) -> str:
    """Clean and format error messages"""
    return re.sub(r'\s+', ' ', error_msg.strip())

def get_file_extension(file_path: str) -> str:
    """Get file extension"""
    return Path(file_path).suffix

def is_test_file(file_path: str) -> bool:
    """Check if file is a test file"""
    path = Path(file_path)
    file_name = path.name.lower()
    return (file_name.startswith('test_') or
            file_name.endswith('_test.py') or
            file_name == 'conftest.py' or
            any(part in TEST_DIRS for part in path.parts[:-1]))
