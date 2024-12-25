mport pytest
from pathlib import Path
import tempfile
import shutil
from typing import Generator

from src.code_rag.indexer import CodeIndexer
from src.code_rag.retriever import CodeRetriever
from src.code_rag.config import settings

@pytest.fixture(scope="session")
def sample_codebase(tmp_path_factory) -> Path:
    """Create a temporary codebase for testing."""
    code_dir = tmp_path_factory.mktemp("sample_code")
    
    # Create sample Python files
    main_py = code_dir / "main.py"
    main_py.write_text("""
def add(a: int, b: int) -> int:
    \"\"\"Add two numbers.\"\"\"
    return a + b

def subtract(a: int, b: int) -> int:
    \"\"\"Subtract two numbers.\"\"\"
    return a - b

if __name__ == "__main__":
    print(add(1, 2))
    """)
    
    utils_py = code_dir / "utils.py"
    utils_py.write_text("""
def validate_input(value: int) -> bool:
    \"\"\"Validate numeric input.\"\"\"
    return isinstance(value, int)

def process_data(data: list) -> list:
    \"\"\"Process a list of numbers.\"\"\"
    return [x * 2 for x in data if validate_input(x)]
    """)
    
    return code_dir

@pytest.fixture
def indexer(sample_codebase) -> CodeIndexer:
    """Create an indexer instance with sample codebase."""
    indexer = CodeIndexer()
    docs = indexer.process_directory(sample_codebase)
    indexer.create_index(docs)
    return indexer

@pytest.fixture
def retriever(indexer) -> CodeRetriever:
    """Create a retriever instance with indexed sample codebase."""
    return CodeRetriever(indexer.index)