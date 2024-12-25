// Source: /Users/ahmadkaddoura/Desktop/code_rag_system/tests/test_utils.py
// Relative path: tests/test_utils.py
from src.code_rag.utils import detect_programming_language, get_file_extensions

def test_language_detection():
    """Test programming language detection."""
    python_code = """
    def hello():
        print("Hello, World!")
    """
    language = detect_programming_language(python_code)
    assert language.lower() == "python"

def test_file_extensions(sample_codebase):
    """Test file extension detection."""
    extensions = get_file_extensions(sample_codebase)
    assert ".py" in extensions
Last edited just now

// Source: /Users/ahmadkaddoura/Desktop/code_rag_system/tests/conftest.py
// Relative path: tests/conftest.py
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

// Source: /Users/ahmadkaddoura/Desktop/code_rag_system/tests/__init__.py
// Relative path: tests/__init__.py


// Source: /Users/ahmadkaddoura/Desktop/code_rag_system/tests/test_web_interface.py
// Relative path: tests/test_web_interface.py
from src.code_rag.web.app import RAGInterface
import gradio as gr
import pytest

@pytest.fixture
def rag_interface(sample_codebase):
    """Create a RAG interface instance."""
    interface = RAGInterface()
    interface.initialize_system(str(sample_codebase), "mistral-openorca", False)
    return interface

def test_initialization(rag_interface, sample_codebase):
    """Test system initialization."""
    status = rag_interface.initialize_system(
        str(sample_codebase),
        "mistral-openorca",
        False
    )
    assert "✅" in status
    assert rag_interface.retriever is not None

def test_query_processing(rag_interface):
    """Test query processing."""
    history = []
    new_history, _ = rag_interface.process_query(
        "What functions are available?",
        history,
        True
    )
    assert len(new_history) == 1
    assert "add" in new_history[0][1].lower()
    assert "process_data" in new_history[0][1].lower()

def test_error_handling(rag_interface):
    """Test error handling in queries."""
    history = []
    new_history, _ = rag_interface.process_query(
        "",  # Empty query
        history,
        True
    )
    assert len(new_history) == 1
    assert "error" in new_history[0][1].lower()


// Source: /Users/ahmadkaddoura/Desktop/code_rag_system/tests/test_indexer.py
// Relative path: tests/test_indexer.py
def test_process_directory(sample_codebase, indexer):
    """Test processing a directory of code files."""
    docs = indexer.process_directory(sample_codebase)
    assert len(docs) > 0
    assert any("add" in doc.content for doc in docs)
    assert any("process_data" in doc.content for doc in docs)

def test_create_index(indexer):
    """Test index creation."""
    assert indexer.index is not None
    assert indexer.index.storage_context is not None

def test_save_load_index(indexer, tmp_path):
    """Test saving and loading an index."""
    save_path = tmp_path / "test_index"
    indexer.save_index(save_path)
    assert save_path.exists()
    
    new_indexer = CodeIndexer()
    new_indexer.load_index(save_path)
    assert new_indexer.index is not None

def test_process_directory(sample_codebase, indexer):
    """Test processing a directory of code files."""
    docs = indexer.process_directory(sample_codebase)
    assert len(docs) > 0
    assert any("add" in doc.content for doc in docs)
    assert any("process_data" in doc.content for doc in docs)

def test_create_index(indexer):
    """Test index creation."""
    assert indexer.index is not None
    assert indexer.index.storage_context is not None

def test_save_load_index(indexer, tmp_path):
    """Test saving and loading an index."""
    save_path = tmp_path / "test_index"
    indexer.save_index(save_path)
    assert save_path.exists()
    
    new_indexer = CodeIndexer()
    new_indexer.load_index(save_path)
    assert new_indexer.index is not None

// Source: /Users/ahmadkaddoura/Desktop/code_rag_system/tests/test_retriever.py
// Relative path: tests/test_retriever.py
def test_basic_query(retriever):
    """Test basic query functionality."""
    result = retriever.process_query("What does the add function do?")
    assert result.response is not None
    assert "add" in result.response.lower()
    assert result.sources is not None
    assert len(result.sources) > 0

def test_code_explanation(retriever):
    """Test code explanation queries."""
    result = retriever.process_query("Explain the process_data function")
    assert result.response is not None
    assert "process" in result.response.lower()
    assert "list" in result.response.lower()

def test_similar_code(retriever):
    """Test finding similar code snippets."""
    snippet = "def multiply(a: int, b: int) -> int:"
    results = retriever.get_similar_code(snippet)
    assert len(results) > 0
    assert all(result.score >= 0 for result in results)
