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
