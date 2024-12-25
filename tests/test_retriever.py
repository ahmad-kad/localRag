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