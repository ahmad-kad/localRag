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