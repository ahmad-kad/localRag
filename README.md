# Code RAG System

A RAG (Retrieval Augmented Generation) system for analyzing and querying codebases using local LLMs.

## Features
- Local LLM support through Ollama
- Code-specific embeddings
- Interactive query interface
- Support for multiple programming languages
- Vector storage for efficient retrieval

## Prerequisites
- Python 3.9+
- Ollama installed (for local LLM support)
- Git

## Quick Start
1. Clone the repository:
```bash
git clone <repository-url>
cd code_rag_system
```

2. Set up the environment:
```bash
make setup
```

3. Configure your environment variables in `.env`

4. Run the development server:
```bash
make run
```

## Development
- Run tests: `make test`
- Format code: `make format`
- Check linting: `make lint`
- Build documentation: `make docs`

## License
MIT License

# 4. Project Metadata (pyproject.toml)
[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "code_rag"
version = "0.1.0"
description = "A RAG system for analyzing codebases"
readme = "README.md"
requires-python = ">=3.9"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "your.email@example.com"},
]

[tool.black]
line-length = 88
target-version = ['py39']
include = '\.pyi?$'

[tool.isort]
profile = "black"
multi_line_output = 3
include_trailing_comma = true
force_grid_wrap = 0
use_parentheses = true
line_length = 88

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-ra -q"
filterwarnings = ["ignore::DeprecationWarning"]