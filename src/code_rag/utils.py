import logging
from pathlib import Path
from typing import List, Set
import mimetypes
from pygments.lexers import guess_lexer

logger = logging.getLogger(__name__)

def setup_logging(log_level: str = "INFO"):
    """Configure logging for the application."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def detect_file_type(file_path: Path) -> str:
    """Detect file type using mimetypes."""
    try:
        mime_type, _ = mimetypes.guess_type(str(file_path))
        return mime_type or "application/octet-stream"
    except Exception as e:
        logger.warning(f"Could not detect file type for {file_path}: {e}")
        return "application/octet-stream"

def detect_programming_language(content: str) -> str:
    """Detect programming language using Pygments."""
    try:
        lexer = guess_lexer(content)
        return lexer.name.lower()
    except Exception as e:
        logger.warning(f"Could not detect language: {e}")
        return "unknown"

def get_file_extensions(directory: Path) -> Set[str]:
    """Get all unique file extensions in directory."""
    extensions = set()
    for file_path in directory.rglob("*"):
        if file_path.is_file():
            extensions.add(file_path.suffix)
    return extensions