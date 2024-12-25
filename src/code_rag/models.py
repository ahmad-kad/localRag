from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from datetime import datetime

class CodeChunk(BaseModel):
    """Represents a chunk of code with metadata."""
    content: str
    file_path: str
    language: Optional[str]
    start_line: int
    end_line: int
    
class Document(BaseModel):
    """Represents a processed document with metadata."""
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    chunks: Optional[List[CodeChunk]] = None

class SearchResult(BaseModel):
    """Represents a search result with relevancy score."""
    document: Document
    score: float
    
class QueryResult(BaseModel):
    """Represents the result of a query."""
    query: str
    response: str
    sources: List[SearchResult]
    confidence: float
    timestamp: datetime = datetime.now()