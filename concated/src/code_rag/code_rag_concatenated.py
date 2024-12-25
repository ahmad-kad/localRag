// Source: /Users/ahmadkaddoura/Desktop/code_rag_system/src/code_rag/config.py
// Relative path: src/code_rag/config.py
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Settings optimized for M3 Pro with RAG system"""
    # Model settings
    MODEL_NAME: str = "mistral-openorca"
    EMBED_MODEL: str = "BAAI/bge-small-en-v1.5"  # Optimized for code understanding
    TEMPERATURE: float = 0.1
    MAX_TOKENS: int = 4096
    
    # M3 Pro specific settings
    DEVICE: str = "mps"  # Metal Performance Shaders for Apple Silicon
    BATCH_SIZE: int = 8  
    
    # Vector store settings
    VECTOR_STORE_TYPE: str = "faiss"  # Using FAISS for vector storage
    VECTOR_DIM: int = 384  # BGE embedding dimension
    
    # Memory-optimized chunking
    CHUNK_SIZE: int = 2048
    CHUNK_OVERLAP: int = 256
    
    # Paths
    CACHE_DIR: Path = Path("data/cache")
    INDEX_DIR: Path = Path("data/indexes")
    FAISS_INDEX_PATH: Path = Path("data/indexes/faiss.index")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        extra='ignore'
    )

// Source: /Users/ahmadkaddoura/Desktop/code_rag_system/src/code_rag/models.py
// Relative path: src/code_rag/models.py
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

// Source: /Users/ahmadkaddoura/Desktop/code_rag_system/src/code_rag/retriever.py
// Relative path: src/code_rag/retriever.py
import logging
from typing import List
from datetime import datetime

# Import specific components instead of high-level ones
from llama_index import VectorStoreIndex
from llama_index.llms import Ollama
from llama_index.indices.query.schema import QueryBundle
from llama_index.indices.service_context import ServiceContext
from llama_index.response.schema import Response
from llama_index.schema import NodeWithScore

from .config import settings
from .models import QueryResult, SearchResult

logger = logging.getLogger(__name__)

class CodeRetriever:
    def __init__(self, index: VectorStoreIndex):
        self.index = index
        self.llm = Ollama(
            model=settings.MODEL_NAME,
            temperature=settings.TEMPERATURE,
            request_timeout=120.0
        )
        
        # Create service context for consistent configuration
        self.service_context = ServiceContext.from_defaults(
            llm=self.llm
        )
        
        # Initialize query engine with direct configuration
        self.query_engine = self.index.as_query_engine(
            service_context=self.service_context,
            similarity_top_k=5,
            streaming=True,
            response_mode="tree_summarize"  # Use string instead of class
        )

    def process_query(self, question: str) -> QueryResult:
        """Process a query using the initialized query engine."""
        logger.info(f"Processing query: {question}")
        
        try:
            # Create query bundle directly
            query_bundle = QueryBundle(
                query_str=question,
                custom_embedding_strs=None
            )
            
            # Get response using query engine
            response = self.query_engine.query(query_bundle)
            
            # Extract source nodes safely
            source_nodes = (
                response.source_nodes 
                if hasattr(response, 'source_nodes') 
                else []
            )
            
            # Create search results
            sources = [
                SearchResult(
                    document=node.document,
                    score=node.score if hasattr(node, 'score') else 0.0
                )
                for node in source_nodes
            ]
            
            return QueryResult(
                query=question,
                response=str(response),
                sources=sources,
                confidence=getattr(response, 'confidence', 0.0),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise

    def get_similar_code(self, code_snippet: str, n_results: int = 5) -> List[SearchResult]:
        """Find similar code snippets using vector similarity search."""
        try:
            # Use lower-level similarity search
            retriever = self.index.as_retriever(
                similarity_top_k=n_results
            )
            
            similar_nodes = retriever.retrieve(code_snippet)
            
            return [
                SearchResult(
                    document=node.document,
                    score=getattr(node, 'score', 0.0)
                )
                for node in similar_nodes
            ]
        except Exception as e:
            logger.error(f"Error finding similar code: {e}")
            raise

// Source: /Users/ahmadkaddoura/Desktop/code_rag_system/src/code_rag/__init__.py
// Relative path: src/code_rag/__init__.py


// Source: /Users/ahmadkaddoura/Desktop/code_rag_system/src/code_rag/embeddings.py
// Relative path: src/code_rag/embeddings.py
# src/code_rag/embeddings.py
import torch
from typing import List
from sentence_transformers import SentenceTransformer
from llama_index.core.embeddings import BaseEmbedding

class CodeEmbedding(BaseEmbedding):
    """Custom embedding class optimized for code using sentence-transformers."""
    
    def __init__(self, model_name: str, device: str = None):
        super().__init__()
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device == 'mps':
            # SentenceTransformer doesn't support MPS, fallback to CPU
            self.device = 'cpu'
        
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embed_dim = self.model.get_sentence_embedding_dimension()
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for a search query."""
        embeddings = self.model.encode(query, normalize_embeddings=True)
        return embeddings.tolist()
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """Generate embedding for a document text."""
        embeddings = self.model.encode(text, normalize_embeddings=True)
        return embeddings.tolist()
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts efficiently."""
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

// Source: /Users/ahmadkaddoura/Desktop/code_rag_system/src/code_rag/utils.py
// Relative path: src/code_rag/utils.py
import logging
from pathlib import Path
from typing import List, Set
import magic
from pygments.lexers import guess_lexer

logger = logging.getLogger(__name__)

def setup_logging(log_level: str = "INFO"):
    """Configure logging for the application."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def detect_file_type(file_path: Path) -> str:
    """Detect file type using libmagic."""
    try:
        mime = magic.Magic(mime=True)
        return mime.from_file(str(file_path))
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

// Source: /Users/ahmadkaddoura/Desktop/code_rag_system/src/code_rag/indexer.py
// Relative path: src/code_rag/indexer.py
from pathlib import Path
from typing import List
import logging
import torch
import faiss
import numpy as np

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.faiss import FaissVectorStore

from .embeddings import CodeEmbedding
from .config import settings

logger = logging.getLogger(__name__)

class CodeIndexer:
    def __init__(self):
        """Initialize the CodeIndexer with optimized settings for code processing."""
        self.index = None
        self.device = self._setup_device()
        logger.info(f"Using device: {self.device}")
        self._setup_components()
    
    def _setup_device(self) -> str:
        """Configure the processing device with appropriate fallback."""
        if torch.backends.mps.is_available():
            return "mps"
        logger.warning("MPS not available, falling back to CPU")
        return "cpu"
    
    def _setup_components(self):
        """Initialize core components with the new embedding model and vector store."""
        try:
            # Initialize our custom embedding model
            self.embed_model = CodeEmbedding(
                model_name=settings.EMBED_MODEL,
                device=self.device
            )
            logger.info("Successfully initialized embedding model")
            
            # Initialize FAISS index
            self.dimension = self.embed_model.embed_dim
            self.faiss_index = faiss.IndexFlatL2(self.dimension)
            
            # Create vector store
            self.vector_store = FaissVectorStore(
                faiss_index=self.faiss_index
            )
            
            # Initialize storage context
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            
            # Configure text splitting for code
            self.text_splitter = SentenceSplitter(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP
            )
            
            # Initialize the language model
            self.llm = Ollama(
                model=settings.MODEL_NAME,
                temperature=settings.TEMPERATURE
            )
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def process_directory(self, directory: Path) -> List:
        """Process a directory of code files into documents."""
        logger.info(f"Processing directory: {directory}")
        
        reader = SimpleDirectoryReader(
            input_dir=str(directory),
            exclude_hidden=True,
            recursive=True
        )
        
        documents = reader.load_data()
        logger.info(f"Processed {len(documents)} documents from directory")
        return documents
    
    def create_index(self, documents: List) -> VectorStoreIndex:
        """Create a vector index from processed documents."""
        logger.info("Creating vector index")
        
        service_context = ServiceContext.from_defaults(
            llm=self.llm,
            embed_model=self.embed_model,
            text_splitter=self.text_splitter,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        
        self.index = VectorStoreIndex.from_documents(
            documents,
            storage_context=self.storage_context,
            service_context=service_context,
        )
        
        logger.info("Vector index created successfully")
        return self.index

    def save_index(self, path: Path):
        """Save the current index and FAISS index to disk."""
        if self.index:
            # Save the llama index
            self.index.storage_context.persist(str(path))
            
            # Save the FAISS index
            faiss_path = path / "faiss.index"
            faiss.write_index(self.faiss_index, str(faiss_path))
            
            logger.info(f"Index saved to {path}")
        else:
            logger.warning("No index to save")
    
    def load_index(self, path: Path):
        """Load an index from disk."""
        if not path.exists():
            logger.error(f"Index path {path} does not exist")
            raise FileNotFoundError(f"No index found at {path}")
            
        # Load the FAISS index
        faiss_path = path / "faiss.index"
        if faiss_path.exists():
            self.faiss_index = faiss.read_index(str(faiss_path))
            self.vector_store = FaissVectorStore(faiss_index=self.faiss_index)
            
        # Load the llama index
        self.index = VectorStoreIndex.load_from_disk(
            str(path),
            storage_context=StorageContext.from_defaults(
                vector_store=self.vector_store
            )
        )
        logger.info(f"Index loaded from {path}")
