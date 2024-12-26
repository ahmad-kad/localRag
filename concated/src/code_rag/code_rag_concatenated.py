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

    # Web interface settings
    API_PORT: int = 8000
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        extra='ignore'
    )

# Create and export a settings instance
settings = Settings()

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

from llama_index.core import VectorStoreIndex
from llama_index.llms.ollama import Ollama
from llama_index.core.indices.query.schema import QueryBundle
from llama_index.core.service_context import ServiceContext
from llama_index.core.schema import NodeWithScore
from llama_index.core.base.response.schema import Response

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
        
        # Create service context with updated settings
        self.service_context = ServiceContext.from_defaults(
            llm=self.llm,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        
        # Initialize query engine with updated settings
        self.query_engine = self.index.as_query_engine(
            service_context=self.service_context,
            similarity_top_k=5,
            streaming=True,
            response_mode="tree_summarize",
            verbose=True
        )

    def process_query(self, question: str) -> QueryResult:
        """Process a query using the initialized query engine."""
        logger.info(f"Processing query: {question}")
        
        try:
            # Create query bundle with metadata
            query_bundle = QueryBundle(
                query_str=question,
                custom_embedding_strs=None,
                metadata={"timestamp": datetime.now().isoformat()}
            )
            
            # Get response using query engine with source nodes
            response = self.query_engine.query(query_bundle)
            
            # Extract source nodes with proper error handling
            source_nodes = []
            if hasattr(response, 'source_nodes'):
                source_nodes = response.source_nodes
            elif hasattr(response, 'metadata') and 'source_nodes' in response.metadata:
                source_nodes = response.metadata['source_nodes']
            
            # Create search results from source nodes
            sources = [
                SearchResult(
                    document=node.node.document,
                    score=float(node.score) if hasattr(node, 'score') else 0.0
                )
                for node in source_nodes
            ]
            
            return QueryResult(
                query=question,
                response=str(response),
                sources=sources,
                confidence=float(getattr(response, 'confidence', 0.0)),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise

    def get_similar_code(self, code_snippet: str, n_results: int = 5) -> List[SearchResult]:
        """Find similar code snippets using vector similarity search."""
        try:
            # Use updated retriever with similarity configuration
            retriever = self.index.as_retriever(
                similarity_top_k=n_results,
                service_context=self.service_context,
                verbose=True
            )
            
            # Perform similarity search with updated response handling
            similar_nodes = retriever.retrieve(code_snippet)
            
            # Create search results from retrieved nodes
            return [
                SearchResult(
                    document=node.node.document,
                    score=float(getattr(node, 'score', 0.0))
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
"""Custom embedding implementations optimized for code understanding.

This module provides embedding models specifically tuned for processing and 
understanding code snippets, documentation, and technical content using 
sentence-transformers as the underlying implementation.
"""

import os
import logging
import torch
import numpy as np
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Dict, Any
from sentence_transformers import SentenceTransformer
from llama_index.core.embeddings import BaseEmbedding
from pydantic import Field, PrivateAttr
from pydantic.config import ConfigDict

# Configure logging
logger = logging.getLogger(__name__)
    


class CodeEmbedding(BaseEmbedding):
    """Custom embedding class optimized for code using sentence-transformers."""
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra='ignore'
    )
    
    # Define public fields
    model_name: str = Field(description="Name of the sentence-transformer model to use")
    device_str: str = Field(default="auto", description="Device to run the model on")
    embed_dim: int = Field(default=384, description="Dimension of the embeddings")
    
    # Private attributes
    _model: SentenceTransformer = PrivateAttr()
    _device: str = PrivateAttr()
    _is_initialized: bool = PrivateAttr(default=False)
    
    @staticmethod
    def configure_mps():
        """Configure MPS settings for Apple Silicon with safe defaults and validation.
        
        This method handles the configuration of Metal Performance Shaders (MPS)
        for PyTorch on Apple Silicon devices. It sets up appropriate memory management
        and fallback settings with validation to ensure values are within safe ranges.
        """
        try:
            # First, clear any existing MPS environment variables to prevent conflicts
            if 'PYTORCH_MPS_HIGH_WATERMARK_RATIO' in os.environ:
                del os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO']
                
            # Get watermark ratio from environment with strict parsing
            try:
                raw_ratio = os.getenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO")
                if raw_ratio is not None:
                    watermark_ratio = float(raw_ratio)
                else:
                    watermark_ratio = 0.7  # Default if not set
            except ValueError:
                logger.warning("Invalid watermark ratio in environment, using default 0.7")
                watermark_ratio = 0.7
            
            # Ensure the ratio is within valid bounds (0.0 to 1.0)
            watermark_ratio = max(0.1, min(0.95, watermark_ratio))
            
            # Set the validated ratio
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = str(watermark_ratio)
            logger.info(f"Set MPS high watermark ratio to: {watermark_ratio}")
            
            # Configure MPS fallback
            if os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
                torch.backends.mps.enable_fallback_for_not_implemented_ops = True
                logger.info("Enabled MPS fallback for unsupported operations")
                
            # Additional MPS configurations if needed
            torch.backends.mps.enable_fallback_for_not_implemented_ops = True
            
        except Exception as e:
            logger.warning(f"Error configuring MPS: {str(e)}. Falling back to CPU")
            return "cpu"

    @staticmethod
    def _determine_device(device: str) -> str:
        """Determine appropriate device with robust fallback handling.
        
        Args:
            device (str): Requested device or 'auto' for automatic selection
            
        Returns:
            str: Selected device string ('cuda', 'mps', or 'cpu')
        """
        try:
            if device == "auto":
                if torch.cuda.is_available():
                    return "cuda"
                elif torch.backends.mps.is_available():
                    try:
                        CodeEmbedding.configure_mps()
                        # Test MPS availability after configuration
                        test_tensor = torch.ones((1, 1), device='mps')
                        return "mps"
                    except Exception as e:
                        logger.warning(f"MPS configuration failed: {e}. Falling back to CPU")
                        return "cpu"
                return "cpu"
            
            valid_devices = {"cuda", "cpu", "mps"}
            if device not in valid_devices:
                raise ValueError(f"Invalid device '{device}'. Must be one of: {valid_devices} or 'auto'")
            
            if device == "mps":
                try:
                    CodeEmbedding.configure_mps()
                    # Verify MPS is working
                    test_tensor = torch.ones((1, 1), device='mps')
                    return "mps"
                except Exception as e:
                    logger.warning(f"MPS configuration failed: {e}. Falling back to CPU")
                    return "cpu"
            
            return device
            
        except Exception as e:
            logger.warning(f"Error in device selection: {e}. Falling back to CPU")
            return "cpu"
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "auto",
        cache_dir: Optional[Path] = None,
        **kwargs
    ) -> None:
        """Initialize the embedding model."""
        # Initialize base class
        super().__init__(
            model_name=model_name or os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
            **kwargs
        )
        
        try:
            # Set up device with environment variable fallback
            env_device = os.getenv("DEVICE", "auto")
            self._device = self._determine_device(device if device != "auto" else env_device)
            logger.info(f"Using device: {self._device}")
            
            # Initialize model with appropriate settings
            model_kwargs = {}
            if cache_dir is not None:
                model_kwargs['cache_folder'] = str(cache_dir)
            
            # Initialize the model with the configured device
            self._model = SentenceTransformer(self.model_name, device=self._device, **model_kwargs)
            self.embed_dim = self._model.get_sentence_embedding_dimension()
            
            # Mark as successfully initialized
            self._is_initialized = True
            logger.info(f"Successfully initialized {self.model_name} with dimension {self.embed_dim} on {self._device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise RuntimeError(f"Model initialization failed: {str(e)}")
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for a query string.
        
        The method applies query-specific preprocessing and normalization
        before generating the embedding.
        
        Args:
            query (str): The query text to embed
            
        Returns:
            List[float]: The normalized embedding vector
            
        Raises:
            RuntimeError: If embedding generation fails
        """
        if not self._is_initialized:
            raise RuntimeError("Model not properly initialized")
        
        try:
            embeddings = self._model.encode(
                query,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            raise RuntimeError(f"Failed to generate query embedding: {str(e)}") from e
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Asynchronously generate embedding for a query string.
        
        Currently falls back to synchronous implementation as sentence-transformers
        doesn't provide native async support.
        
        Args:
            query (str): The query text to embed
            
        Returns:
            List[float]: The normalized embedding vector
        """
        return self._get_query_embedding(query)
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """Generate embedding for a text string.
        
        Applies text-specific preprocessing and normalization before
        generating the embedding.
        
        Args:
            text (str): The text to embed
            
        Returns:
            List[float]: The normalized embedding vector
            
        Raises:
            RuntimeError: If embedding generation fails
        """
        if not self._is_initialized:
            raise RuntimeError("Model not properly initialized")
        
        try:
            embeddings = self._model.encode(
                text,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating text embedding: {str(e)}")
            raise RuntimeError(f"Failed to generate text embedding: {str(e)}") from e
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts in batch.
        
        Optimized for batch processing, using the model's native batch
        support for better performance.
        
        Args:
            texts (List[str]): List of texts to embed
            
        Returns:
            List[List[float]]: List of normalized embedding vectors
            
        Raises:
            RuntimeError: If batch embedding generation fails
        """
        if not self._is_initialized:
            raise RuntimeError("Model not properly initialized")
        
        try:
            embeddings = self._model.encode(
                texts,
                normalize_embeddings=True,
                batch_size=self.embed_batch_size,
                show_progress_bar=False
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            raise RuntimeError(f"Failed to generate batch embeddings: {str(e)}") from e
    
    def similarity(
        self,
        embedding1: List[float],
        embedding2: List[float],
        method: str = "cosine"
    ) -> float:
        """Calculate similarity between two embeddings.
        
        Supports multiple similarity metrics for different use cases.
        
        Args:
            embedding1 (List[float]): First embedding vector
            embedding2 (List[float]): Second embedding vector
            method (str): Similarity method ('cosine', 'dot', 'euclidean')
            
        Returns:
            float: Similarity score between the embeddings
            
        Raises:
            ValueError: If invalid similarity method specified
        """
        # Convert to numpy arrays for efficient computation
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        if method == "cosine":
            return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
        elif method == "dot":
            return float(np.dot(vec1, vec2))
        elif method == "euclidean":
            return float(-np.linalg.norm(vec1 - vec2))
        else:
            raise ValueError(
                f"Invalid similarity method '{method}'. "
                "Must be one of: 'cosine', 'dot', 'euclidean'"
            )

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model configuration.
        
        Returns:
            Dict[str, Any]: Dictionary containing model information
        """
        return {
            "model_name": self.model_name,
            "device": self._device,
            "embedding_dimension": self.embed_dim,
            "batch_size": self.embed_batch_size,
            "is_initialized": self._is_initialized
        }

// Source: /Users/ahmadkaddoura/Desktop/code_rag_system/src/code_rag/utils.py
// Relative path: src/code_rag/utils.py
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

// Source: /Users/ahmadkaddoura/Desktop/code_rag_system/src/code_rag/indexer.py
// Relative path: src/code_rag/indexer.py
# Update imports in src/code_rag/indexer.py
from pathlib import Path
from typing import List
import logging
import torch
import faiss
import numpy as np
import json

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.faiss.base import FaissVectorStore  # Changed this line
from llama_index.llms.ollama import Ollama

from .embeddings import CodeEmbedding
from .config import settings

logger = logging.getLogger(__name__)

class CodeIndexer:
    def __init__(self):
            """Initialize the CodeIndexer with optimized settings for code processing.
            
            This initializer sets up all necessary components and ensures the storage
            structure exists. If no existing index is found, it prepares the system
            for creating a new one.
            """
            # Initialize basic attributes
            self.index = None
            self.device = self._setup_device()
            logger.info(f"Using device: {self.device}")
            
            # Initialize storage directories
            self.index_path = Path(settings.INDEX_DIR)
            self.cache_path = Path(settings.CACHE_DIR)
            self._ensure_storage_structure()
            
            # Set up components
            self._setup_components()
    
    def _setup_device(self) -> str:
        """Configure the processing device with appropriate fallback."""
        if torch.backends.mps.is_available():
            return "mps"
        logger.warning("MPS not available, falling back to CPU")
        return "cpu"
    
    def _setup_components(self):
        """Initialize core components with the new embedding model and vector store.
        
        This method sets up all necessary components for the indexing system:
        - Embedding model for converting text to vectors
        - FAISS index for efficient similarity search
        - Vector store for managing embeddings
        - Text splitter for processing code into chunks
        - Language model for analysis
        """
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
            
            # Create vector store without explicit dimension
            self.vector_store = FaissVectorStore(
                faiss_index=self.faiss_index
            )
            
            # Initialize storage context with vector store
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store,
                persist_dir=str(settings.INDEX_DIR)
            )
            
            # Configure text splitting for code with updated settings
            self.text_splitter = SentenceSplitter(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
                separator="\n"  # Better for code splitting
            )
            
            # Initialize the language model with updated kwargs
            self.llm = Ollama(
                model=settings.MODEL_NAME,
                temperature=settings.TEMPERATURE,
                request_timeout=120.0,
                additional_kwargs={
                    "system": "You are an expert code analysis assistant. "
                            "Analyze code thoroughly and provide detailed, accurate explanations."
                }
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
            recursive=True,
            filename_as_id=True  # Enable filename as document ID
        )
        
        documents = reader.load_data()
        logger.info(f"Processed {len(documents)} documents from directory")
        return documents

    def _ensure_storage_structure(self):
        """Ensure all necessary storage directories and files exist.
        
        This method creates the required directory structure and initializes
        empty storage files if they don't exist. This prevents file not found
        errors later in the process.
        """
        try:
            # Create necessary directories
            self.index_path.mkdir(parents=True, exist_ok=True)
            self.cache_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize docstore.json if it doesn't exist
            docstore_path = self.index_path / "docstore.json"
            if not docstore_path.exists():
                # Create an empty docstore with the required structure
                empty_docstore = {
                    "docstore/data": {},
                    "docstore/metadata": {},
                    "version": "1.0.0"
                }
                with open(docstore_path, 'w') as f:
                    json.dump(empty_docstore, f, indent=2)
                logger.info("Initialized empty document store")
            
            # Initialize other necessary index files
            vector_store_path = self.index_path / "vector_store.json"
            if not vector_store_path.exists():
                empty_vector_store = {
                    "vector_store": {"embeddings": {}, "metadata": {}},
                    "version": "1.0.0"
                }
                with open(vector_store_path, 'w') as f:
                    json.dump(empty_vector_store, f, indent=2)
                logger.info("Initialized empty vector store")
            
        except Exception as e:
            logger.error(f"Failed to initialize storage structure: {e}")
            raise RuntimeError(f"Storage initialization failed: {str(e)}")

    def create_index(self, documents: List) -> VectorStoreIndex:
        """Create a vector index from processed documents.
        
        This method processes the input documents and creates a searchable index.
        It ensures all necessary storage components are properly initialized and
        updated.
        
        Args:
            documents: List of documents to process and index
            
        Returns:
            VectorStoreIndex: The created index ready for querying
        """
        logger.info("Creating vector index")
        
        try:
            # Create service context with updated configurations
            service_context = ServiceContext.from_defaults(
                llm=self.llm,
                embed_model=self.embed_model,
                node_parser=self.text_splitter,
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP
            )
            
            # Create index with updated configurations
            self.index = VectorStoreIndex.from_documents(
                documents,
                storage_context=self.storage_context,
                service_context=service_context,
                show_progress=True
            )
            
            # Save the index immediately after creation
            self.save_index(self.index_path)
            
            logger.info("Vector index created and saved successfully")
            return self.index
            
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            raise RuntimeError(f"Index creation failed: {str(e)}")

    def save_index(self, path: Path):
        """Save the current index and associated data to disk.
        
        This method ensures all components of the index, including the document
        store, vector store, and FAISS index are properly saved to disk.
        """
        if not self.index:
            logger.warning("No index to save")
            return
            
        try:
            # Create the directory if it doesn't exist
            path.mkdir(parents=True, exist_ok=True)
            
            # Save the llama index with storage context
            self.index.storage_context.persist(persist_dir=str(path))
            
            # Save the FAISS index separately
            faiss_path = path / "faiss.index"
            faiss.write_index(self.faiss_index, str(faiss_path))
            
            logger.info(f"Index successfully saved to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            raise RuntimeError(f"Index saving failed: {str(e)}")

    def load_index(self, path: Path):
        """Load an index from disk."""
        if not path.exists():
            logger.error(f"Index path {path} does not exist")
            raise FileNotFoundError(f"No index found at {path}")
            
        # Load the FAISS index with dimension check
        faiss_path = path / "faiss.index"
        if faiss_path.exists():
            self.faiss_index = faiss.read_index(str(faiss_path))
            if self.faiss_index.d != self.dimension:
                raise ValueError(f"FAISS index dimension {self.faiss_index.d} does not match expected dimension {self.dimension}")
            self.vector_store = FaissVectorStore(
                faiss_index=self.faiss_index,
                dim=self.dimension
            )
            
        # Load the llama index with updated storage context
        storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store,
            persist_dir=str(path)
        )
        
        self.index = VectorStoreIndex.load_from_storage(
            storage_context=storage_context,
            service_context=ServiceContext.from_defaults(
                llm=self.llm,
                embed_model=self.embed_model
            )
        )
        logger.info(f"Index loaded from {path}")
