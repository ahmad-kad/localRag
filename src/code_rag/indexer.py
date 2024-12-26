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