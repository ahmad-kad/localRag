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