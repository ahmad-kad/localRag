# Standard library imports
import os
from pathlib import Path
import logging
import torch

# Environment and configuration
from dotenv import load_dotenv

# LlamaIndex imports
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.settings import Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.ollama import Ollama 

# LangChain imports
from langchain_huggingface import HuggingFaceEmbeddings


# Load environment variables and set up logging
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_rag_system():
    """Initialize the RAG system with proper MPS configuration for M3 Pro."""
    # Clear any existing MPS settings first
    if 'PYTORCH_MPS_HIGH_WATERMARK_RATIO' in os.environ:
        del os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO']
    
    # Set MPS settings before any PyTorch operations
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.4'
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    # Force PyTorch to recognize these settings
    import torch
    torch.backends.mps.enable_fallback_for_not_implemented_ops = True
    
    # Verify MPS availability and settings
    if torch.backends.mps.is_available():
        device = 'mps'
        logger.info("MPS is available. Verifying settings...")
        try:
            # Test MPS with a small tensor operation
            test_tensor = torch.ones((1, 1), device='mps')
            logger.info("MPS test successful")
        except Exception as e:
            logger.warning(f"MPS test failed: {str(e)}")
            device = 'cpu'
            logger.info("Falling back to CPU")
    else:
        device = 'cpu'
        logger.warning("MPS is not available. Using CPU.")

    # Set up directory structure
    cache_dir = Path(os.getenv('CACHE_DIR', './data/cache'))
    index_dir = Path(os.getenv('INDEX_DIR', './data/indexes'))
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    # Initialize the language model
    llm = Ollama(
        model=os.getenv('MODEL_NAME', 'mistral-openorca'),
        temperature=float(os.getenv('TEMPERATURE', '0.1')),
        request_timeout=120.0,
        additional_kwargs={
            "system": "You are an expert code analysis assistant. "
                     "Analyze code thoroughly and provide detailed, accurate explanations."
        }
    )
    logger.info(f"Initialized language model: {os.getenv('MODEL_NAME', 'mistral-openorca')}")

    try:
        # Initialize the embedding model with explicit device verification
        embed_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={
                'device': device,
            }
        )
        logger.info(f"Successfully initialized embedding model on {device}")
    except RuntimeError as e:
        logger.warning(f"Failed to initialize on {device}, falling back to CPU. Error: {str(e)}")
        embed_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={
                'device': 'cpu',
            }
        )
        logger.info("Initialized embedding model on CPU")

    # Rest of the setup remains the same...
    node_parser = SimpleNodeParser.from_defaults(
        chunk_size=int(os.getenv('CHUNK_SIZE', '1024')),
        chunk_overlap=int(os.getenv('CHUNK_OVERLAP', '128')),
        include_metadata=True,
        include_prev_next_rel=True
    )

    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.node_parser = node_parser

    logger.info("RAG system initialization complete")
    return Settings

def process_code_directory(directory_path: str):
    """Process a directory of code files to create a searchable index.
    
    This function reads all code files from the specified directory and creates
    a vector store index that can be used for semantic code search and analysis.
    The index uses the globally configured settings for processing and storing
    the code chunks.
    
    Args:
        directory_path: Path to the directory containing code files
    
    Returns:
        VectorStoreIndex: An index that can be queried to find relevant code sections
    """
    logger.info(f"Processing directory: {directory_path}")
    
    # Load all documents from the specified directory
    reader = SimpleDirectoryReader(
        input_dir=directory_path,
        exclude_hidden=True
    )
    documents = reader.load_data()
    logger.info(f"Loaded {len(documents)} documents")
    
    # Create the index using our configured settings
    index = VectorStoreIndex.from_documents(documents)
    
    return index

def test_queries(index):
    """Test the system with example queries to verify its functionality.
    
    This function runs a series of test queries against the index to ensure
    the system can effectively analyze and explain code patterns.
    
    Args:
        index: The vector store index containing our processed code
    """
    query_engine = index.as_query_engine()
    
    test_queries = [
        "What are the main classes in this codebase?",
        "Explain the error handling patterns used in the code",
        "How is data validation implemented?",
        "Show me examples of good coding practices in this codebase"
    ]
    
    for query in test_queries:
        logger.info(f"\nExecuting query: {query}")
        response = query_engine.query(query)
        print(f"\nResponse: {response}\n")
        print("-" * 80)

if __name__ == "__main__":
    try:
        # Set up the RAG system with the new settings approach
        settings = setup_rag_system()
        
        # Process your code directory
        code_directory = "src/code_rag"  # Adjust this path to your code directory
        index = process_code_directory(code_directory)
        
        # Run test queries
        test_queries(index)
        
    except Exception as e:
        logger.error(f"An error occurred during setup: {str(e)}")
        raise