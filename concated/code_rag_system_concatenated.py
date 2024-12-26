// Source: /Users/ahmadkaddoura/Desktop/code_rag_system/verify_setup.py
// Relative path: verify_setup.py
# verify_setup.py
import torch
import platform
import psutil
import os
from pathlib import Path

def verify_system():
    print("System Configuration:")
    print(f"macOS Version: {platform.mac_ver()[0]}")
    print(f"Python Version: {platform.python_version()}")
    print(f"Total RAM: {psutil.virtual_memory().total / (1024**3):.1f}GB")
    
    print("\nPyTorch Configuration:")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"MPS Available: {torch.backends.mps.is_available()}")
    print(f"MPS Built: {torch.backends.mps.is_built()}")
    
    print("\nOllama Configuration:")
    ollama_config = Path.home() / ".ollama/config.json"
    if ollama_config.exists():
        print("Ollama config found ✓")
    else:
        print("Warning: Ollama config not found")
    
    print("\nEnvironment Variables:")
    mps_ratio = os.getenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "Not Set")
    mps_fallback = os.getenv("PYTORCH_ENABLE_MPS_FALLBACK", "Not Set")
    print(f"MPS Watermark Ratio: {mps_ratio}")
    print(f"MPS Fallback: {mps_fallback}")

if __name__ == "__main__":
    verify_system()

// Source: /Users/ahmadkaddoura/Desktop/code_rag_system/concat.py
// Relative path: concat.py
import os
import sys
from pathlib import Path
import fnmatch
from typing import List, Set

class GitignoreParser:
    """
    Handles parsing and matching of .gitignore patterns.
    """
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.ignore_patterns: List[str] = []
        self.load_gitignore()

    def load_gitignore(self) -> None:
        """
        Loads all .gitignore files from the root directory and its parents.
        This respects the git behavior of considering all .gitignore files
        in the hierarchy.
        """
        current_dir = self.root_dir
        while current_dir.exists():
            gitignore_path = current_dir / '.gitignore'
            if gitignore_path.is_file():
                with open(gitignore_path, 'r', encoding='utf-8') as f:
                    # Filter out empty lines and comments
                    patterns = [
                        line.strip() 
                        for line in f 
                        if line.strip() and not line.startswith('#')
                    ]
                    self.ignore_patterns.extend(patterns)
            # Move up to parent directory
            parent_dir = current_dir.parent
            if parent_dir == current_dir:  # Reached root
                break
            current_dir = parent_dir

    def should_ignore(self, path: Path) -> bool:
        """
        Determines if a path should be ignored based on .gitignore rules.
        
        Args:
            path: Path to check against gitignore rules
            
        Returns:
            bool: True if the path should be ignored, False otherwise
        """
        # Get relative path from root directory
        rel_path = str(path.relative_to(self.root_dir))
        
        # Convert Windows paths to forward slashes for consistency
        rel_path = rel_path.replace(os.sep, '/')
        
        for pattern in self.ignore_patterns:
            # Handle both directory and file patterns
            if pattern.endswith('/'):
                if fnmatch.fnmatch(f"{rel_path}/", pattern):
                    return True
            if fnmatch.fnmatch(rel_path, pattern):
                return True
        return False

def concat_py_files(root_dir: str) -> None:
    """
    Concatenates JavaScript files respecting directory structure and .gitignore rules.
    Creates separate concatenated files for each directory level.
    
    Args:
        root_dir: Root directory to start processing from
    """
    root_path = Path(root_dir).resolve()
    output_dir = root_path / 'concated'
    output_dir.mkdir(exist_ok=True)
    
    # Initialize gitignore parser
    gitignore = GitignoreParser(root_path)
    
    def process_directory(dir_path: Path, relative_output_path: Path) -> None:
        """
        Recursively processes directories and their JavaScript files.
        
        Args:
            dir_path: Current directory being processed
            relative_output_path: Relative path for maintaining directory structure
        """
        # Skip if directory should be ignored
        if gitignore.should_ignore(dir_path):
            print(f'Skipping ignored directory: {dir_path}')
            return
        
        # Create corresponding output directory
        current_output_dir = output_dir / relative_output_path
        current_output_dir.mkdir(exist_ok=True, parents=True)
        
        # Get all py files in current directory
        py_files = [
            f for f in dir_path.glob('*.py')
            if not gitignore.should_ignore(f)
        ]
        
        if py_files:
            # Create concatenated file for current directory
            concatenated = []
            for py_file in py_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        full_path = py_file.resolve()
                        concatenated.append(
                            f'// Source: {full_path}\n'
                            f'// Relative path: {py_file.relative_to(root_path)}\n'
                            f'{f.read()}\n'
                        )
                except Exception as e:
                    print(f'Error reading {py_file}: {str(e)}')
                    continue
            
            # Write concatenated file for current directory
            output_file = current_output_dir / f'{dir_path.name}_concatenated.py'
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(concatenated))
            
            print(f'Created {output_file} from {len(py_files)} files')
        
        # Process subdirectories
        for subdir in dir_path.iterdir():
            if subdir.is_dir() and subdir.name != 'concated':
                process_directory(
                    subdir,
                    relative_output_path / subdir.name
                )

    # Start processing from root directory
    process_directory(root_path, Path(''))

if __name__ == '__main__':
    target_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    
    try:
        concat_py_files(target_dir)
        print('Concatenation complete!')
    except Exception as e:
        print(f'Error: {str(e)}')
        sys.exit(1)

// Source: /Users/ahmadkaddoura/Desktop/code_rag_system/test_core.py
// Relative path: test_core.py
# test_core.py
from pathlib import Path
import time
import psutil
import logging
from src.code_rag.indexer import CodeIndexer
from src.code_rag.retriever import CodeRetriever

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_system_usage():
    """Monitor system resources."""
    process = psutil.Process()
    memory_info = process.memory_info()
    logger.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
    logger.info(f"CPU usage: {psutil.cpu_percent()}%")

def test_system():
    start_time = time.time()
    indexer = CodeIndexer()
    
    # Create test directory
    test_dir = Path("test_code")
    test_dir.mkdir(exist_ok=True)
    
    # Create sample file
    test_file = test_dir / "sample.py"
    test_file.write_text("""
def hello():
    '''A simple greeting function'''
    print("Hello, World!")

def add(a, b):
    '''Add two numbers'''
    return a + b
    """)
    
    try:
        # Process directory
        logger.info("Processing directory...")
        log_system_usage()
        docs = indexer.process_directory(test_dir)
        logger.info(f"Found {len(docs)} documents")
        
        # Create index
        logger.info("Creating index...")
        log_system_usage()
        index = indexer.create_index(docs)
        
        # Setup retriever
        logger.info("Setting up retriever...")
        retriever = CodeRetriever(index)
        
        # Test query
        logger.info("Testing query...")
        result = retriever.process_query("What does the hello function do?")
        logger.info("\nQuery result:")
        print(result.response)
        
        # Log performance
        end_time = time.time()
        logger.info(f"Total execution time: {end_time - start_time:.2f} seconds")
        log_system_usage()
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise
    
    finally:
        # Cleanup
        if test_file.exists():
            test_file.unlink()
        if test_dir.exists():
            test_dir.rmdir()

if __name__ == "__main__":
    test_system()

// Source: /Users/ahmadkaddoura/Desktop/code_rag_system/test_deepseek.py
// Relative path: test_deepseek.py
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

// Source: /Users/ahmadkaddoura/Desktop/code_rag_system/test_rag.py
// Relative path: test_rag.py
# test_rag.py
from pathlib import Path
import time
import psutil
import torch
from src.code_rag.indexer import CodeIndexer
from src.code_rag.retriever import CodeRetriever
import logging

logging.basicConfig(level=logging.INFO)

def test_rag():
    # Performance monitoring
    start_time = time.time()
    initial_memory = psutil.Process().memory_info().rss / 1024 / 1024

    # Test directory setup
    test_dir = Path("test_code")
    test_dir.mkdir(exist_ok=True)
    
    # Sample code with different patterns
    (test_dir / "main.py").write_text("""
class DataProcessor:
    def process_batch(self, data):
        '''Process a batch of data'''
        return [x * 2 for x in data]
        
    def validate(self, item):
        '''Validate input data'''
        return isinstance(item, (int, float))
""")

    try:
        # Initialize and test
        indexer = CodeIndexer()
        docs = indexer.process_directory(test_dir)
        index = indexer.create_index(docs)
        retriever = CodeRetriever(index)

        # Test queries
        queries = [
            "What does the DataProcessor class do?",
            "How does the validation work?",
            "Show me the data processing logic"
        ]

        for query in queries:
            print(f"\nQuery: {query}")
            response = retriever.process_query(query)
            print(f"Response: {response}")

        # Performance metrics
        end_time = time.time()
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        print(f"\nPerformance Metrics:")
        print(f"Total time: {end_time - start_time:.2f}s")
        print(f"Memory usage: {final_memory - initial_memory:.2f}MB")
        print(f"Using MPS: {torch.backends.mps.is_available()}")

    finally:
        # Cleanup
        for file in test_dir.glob("*"):
            file.unlink()
        test_dir.rmdir()

if __name__ == "__main__":
    test_rag()
