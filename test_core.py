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