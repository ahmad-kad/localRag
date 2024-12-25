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