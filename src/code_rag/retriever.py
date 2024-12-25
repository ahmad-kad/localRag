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