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