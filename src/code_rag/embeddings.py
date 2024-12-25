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