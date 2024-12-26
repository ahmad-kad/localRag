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