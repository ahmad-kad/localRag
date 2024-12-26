from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Settings optimized for M3 Pro with RAG system"""
    # Model settings
    MODEL_NAME: str = "mistral-openorca"
    EMBED_MODEL: str = "BAAI/bge-small-en-v1.5"  # Optimized for code understanding
    TEMPERATURE: float = 0.1
    MAX_TOKENS: int = 4096
    
    # M3 Pro specific settings
    DEVICE: str = "mps"  # Metal Performance Shaders for Apple Silicon
    BATCH_SIZE: int = 8  
    
    # Vector store settings
    VECTOR_STORE_TYPE: str = "faiss"  # Using FAISS for vector storage
    VECTOR_DIM: int = 384  # BGE embedding dimension
    
    # Memory-optimized chunking
    CHUNK_SIZE: int = 2048
    CHUNK_OVERLAP: int = 256
    
    # Paths
    CACHE_DIR: Path = Path("data/cache")
    INDEX_DIR: Path = Path("data/indexes")
    FAISS_INDEX_PATH: Path = Path("data/indexes/faiss.index")

    # Web interface settings
    API_PORT: int = 8000
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        extra='ignore'
    )

# Create and export a settings instance
settings = Settings()