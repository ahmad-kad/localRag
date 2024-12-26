import os
import gradio as gr
import logging
import torch
from pathlib import Path
from typing import List, Tuple
from datetime import datetime
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

class RAGInterface:
    """Web interface for the RAG system that handles code analysis."""
    
    def __init__(self):
        """Initialize the RAG interface with the tested configuration."""
        # Set up directory structure
        self.cache_dir = Path(os.getenv('CACHE_DIR', './data/cache'))
        self.index_dir = Path(os.getenv('INDEX_DIR', './data/indexes'))
        self._create_directories()
        
        # Initialize core components
        self.settings = None
        self.index = None
        self.query_engine = None
        
        # Set up the RAG system
        self._setup_rag_system()
        logger.info("RAG Interface initialized successfully")

    def _create_directories(self):
        """Create necessary directories for storing indexes and cache."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directories: cache={self.cache_dir}, index={self.index_dir}")

    def _setup_rag_system(self):
        """Initialize the RAG system with proven MPS configuration."""
        # Clear any existing MPS settings first
        if 'PYTORCH_MPS_HIGH_WATERMARK_RATIO' in os.environ:
            del os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO']
        
        # Set MPS settings before any PyTorch operations
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.4'
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        
        # Force PyTorch to recognize these settings
        torch.backends.mps.enable_fallback_for_not_implemented_ops = True
        
        # Determine device
        device = self._determine_device()
        logger.info(f"Using device: {device}")
        
        try:
            # Initialize the language model
            self.llm = Ollama(
                model=os.getenv('MODEL_NAME', 'mistral-openorca'),
                temperature=float(os.getenv('TEMPERATURE', '0.1')),
                request_timeout=120.0,
                additional_kwargs={
                    "system": "You are an expert code analysis assistant. "
                             "Analyze code thoroughly and provide detailed, accurate explanations."
                }
            )
            
            # Initialize embedding model
            self.embed_model = HuggingFaceEmbeddings(
                model_name="BAAI/bge-small-en-v1.5",
                model_kwargs={'device': device}
            )
            
            # Configure node parser
            self.node_parser = SimpleNodeParser.from_defaults(
                chunk_size=int(os.getenv('CHUNK_SIZE', '1024')),
                chunk_overlap=int(os.getenv('CHUNK_OVERLAP', '128')),
                include_metadata=True,
                include_prev_next_rel=True
            )
            
            # Set up global settings
            Settings.llm = self.llm
            Settings.embed_model = self.embed_model
            Settings.node_parser = self.node_parser
            self.settings = Settings
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            raise

    def _determine_device(self) -> str:
        """Determine the appropriate device with MPS support."""
        if torch.backends.mps.is_available():
            logger.info("MPS is available. Verifying settings...")
            try:
                # Test MPS with a small tensor operation
                test_tensor = torch.ones((1, 1), device='mps')
                logger.info("MPS test successful")
                return 'mps'
            except Exception as e:
                logger.warning(f"MPS test failed: {str(e)}")
                logger.info("Falling back to CPU")
                return 'cpu'
        
        logger.warning("MPS is not available. Using CPU.")
        return 'cpu'

    def initialize_system(
        self,
        directory: str,
        model_name: str,
        load_existing: bool
    ) -> str:
        """Initialize the system with a code directory."""
        try:
            directory_path = Path(directory)
            if not directory_path.exists():
                return "❌ Directory does not exist"
            
            logger.info(f"Processing directory: {directory}")
            
            # Load documents using SimpleDirectoryReader
            reader = SimpleDirectoryReader(
                input_dir=directory,
                exclude_hidden=True
            )
            documents = reader.load_data()
            logger.info(f"Loaded {len(documents)} documents")
            
            if not documents:
                return "❌ No documents found in directory"
            
            # Create index using our configured settings
            self.index = VectorStoreIndex.from_documents(documents)
            self.query_engine = self.index.as_query_engine()
            
            return f"✅ Successfully processed {len(documents)} files"
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            return f"❌ Error: {str(e)}"

    def process_query(
        self,
        query: str,
        history: List[Tuple[str, str]],
        show_sources: bool
    ) -> Tuple[List[Tuple[str, str]], str]:
        """Process a query about the codebase."""
        if not self.query_engine:
            return (
                history + [(query, "⚠️ Please select and process a code directory first")],
                ""
            )
        
        try:
            response = self.query_engine.query(query)
            formatted_response = str(response)
            
            if show_sources and hasattr(response, 'source_nodes'):
                source_text = "\n\n📚 Source Files:"
                for idx, node in enumerate(response.source_nodes, 1):
                    source_text += f"\n{idx}. {node.metadata.get('file_path', 'Unknown')}"
                formatted_response += source_text
            
            return history + [(query, formatted_response)], ""
            
        except Exception as e:
            logger.error(f"Query error: {e}")
            return history + [(query, f"❌ Error: {str(e)}")], ""

    def clear_history(self) -> List[Tuple[str, str]]:
        """Clear the chat history."""
        return []

def create_interface() -> gr.Blocks:
    """Create the Gradio interface for the RAG system."""
    try:
        rag = RAGInterface()
        
        with gr.Blocks(
            title="Code RAG Assistant",
            theme=gr.themes.Soft()
        ) as interface:
            gr.Markdown("# 🤖 Code RAG Assistant")
            gr.Markdown("Ask questions about your codebase using natural language.")
            
            with gr.Row():
                with gr.Column(scale=1):
                    # System initialization
                    with gr.Group():
                        gr.Markdown("### System Setup")
                        directory_input = gr.Textbox(
                            label="Code Directory Path",
                            placeholder="/path/to/your/code"
                        )
                        model_dropdown = gr.Dropdown(
                            choices=["mistral-openorca", "llama2", "codellama"],
                            value="mistral-openorca",
                            label="Select Model"
                        )
                        load_existing = gr.Checkbox(
                            label="Load Existing Index",
                            value=False
                        )
                        initialize_btn = gr.Button("Initialize System")
                        status_box = gr.Textbox(
                            label="Status",
                            interactive=False
                        )
                    
                    # Settings
                    with gr.Group():
                        gr.Markdown("### Settings")
                        show_sources = gr.Checkbox(
                            label="Show Source Files",
                            value=True
                        )
                
                with gr.Column(scale=2):
                    # Chat interface
                    chatbot = gr.Chatbot(
                        label="Chat History",
                        height=500,
                        show_label=False
                    )
                    query_box = gr.Textbox(
                        label="Ask about your code",
                        placeholder="Enter your question here...",
                        lines=2
                    )
                    clear_btn = gr.Button("Clear History")
            
            # Example queries
            gr.Examples(
                examples=[
                    "What are the main functions in this codebase?",
                    "Explain the error handling patterns used",
                    "How is data validation implemented?",
                    "Show me the main data processing pipeline",
                    "What external dependencies are used?"
                ],
                inputs=query_box
            )
            
            # Event handlers
            initialize_btn.click(
                fn=rag.initialize_system,
                inputs=[directory_input, model_dropdown, load_existing],
                outputs=status_box
            )
            
            query_box.submit(
                fn=rag.process_query,
                inputs=[query_box, chatbot, show_sources],
                outputs=[chatbot, query_box]
            )
            
            clear_btn.click(
                fn=rag.clear_history,
                outputs=chatbot
            )
        
        return interface
        
    except Exception as e:
        logger.error(f"Failed to create interface: {e}")
        raise RuntimeError(f"Interface creation failed: {str(e)}")

if __name__ == "__main__":
    try:
        # Create and launch the interface
        demo = create_interface()
        demo.launch(
            server_name="0.0.0.0",
            server_port=int(os.getenv('API_PORT', '8000')),
            share=True,
            debug=os.getenv('DEBUG', 'False').lower() == 'true'
        )
    except Exception as e:
        logger.error(f"Failed to launch application: {e}")
        raise