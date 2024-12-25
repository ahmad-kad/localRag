// Source: /Users/ahmadkaddoura/Desktop/code_rag_system/src/code_rag/web/__init__.py
// Relative path: src/code_rag/web/__init__.py


// Source: /Users/ahmadkaddoura/Desktop/code_rag_system/src/code_rag/web/app.py
// Relative path: src/code_rag/web/app.py
import gradio as gr
import logging
from pathlib import Path
from typing import List, Tuple
from datetime import datetime

from ..config import settings
from ..indexer import CodeIndexer
from ..retriever import CodeRetriever
from ..utils import setup_logging

logger = logging.getLogger(__name__)

class RAGInterface:
    """Web interface for the RAG system."""
    
    def __init__(self):
        setup_logging(settings.LOG_LEVEL)
        self.indexer = CodeIndexer()
        self.retriever = None
        self.index_status = "Not initialized"
        
    def initialize_system(
        self,
        directory: str,
        model_name: str,
        load_existing: bool
    ) -> str:
        """Initialize or load the RAG system."""
        try:
            directory_path = Path(directory)
            
            if load_existing:
                logger.info("Loading existing index...")
                self.indexer.load_index()
                self.retriever = CodeRetriever(self.indexer.index)
                return "✅ Loaded existing index"
            
            if not directory_path.exists():
                return "❌ Invalid directory path"
            
            logger.info(f"Processing directory: {directory}")
            docs = self.indexer.process_directory(directory_path)
            index = self.indexer.create_index(docs)
            self.indexer.save_index()  # Save for future use
            
            self.retriever = CodeRetriever(index)
            return f"✅ System initialized with {len(docs)} documents"
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            return f"❌ Error: {str(e)}"
    
    def process_query(
        self,
        query: str,
        history: List[Tuple[str, str]],
        show_sources: bool
    ) -> Tuple[List[Tuple[str, str]], str]:
        """Process a query and update chat history."""
        if not self.retriever:
            return history + [(query, "⚠️ Please initialize the system first")], ""
            
        try:
            result = self.retriever.process_query(query)
            
            # Format response
            response = result.response
            
            # Add sources if requested
            if show_sources and result.sources:
                source_text = "\n\n📚 Sources:\n"
                for idx, source in enumerate(result.sources, 1):
                    file_path = source.document.metadata.get('file_path', 'Unknown')
                    score = f"{source.score:.2f}"
                    source_text += f"{idx}. {file_path} (Score: {score})\n"
                response += source_text
            
            return history + [(query, response)], ""
            
        except Exception as e:
            logger.error(f"Query processing error: {e}")
            return history + [(query, f"❌ Error: {str(e)}")], ""
    
    def clear_history(self) -> List[Tuple[str, str]]:
        """Clear chat history."""
        return []

def create_interface() -> gr.Blocks:
    """Create the Gradio interface."""
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
        
        # Add keyboard shortcuts
        query_box.submit(lambda: "", outputs=query_box)  # Clear input after submit
        
    return interface

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=settings.API_PORT,
        share=True,
        debug=settings.DEBUG
    )
