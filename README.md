# Code RAG System

The Code RAG System is an advanced Retrieval-Augmented Generation solution engineered for deep code analysis and intuitive querying of large-scale codebases. Leveraging local LLMs via Ollama and state-of-the-art code-specific embeddings, this platform empowers developers to rapidly gain insights into complex code architectures and accelerate their development workflows.

## Key Highlights

- **Local LLM Integration:** Utilizes Ollama for on-premise language model support, ensuring data privacy and reduced latency.
- **Custom Code Embeddings:** Implements specialized embeddings to capture the unique syntax and semantics of diverse programming languages.
- **Interactive Query Interface:** Provides a natural language interface for direct, conversational code exploration.
- **Multi-Language & Scalable:** Designed to process and analyze projects in various programming languages while handling both small and large codebases effectively.
- **Efficient Retrieval:** Employs vector storage (powered by FAISS) for lightning-fast similarity searches and retrieval of relevant code snippets.

## Technical Overview

- **Modern Tech Stack:** Built with Python 3.9+, PyTorch, Sentence-Transformers, and FAISS to ensure high performance and accuracy.
- **Optimized for Performance:** Configured to leverage hardware accelerations such as Metal Performance Shaders (MPS) on Apple Silicon.
- **Containerized Deployment:** Supports Docker and Docker Compose for streamlined setup, development, and production deployment.
- **Robust & Extensible:** Uses Pydantic for configuration management, enabling easy customization and scalable architecture.

## Quick Start

1. **Clone the Repository:**
   ```bash
   git clone <repository-url>
   cd code_rag_system
   ```

2. **Set Up the Environment:**
   ```bash
   make setup
   ```

3. **Configure Environment Variables:**
   - Update the `.env` file with your custom settings.

4. **Launch the Application:**
   ```bash
   make run
   ```

## Collaboration & Contribution

This open-source project welcomes contributions. Whether you're optimizing performance, adding new features, or integrating additional language support, your ideas and expertise can drive further innovation in intelligent code analysis.

## License

This project is licensed under the MIT License.

---

Elevate your code analysis capabilities with the Code RAG System — a cutting-edge tool that transforms complex codebases into actionable insights, enhancing both productivity and innovation.
