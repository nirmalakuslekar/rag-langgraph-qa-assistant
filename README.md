<<<<<<< HEAD
# rag-langgraph-qa-assistant
RAG-powered LangGraph QA Assistant using embeddings and LLM
=======
# RAG-Powered LangGraph QA Assistant

This project is a **Retrieval-Augmented Generation (RAG)** powered AI assistant that answers questions based on a text file (`sample.txt`) using embeddings and a large language model (LLM).

## Features
- Uses **FAISS** for vector search on text chunks.
- Integrates **Sentence Transformers** for embeddings.
- Uses **FLAN-T5** as the LLM for question answering.
- Provides a **Gradio web interface** for interactive QA.
- Example questions included in sidebar for quick testing.

## Setup Instructions

1. **Clone/download this repository**.
2. **Ensure Python 3.10+ is installed** and added to PATH.
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
 
### How To Run 
1. **Open a terminal/command prompt in the project folder.**
2. **Run: python main.py Or, if your system uses python3: python3 main.py**
3. **The Gradio web interface will start, and a local URL will be printed in the console.**
4. **Open this URL in your browser to interact with the assistant.**

> Note: The FAISS index (`faiss_index.bin`) is automatically created when you run `main.py` for the first time.
>>>>>>> 1d911e5 (Initial commit: RAG LangGraph QA Assistant)
