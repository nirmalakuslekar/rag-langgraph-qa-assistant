<<<<<<< HEAD
# rag-langgraph-qa-assistant
RAG-powered LangGraph QA Assistant using embeddings and LLM
=======
# RAG-Powered LangGraph QA Assistant

This project is a **Retrieval-Augmented Generation (RAG)** powered AI assistant that answers questions based on a text file (`sample.txt`) using embeddings and a large language model (LLM).


- **RAG Architecture**: Combines vector similarity search with a local LLM for grounded responses.
- **Embedding Search**: Uses `sentence-transformers/all-MiniLM-L6-v2` to retrieve semantically similar text chunks.
- **Contextual Answering**: Generates responses based only on retrieved context (prevents hallucinations).
- **Memory Simulation**: Maintains ongoing chat history to simulate conversational memory.
- **Interactive Chat UI**: Built using `Gradio` for an intuitive, ChatGPT-style interface.
- **FAISS Vector Index**: Efficient similarity search for retrieval.
- **Prompt Engineering**: Strong system prompt for clarity and accuracy.

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
