import os
import faiss
import numpy as np
import gradio as gr
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Environment + Config
# -----------------------------
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.getenv("DATA_PATH", os.path.join(BASE_DIR, "sample.txt"))
INDEX_PATH = os.getenv("INDEX_PATH", os.path.join(BASE_DIR, "faiss_index.bin"))
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "google/flan-t5-small")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
TOP_K = int(os.getenv("TOP_K", 5))
OVERLAP = int(os.getenv("OVERLAP", 100))

# -----------------------------
# Load Embeddings + FAISS Index
# -----------------------------
embedder = SentenceTransformer(EMBED_MODEL_NAME)

def chunk_text(text, size=CHUNK_SIZE, overlap=OVERLAP):
    """Split text into overlapping chunks for better context retention."""
    paragraphs = text.split("\n\n")
    chunks = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        start = 0
        while start < len(para):
            end = min(len(para), start + size)
            chunks.append(para[start:end])
            start += size - overlap
    return chunks

def load_documents(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return chunk_text(text, CHUNK_SIZE, OVERLAP)

docs = load_documents(DATA_PATH)

def build_or_load_index(docs_list):
    embeddings = np.array(embedder.encode(docs_list, show_progress_bar=True), dtype="float32")
    embeddings = normalize(embeddings, axis=1)
    dim = embeddings.shape[1]

    if os.path.exists(INDEX_PATH):
        idx = faiss.read_index(INDEX_PATH)
        if idx.ntotal == 0:
            idx.add(embeddings)
            faiss.write_index(idx, INDEX_PATH)
    else:
        idx = faiss.IndexFlatIP(dim)
        idx.add(embeddings)
        faiss.write_index(idx, INDEX_PATH)
    return idx

index = build_or_load_index(docs)

# -----------------------------
# Load LLM
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME)

# -----------------------------
# Prompt
# -----------------------------
STRONG_SYSTEM_INSTRUCTION = (
    "You are a highly knowledgeable AI assistant. "
    "Answer the user's question using ONLY the provided CONTEXT. "
    "Do NOT hallucinate. "
    "If the context does not contain a direct answer, respond: 'I don't know based on the provided context.' "
    "Answer in a clear, detailed, and professional manner."
)

# -----------------------------
# Memory & Evaluation
# -----------------------------
chat_memory = []

def evaluate_retrieval_quality(question, context):
    """Calculate cosine similarity between query and retrieved context."""
    q_vec = embedder.encode([question])
    c_vec = embedder.encode([context])
    score = cosine_similarity(q_vec, c_vec)[0][0]
    return round(float(score), 4)

# -----------------------------
# Answer Generation
# -----------------------------
def generate_answer(question, top_k=TOP_K, max_new_tokens=300):
    try:
        q_vec = np.array([embedder.encode(question)], dtype="float32")
        q_vec = normalize(q_vec, axis=1)
        D, I = index.search(q_vec, k=min(top_k, len(docs)))

        context = "\n\n".join([docs[int(idx)] for idx in I[0] if idx < len(docs)])
        retrieval_score = evaluate_retrieval_quality(question, context)

        conversation_context = " ".join([f"User: {q}\nAI: {a}" for q, a in chat_memory[-3:]])

        prompt = (
            STRONG_SYSTEM_INSTRUCTION + "\n\n"
            f"PREVIOUS CONVERSATION:\n{conversation_context}\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {question}\n\n"
            "Answer now:"
        )

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        chat_memory.append((question, answer))
        print(f"🔍 Retrieval similarity score: {retrieval_score}")
        return f"{answer}\n\n🧭 Retrieval Similarity Score: {retrieval_score}"
    except Exception as e:
        return f"⚠️ Error generating answer: {e}"

# -----------------------------
# Chat UI
# -----------------------------
def chat_stream(user_message, chat_history):
    chat_history = chat_history or []
    chat_history.append((user_message, "Thinking..."))
    yield chat_history, chat_history, ""
    answer = generate_answer(user_message)
    chat_history[-1] = (user_message, answer)
    yield chat_history, chat_history, ""

# -----------------------------
# Sidebar Questions
# -----------------------------
SIDEBAR_QUESTIONS = """
**💡 Example Questions (copy & paste to ask):**

- What is LangGraph?
- What are the key features of LangGraph?
- How does LangGraph help in collaboration?
- What are Trees?
- What are Embeddings?
- What is RAG (Retrieval-Augmented Generation)?
- What are Python & AI Concepts?
- What is a Binary Search Tree (BST)?
- What is the role of a retriever in RAG?
- Why do we split documents into chunks for RAG?
- What are Transformers models used for?
- How does combining embeddings with LLMs help?
- How are embeddings used in chatbots?
- How can embeddings help in semantic search?
- Name some Python libraries used for AI and machine learning.
- Name some types of trees.
- What are some applications of trees?
"""

# -----------------------------
# Gradio Interface
# -----------------------------
with gr.Blocks(title="RAG-Powered LangGraph QA Assistant") as demo:
    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("## 💬 RAG-Powered LangGraph QA Assistant\nAsk questions on LangGraph and get detailed answers")
            chatbot = gr.Chatbot()
            state = gr.State([])

            txt = gr.Textbox(show_label=False, placeholder="Type your question here...")
            submit = gr.Button("Send")
            submit.click(fn=chat_stream, inputs=[txt, state], outputs=[chatbot, state, txt])
            txt.submit(fn=chat_stream, inputs=[txt, state], outputs=[chatbot, state, txt])
            clear_btn = gr.Button("Clear")
            clear_btn.click(lambda: ([], []), inputs=None, outputs=[chatbot, state])

        with gr.Column(scale=1):
            gr.Markdown(SIDEBAR_QUESTIONS)

demo.launch(share=False)
