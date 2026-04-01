import fitz
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
import streamlit as st

# -----------------------------
# GROQ CONFIG
# -----------------------------
MODEL_NAME = "llama-3.1-8b-instant"


class RAGEngine:

    def __init__(self):

        # ✅ Groq client (CORRECT PLACE)
        self.client = Groq(
            api_key=st.secrets["GROQ_API_KEY"]
        )

        self.embedder = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device="cpu"
        )

        self.chunks = []
        self.embeddings = None
        self.chat_history = []

    # --------------------------------
    # TEXT SPLITTER
    # --------------------------------
    def split_text(self, text, chunk_size=500, overlap=100):

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap

        return chunks

    # --------------------------------
    # LOAD PDFs
    # --------------------------------
    def load_pdfs(self, files):

        self.chunks = []

        for file in files:
            doc = fitz.open(stream=file.read(), filetype="pdf")

            for page_num, page in enumerate(doc):

                text = page.get_text().strip()
                if not text:
                    continue

                for chunk in self.split_text(text):
                    self.chunks.append({
                        "text": chunk,
                        "source": file.name,
                        "page": page_num + 1
                    })

        if not self.chunks:
            raise ValueError("No readable text found in PDFs.")

        texts = [c["text"] for c in self.chunks]

        self.embeddings = self.embedder.encode(
            texts,
            normalize_embeddings=True
        )

        if self.embeddings is None:
            raise RuntimeError("Embedding generation failed")

    # --------------------------------
    # RETRIEVAL
    # --------------------------------
    def retrieve(self, query, top_k=4):

        query_embedding = self.embedder.encode(
            [query],
            normalize_embeddings=True
        )[0]

        scores = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(scores)[-top_k:][::-1]

        return [self.chunks[i] for i in top_indices]

    # --------------------------------
    # PROMPT BUILDER
    # --------------------------------
    def build_prompt(self, query, contexts):

        context_text = "\n\n".join(
            f"(Page {c['page']}): {c['text']}"
            for c in contexts
        )

        return f"""
You are an AI assistant answering questions from documents.

Context:
{context_text}

Question:
{query}

Answer clearly using only the context.
"""

    # --------------------------------
    # STREAM CHAT ANSWER
    # --------------------------------
    def stream_answer(self, query):

        contexts = self.retrieve(query)
        prompt = self.build_prompt(query, contexts)

        stream = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024,
            stream=True,
        )

        full_text = ""

        for chunk in stream:
            delta = chunk.choices[0].delta
            token = getattr(delta, "content", "") if delta else ""

            full_text += token
            yield token, contexts

        self.chat_history.append(f"User: {query}")
        self.chat_history.append(f"Assistant: {full_text}")

    # --------------------------------
    # FOLLOW-UP QUERY HANDLING
    # --------------------------------
    def reformulate_query(self, query):

        follow_words = ["explain more", "tell more", "why", "how"]

        if any(w in query.lower() for w in follow_words):

            for msg in reversed(self.chat_history):
                if msg.startswith("User:"):
                    last_question = msg.replace("User:", "").strip()
                    return f"{query} (regarding: {last_question})"

        return query

    # --------------------------------
    # STREAM SUMMARY (GROQ SAFE)
    # --------------------------------
    def stream_summary(self):

    # ✅ token-safe limit for llama3-8b-8192
        MAX_CHARS = 6000   # SAFE (~1500 tokens)

        if not self.chunks:
            yield "⚠️ No documents loaded.", False
            return

        collected_text = ""

    # ✅ build limited context
        for c in self.chunks:
            text = c.get("text", "").strip()

            if not text:
                continue

            if len(collected_text) + len(text) > MAX_CHARS:
                break

            collected_text += text + "\n"

        if not collected_text.strip():
            yield "⚠️ No readable text for summarization.", False
            return

        prompt = f"""
    Summarize the document clearly and concisely.

    Document:
    {collected_text}
    """

        try:
            stream = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=800,   # ✅ prevent overflow
                stream=True,
        )

            for chunk in stream:
                delta = chunk.choices[0].delta
                token = getattr(delta, "content", "") if delta else ""

                if token:
                    yield token, True

        except Exception as e:
            yield f"⚠️ Groq API Error: {str(e)}", False
