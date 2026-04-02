import fitz
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
import streamlit as st

MODEL_NAME = "llama-3.1-8b-instant"


class RAGEngine:

    def __init__(self):
        self.client = Groq(api_key=st.secrets["GROQ_API_KEY"])

        self.embedder = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device="cpu"
        )

        self.chunks = []
        self.embeddings = None
        self.chat_history = []

    # -------- TEXT SPLIT --------
    def split_text(self, text, chunk_size=500, overlap=100):
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap

        return chunks

    # -------- LOAD PDFs --------
    def load_pdfs(self, files):
        self.chunks = []

        for file in files:
            file.seek(0)
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
            raise ValueError("No readable text found.")

    # -------- RETRIEVE --------
    def retrieve(self, query, source_filter=None, top_k=4):

        filtered_chunks = self.chunks

        if source_filter:
            filtered_chunks = [
                c for c in self.chunks if c["source"] == source_filter
            ]

        texts = [c["text"] for c in filtered_chunks]

        embeddings = self.embedder.encode(texts, normalize_embeddings=True)
        query_embedding = self.embedder.encode([query], normalize_embeddings=True)[0]

        scores = np.dot(embeddings, query_embedding)
        top_indices = np.argsort(scores)[-top_k:][::-1]

        return [filtered_chunks[i] for i in top_indices]

    # -------- PROMPT --------
    def build_prompt(self, query, contexts):

        context_text = "\n\n".join(
            f"(Page {c['page']}): {c['text']}"
            for c in contexts
        )

        return f"""
Answer ONLY from the given context.

If answer not present, say:
"Not mentioned in document"

Context:
{context_text}

Question:
{query}
"""

    # -------- Q&A --------
    def stream_answer(self, query, source_filter=None):

        contexts = self.retrieve(query, source_filter=source_filter)
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
            try:
                token = getattr(chunk.choices[0].delta, "content", "")
                if token:
                    full_text += token
                    yield token
            except:
                continue

        # Save chat
        self.chat_history.append({"role": "user", "content": query})
        self.chat_history.append({"role": "assistant", "content": full_text})

    # -------- SUMMARY --------
    def stream_summary(self):

        if not self.chunks:
            yield "⚠️ No documents loaded."
            return

        docs = {}

        for c in self.chunks:
            docs.setdefault(c["source"], []).append(c["text"])

        for filename, texts in docs.items():

            yield f"\n\n### 📄 {filename}\n\n"

            collected_text = ""

            for t in texts:
                if len(collected_text) > 3500:
                    break
                collected_text += t + "\n"

            prompt = f"""
Summarize in bullet points ONLY:

- Use "-"
- 5 to 8 points
- No paragraphs

TEXT:
{collected_text}
"""

            stream = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=600,
                stream=True,
            )

            full_text = ""

            for chunk in stream:
                token = getattr(chunk.choices[0].delta, "content", "")
                if token:
                    full_text += token

            lines = full_text.split("\n")
            clean = []

            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if not line.startswith("-"):
                    line = "- " + line
                clean.append(line)

            yield "\n".join(clean)

    # -------- CLEAR --------
    def clear(self):
        self.chunks = []
        self.chat_history = []
