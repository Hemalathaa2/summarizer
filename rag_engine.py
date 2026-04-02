import fitz
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
import streamlit as st

MODEL_NAME = "llama-3.1-8b-instant"


class RAGEngine:

    def __init__(self):
        self.client = Groq(
            api_key=st.secrets["GROQ_API_KEY"]
        )

        self.embedder = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device="cpu"
        )

        self.chunks = []
        self.embeddings = None
        self.doc_embeddings = {}
        self.chat_history = []

    # -------- TEXT SPLITTER --------
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
            raise ValueError("No readable text found in PDFs.")

        texts = [c["text"] for c in self.chunks]

        self.embeddings = self.embedder.encode(
            texts,
            normalize_embeddings=True
        )

        # -------- DOCUMENT EMBEDDINGS --------
        self.doc_embeddings = {}

        for chunk in self.chunks:
            src = chunk["source"]
            self.doc_embeddings.setdefault(src, []).append(chunk["text"])

        for src, texts in self.doc_embeddings.items():
            emb = self.embedder.encode(texts, normalize_embeddings=True)
            self.doc_embeddings[src] = np.mean(emb, axis=0)

    # -------- RETRIEVE --------
    def retrieve(self, query, top_k=4):

        query_embedding = self.embedder.encode(
            [query],
            normalize_embeddings=True
        )[0]

        scores = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(scores)[-top_k:][::-1]

        return [self.chunks[i] for i in top_indices]

    # -------- PROMPT --------
    def build_prompt(self, query, contexts):

        context_text = "\n\n".join(
            f"(Page {c['page']}): {c['text']}"
            for c in contexts
        )

        return f"""
Answer ONLY from the given context.

STRICT RULES:
- Do NOT assume or infer
- If answer not present, say: "Not mentioned in document"

Context:
{context_text}

Question:
{query}
"""

    # -------- Q&A --------
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
            try:
                delta = chunk.choices[0].delta
                token = getattr(delta, "content", "") if delta else ""
        
                if token:
                    full_text += token
                    yield token, contexts
        
            except Exception:
                continue

        self.chat_history.append(f"User: {query}")
        self.chat_history.append(f"Assistant: {full_text}")

    # -------- SUMMARY --------
    def stream_summary(self):

        if not self.chunks:
            yield "⚠️ No documents loaded.", False
            return

        docs = {}

        # Group chunks by file
        for c in self.chunks:
            docs.setdefault(c["source"], []).append(c["text"])

        # -------- PER FILE SUMMARY --------
        for filename, texts in docs.items():

            yield f"\n\n### 📄 {filename}\n\n", True

            MAX_CHARS = 3500
            collected_text = ""

            for t in texts:
                if len(collected_text) + len(t) > MAX_CHARS:
                    break
                collected_text += t + "\n"

            prompt = f"""
Generate summary STRICTLY like this:

- Point 1
- Point 2
- Point 3

RULES:
- MUST use "-"
- NO paragraphs
- 5 to 8 points
- Max 2 lines each
- Do NOT copy sentences

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
                    yield token, True

            # -------- FORCE BULLET FORMAT --------
            if "-" not in full_text:
                lines = full_text.split(". ")
                fixed = "\n".join(
                    [f"- {line.strip()}" for line in lines if line.strip()]
                )
                yield "\n" + fixed, True

    # -------- CLEAR --------
    def clear(self):
        self.chunks = []
        self.embeddings = None
        self.doc_embeddings = {}
        self.chat_history = []
