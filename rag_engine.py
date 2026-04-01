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
            file.seek(0)  # ✅ FIX duplicate reading issue
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

        # -------- Document embeddings (for similarity check)
        self.doc_embeddings = {}

        for chunk in self.chunks:
            src = chunk["source"]
            self.doc_embeddings.setdefault(src, []).append(chunk["text"])

        for src, texts in self.doc_embeddings.items():
            emb = self.embedder.encode(texts, normalize_embeddings=True)
            self.doc_embeddings[src] = np.mean(emb, axis=0)

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
    # TOPIC SIMILARITY CHECK
    # --------------------------------
    def documents_are_similar(self, threshold=0.75):

        docs = list(self.doc_embeddings.keys())

        if len(docs) <= 1:
            return False

        vectors = [self.doc_embeddings[d] for d in docs]

        similarities = []

        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                sim = np.dot(vectors[i], vectors[j])
                similarities.append(sim)

        return np.mean(similarities) >= threshold

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
    # STREAM SUMMARY
    # --------------------------------
    def stream_summary(self):

        if not self.chunks:
            yield "⚠️ No documents loaded.", False
            return

        similar = self.documents_are_similar()

        # ==============================
        # CASE 1 — COMBINED SUMMARY
        # ==============================
        if similar:

            yield "📚 **Combined Summary (Similar Topics Detected)**\n\n", True

            MAX_CHARS = 3500
            collected_text = ""

            for c in self.chunks:
                text = c["text"]
                if len(collected_text) + len(text) > MAX_CHARS:
                    break
                collected_text += text + "\n"

            prompt = f"""
You are an expert document summarizer.

Create a concise summary (5–8 bullet points).

STRICT RULES:
- Do NOT copy sentences.
- Do NOT repeat document text.
- Compress information.
- Focus only on key ideas.

DOCUMENTS:
{collected_text}
"""

            stream = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=600,
                stream=True,
            )

            for chunk in stream:
                token = getattr(chunk.choices[0].delta, "content", "")
                if token:
                    yield token, True

        # ==============================
        # CASE 2 — SEPARATE SUMMARIES
        # ==============================
        else:

            yield "📄 **Separate Summaries (Different Topics Detected)**\n", True

            docs = {}
            for c in self.chunks:
                docs.setdefault(c["source"], []).append(c["text"])

            for filename, texts in docs.items():

                yield f"\n\n### {filename}\n\n", True

                MAX_CHARS = 3500
                collected_text = ""

                for t in texts:
                    if len(collected_text) + len(t) > MAX_CHARS:
                        break
                    collected_text += t + "\n"

                prompt = f"""
You are an expert document summarizer.

Summarize this document in 5–8 concise bullet points.

Rules:
- Do NOT copy text.
- Do NOT rewrite paragraphs.
- Extract only important concepts.

DOCUMENT:
{collected_text}
"""

                stream = self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=600,
                    stream=True,
                )

                for chunk in stream:
                    token = getattr(chunk.choices[0].delta, "content", "")
                    if token:
                        yield token, True


"""Reset stored PDFs and embeddings"""
      def clear(self):
          self.chunks = []
          self.embeddings = []
        
