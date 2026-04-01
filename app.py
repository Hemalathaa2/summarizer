import streamlit as st
from rag_engine import RAGEngine

st.set_page_config(page_title="PDF AI Assistant")

st.title("📚 Multi-PDF AI Knowledge Assistant")

# -------------------------
# SESSION STATE INIT
# -------------------------
if "rag" not in st.session_state:
    st.session_state.rag = RAGEngine()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_uploaded_names" not in st.session_state:
    st.session_state.last_uploaded_names = []

# -------------------------
# FILE UPLOADER
# -------------------------
files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

# -------------------------
# ✅ DETECT NEW UPLOADS
# -------------------------
if files:

    current_names = [f.name for f in files]

    # If uploaded files changed → RESET EVERYTHING
    if current_names != st.session_state.last_uploaded_names:

        # reset RAG completely
        st.session_state.rag = RAGEngine()

        # clear old summaries/chat
        st.session_state.messages = []

        # remember uploaded files
        st.session_state.last_uploaded_names = current_names

        with st.spinner("Indexing PDFs..."):
            st.session_state.rag.load_pdfs(files)

        st.success("PDFs Ready!")

# -------------------------
# SUMMARY BUTTON
# -------------------------
if files:

    if st.button("🧠 Generate Document Summary"):

        # ✅ clear previous summaries automatically
        st.session_state.messages = []

        with st.chat_message("assistant"):

            placeholder = st.empty()
            summary_text = ""

            for token, _ in st.session_state.rag.stream_summary():
                summary_text += token
                placeholder.markdown(summary_text + "▌")

            placeholder.markdown(summary_text)

        st.session_state.messages.append(
            {"role": "assistant", "content": summary_text}
        )

# -------------------------
# DISPLAY CHAT HISTORY
# -------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------
# CHAT INPUT
# -------------------------
query = st.chat_input("Ask something about PDFs...")

if query:

    rag = st.session_state.rag

    st.session_state.messages.append(
        {"role": "user", "content": query}
    )

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):

        placeholder = st.empty()
        full_response = ""

        for token, contexts in rag.stream_answer(query):
            full_response += token
            placeholder.markdown(full_response + "▌")

        citation_text = "\n\n---\n**Sources:**\n"
        for c in contexts:
            citation_text += f"- 📄 {c['source']} (Page {c['page']})\n"

        placeholder.markdown(full_response + citation_text)

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )
