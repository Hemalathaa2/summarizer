import streamlit as st
from rag_engine import RAGEngine

st.set_page_config(page_title="PDF AI Assistant")
st.title("📚 Multi-PDF Summarizer")

# -------- SESSION --------
if "rag" not in st.session_state:
    st.session_state.rag = RAGEngine()

if "files_hash" not in st.session_state:
    st.session_state.files_hash = None

if "summary" not in st.session_state:
    st.session_state.summary = None

# -------- UPLOAD --------
uploaded_files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

# -------- LOAD PDFs --------
if uploaded_files:
    new_hash = tuple((f.name, f.size) for f in uploaded_files)

    if new_hash != st.session_state.files_hash:
        st.session_state.files_hash = new_hash
        st.session_state.summary = None

        st.session_state.rag.clear()
        st.session_state.rag.load_pdfs(uploaded_files)

        st.success("✅ PDFs loaded successfully")

# -------- SUMMARY --------
if st.button("Generate Summary"):

    if not uploaded_files:
        st.warning("Upload PDFs first")
    else:
        with st.spinner("Generating summary..."):

            summary_text = ""
            placeholder = st.empty()

            for token, _ in st.session_state.rag.stream_summary():
                summary_text += token
                placeholder.markdown(summary_text + "▌")

            placeholder.markdown(summary_text)

            # Store but DO NOT reprint
            st.session_state.summary = summary_text

# -------- Q&A --------
st.markdown("---")
st.subheader("💬 Ask Questions")

query = st.text_input("Enter your question")

if st.button("Ask"):

    if not uploaded_files:
        st.warning("Upload PDFs first")

    elif query:
        answer_text = ""
        placeholder = st.empty()
        contexts_used = []

        for token, contexts in st.session_state.rag.stream_answer(query):
            answer_text += token
            contexts_used = contexts
            placeholder.markdown(answer_text + "▌")

        placeholder.markdown(answer_text)

        with st.expander("📄 Sources"):
            for c in contexts_used:
                st.write(f"{c['source']} - Page {c['page']}")

    else:
        st.warning("Enter a question")

# -------- CHAT HISTORY --------
if st.session_state.rag.chat_history:
    st.markdown("### 🧠 Chat History")
    for msg in st.session_state.rag.chat_history:
        st.write(msg)
