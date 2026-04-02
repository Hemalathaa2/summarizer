import streamlit as st
from rag_engine import RAGEngine

st.set_page_config(page_title="PDF AI Assistant")
st.title("📚 Multi-PDF Summarizer")

# -------- SESSION --------
if "rag" not in st.session_state:
    st.session_state.rag = RAGEngine()

if "files_hash" not in st.session_state:
    st.session_state.files_hash = None

if "chat_input" not in st.session_state:
    st.session_state.chat_input = ""

# -------- UPLOAD --------
uploaded_files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    new_hash = tuple((f.name, f.size) for f in uploaded_files)

    if new_hash != st.session_state.files_hash:
        st.session_state.files_hash = new_hash
        st.session_state.chat_input = ""

        st.session_state.rag.clear()
        st.session_state.rag.load_pdfs(uploaded_files)

        st.success("✅ PDFs loaded successfully")

# -------- SUMMARY --------
if st.button("Generate Summary"):

    if not uploaded_files:
        st.warning("Upload PDFs first")
    else:
        with st.spinner("Generating summary..."):

            text = ""
            placeholder = st.empty()

            for token in st.session_state.rag.stream_summary():
                text += token
                placeholder.markdown(text + "▌")

            placeholder.markdown(text)

# -------- SELECT PDF --------
pdf_list = list(set([c["source"] for c in st.session_state.rag.chunks]))

selected_pdf = None
if pdf_list:
    selected_pdf = st.selectbox("📄 Select PDF", pdf_list)

# -------- CHAT UI --------
st.markdown("---")

for msg in st.session_state.rag.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.text_input("Ask a question", key="chat_input")

if st.button("Ask"):

    if not uploaded_files:
        st.warning("Upload PDFs first")

    elif not selected_pdf:
        st.warning("Select a PDF")

    elif query:

        with st.chat_message("user"):
            st.markdown(query)

        answer_text = ""

        with st.chat_message("assistant"):
            placeholder = st.empty()

            for token in st.session_state.rag.stream_answer(
                query,
                source_filter=selected_pdf
            ):
                answer_text += token
                placeholder.markdown(answer_text + "▌")

            placeholder.markdown(answer_text)

# -------- CLEAR CHAT --------
if st.button("🗑️ Clear Chat"):
    st.session_state.rag.chat_history = []
