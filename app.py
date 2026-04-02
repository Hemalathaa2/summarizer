import streamlit as st
from rag_engine import RAGEngine

st.set_page_config(page_title="PDF AI Assistant")
st.title("📚 Multi-PDF Summarizer")

# ---------------- SESSION STATE ----------------
if "rag" not in st.session_state:
    st.session_state.rag = RAGEngine()

if "files_hash" not in st.session_state:
    st.session_state.files_hash = None

if "summary" not in st.session_state:
    st.session_state.summary = None

if "generate_clicked" not in st.session_state:
    st.session_state.generate_clicked = False


# ---------------- FILE UPLOAD ----------------
uploaded_files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

# ---------------- RESET WHEN FILES CHANGE ----------------
if uploaded_files:
    new_hash = tuple((f.name, f.size) for f in uploaded_files)

    if new_hash != st.session_state.files_hash:
        st.session_state.files_hash = new_hash
        st.session_state.summary = None
        st.session_state.generate_clicked = False

        # Clear old memory
        st.session_state.rag.clear()

        # Load PDFs
        st.session_state.rag.load_pdfs(uploaded_files)

        st.success("✅ PDFs loaded successfully")


# ---------------- GENERATE SUMMARY ----------------
if st.button("Generate Summary"):
    if uploaded_files:
        st.session_state.generate_clicked = True
        st.session_state.summary = None
    else:
        st.warning("Upload PDFs first")


# ---------------- RUN SUMMARY ONLY ONCE ----------------
if st.session_state.generate_clicked:

    with st.spinner("Generating summary..."):

        summary_text = ""
        placeholder = st.empty()

        for token, _ in st.session_state.rag.stream_summary():
            summary_text += token
            placeholder.markdown(summary_text + "▌")

        placeholder.markdown(summary_text)

        st.session_state.summary = summary_text
        st.session_state.generate_clicked = False


# ---------------- SHOW SAVED SUMMARY ----------------
if st.session_state.summary:
    st.markdown("### 📌 Summary")
    st.markdown(st.session_state.summary)


# ---------------- Q&A SECTION ----------------
st.markdown("---")
st.subheader("💬 Ask Questions from PDFs")

query = st.text_input("Enter your question")

if st.button("Ask"):
    if not uploaded_files:
        st.warning("Please upload PDFs first")
    elif query:
        answer_text = ""
        answer_placeholder = st.empty()
        contexts_used = []

        for token, contexts in st.session_state.rag.stream_answer(query):
            answer_text += token
            contexts_used = contexts
            answer_placeholder.markdown(answer_text + "▌")

        answer_placeholder.markdown(answer_text)

        # Show sources
        with st.expander("📄 Sources"):
            for c in contexts_used:
                st.write(f"{c['source']} - Page {c['page']}")
    else:
        st.warning("Enter a question")


# ---------------- CHAT HISTORY (OPTIONAL) ----------------
if st.session_state.rag.chat_history:
    st.markdown("### 🧠 Chat History")
    for msg in st.session_state.rag.chat_history:
        st.write(msg)
