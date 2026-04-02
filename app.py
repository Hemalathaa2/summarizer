import streamlit as st
from rag_engine import RAGEngine

st.set_page_config(page_title="PDF AI Assistant")
st.title("📚 Multi-PDF Summarizer")

# -------- SESSION STATE --------
if "rag" not in st.session_state:
    st.session_state.rag = RAGEngine()

if "files_hash" not in st.session_state:
    st.session_state.files_hash = None

if "summary" not in st.session_state:
    st.session_state.summary = None

if "chat_input" not in st.session_state:
    st.session_state.chat_input = ""

# -------- FILE UPLOAD --------
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
        st.session_state.chat_input = ""

        # Reset everything
        st.session_state.rag.clear()

        # Load PDFs
        st.session_state.rag.load_pdfs(uploaded_files)

        st.success("✅ PDFs loaded successfully")

# -------- GENERATE SUMMARY --------
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

            st.session_state.summary = summary_text

# -------- PDF SELECTOR --------
pdf_list = list(set([c["source"] for c in st.session_state.rag.chunks]))

selected_pdf = None
if pdf_list:
    selected_pdf = st.selectbox("📄 Select PDF for Q&A", pdf_list)

# -------- CHAT SECTION --------
st.markdown("---")
st.subheader("💬 Q & A")

# Show chat history (ChatGPT style)
for msg in st.session_state.rag.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box
query = st.text_input("Enter your question", key="chat_input")

# -------- ASK BUTTON --------
if st.button("Ask"):

    if not uploaded_files:
        st.warning("Upload PDFs first")

    elif not selected_pdf:
        st.warning("Select a PDF")

    elif query:

        # Show user message
        with st.chat_message("user"):
            st.markdown(query)

        answer_text = ""

        # Show assistant streaming
        with st.chat_message("assistant"):
            placeholder = st.empty()

            

            for token, contexts in st.session_state.rag.stream_answer(
                query,
                source_filter=selected_pdf
            ):
                answer_text += token
                
                placeholder.markdown(answer_text + "▌")

            placeholder.markdown(answer_text)

        

    else:
        st.warning("Enter a question")

# -------- CLEAR CHAT --------
if st.button("🗑️ Clear Chat"):
    st.session_state.rag.chat_history = []
