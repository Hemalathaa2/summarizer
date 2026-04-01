import streamlit as st
from rag_engine import RAGEngine

st.set_page_config(page_title="PDF AI Assistant")
st.title("📚 Multi-PDF AI Knowledge Assistant")

# ---------------- SESSION STATE ----------------
if "rag" not in st.session_state:
    st.session_state.rag = RAGEngine()

if "files_hash" not in st.session_state:
    st.session_state.files_hash = None

if "summary" not in st.session_state:
    st.session_state.summary = None

# ---------------- FILE UPLOAD ----------------
uploaded_files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

# ---------------- RESET WHEN FILES CHANGE ----------------
if uploaded_files:
    new_hash = tuple((f.name, f.size) for f in uploaded_files)

    # Detect NEW upload set
    if new_hash != st.session_state.files_hash:
        st.session_state.files_hash = new_hash
        st.session_state.summary = None

        # Reset RAG memory
        st.session_state.rag.clear()

        # Load PDFs
        for file in uploaded_files:
            st.session_state.rag.add_pdf(file)

        st.success("✅ PDFs loaded successfully")

# ---------------- GENERATE SUMMARY ----------------
if st.button("Generate Summary"):
    if uploaded_files:
        with st.spinner("Generating summary..."):
            st.session_state.summary = (
                st.session_state.rag.generate_summary()
            )
    else:
        st.warning("Upload PDFs first")

# ---------------- DISPLAY (ONLY HERE) ----------------
if st.session_state.summary:
    st.subheader("📄 Summary")
    st.write(st.session_state.summary)
