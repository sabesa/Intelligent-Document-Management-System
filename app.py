import streamlit as st
from rag_pipeline import get_rag_pipeline
from summarizer import summarize_document
from document_loader import ingest_document
import tempfile
import os

st.set_page_config(
    page_title="Project Polaris - AI Knowledge Assistant",
    page_icon="🧠",
    layout="wide"
)

# Sidebar
st.sidebar.title("⚙️ System Control")
st.sidebar.info("Manage document ingestion, search, and summarization")

page = st.sidebar.radio("Choose a Function", ["📥 Ingest Documents", "🔍 Query Knowledge Base", "🧾 Summarize Document"])

# --- 1️⃣ Document Ingestion ---
if page == "📥 Ingest Documents":
    st.title("📥 Document Ingestion")
    st.write("Upload your PDF, DOCX, or TXT file for processing and embedding into the vector store.")

    uploaded_file = st.file_uploader("Upload Document", type=["pdf", "docx", "txt"])

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        st.info("Processing document... please wait ⏳")
        try:
            ingest_document(tmp_path)
            st.success(f"✅ {uploaded_file.name} successfully processed and stored in Qdrant.")
        except Exception as e:
            st.error(f"❌ Failed to ingest document: {e}")
        finally:
            os.remove(tmp_path)

# --- 2️⃣ Query Knowledge Base ---
elif page == "🔍 Query Knowledge Base":
    st.title("🔍 Ask Questions from Your Knowledge Base")
    st.write("Enter a question related to any previously uploaded document.")

    query = st.text_input("Enter your question:")
    if st.button("Get Answer"):
        if not query.strip():
            st.warning("Please enter a valid question.")
        else:
            try:
                rag = get_rag_pipeline()
                result = rag({"query": query})
                st.subheader("💬 AI Answer")
                st.write(result["result"])

                with st.expander("📚 Source Chunks"):
                    for doc in result["source_documents"]:
                        st.markdown(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
                        st.write(doc.page_content[:500] + "...")
                        st.divider()

            except Exception as e:
                st.error(f"Error during query: {e}")

# --- 3️⃣ Summarization ---
elif page == "🧾 Summarize Document":
    st.title("🧾 Document Summarization")
    st.write("Upload a long document to generate a concise executive summary.")

    uploaded_file = st.file_uploader("Upload Document for Summarization", type=["pdf", "docx", "txt"])

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        if st.button("Generate Summary"):
            st.info("Generating summary... please wait ⏳")
            try:
                summary = summarize_document(tmp_path)
                st.subheader("📋 Summary Output")
                st.write(summary)
                st.success("✅ Summarization complete.")
            except Exception as e:
                st.error(f"❌ Error during summarization: {e}")
            finally:
                os.remove(tmp_path)
