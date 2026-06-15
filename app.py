import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import tempfile, os

st.set_page_config(page_title="Business Document Analyst", page_icon="📄")
st.title("📄 RAG (Retrieval Augmented Generation) based Business Document Analyst")
st.caption("Upload any business PDF and ask questions about it")

api_key = st.text_input("Paste your Gemini API key here", type="password")
uploaded_file = st.file_uploader("Upload a PDF (annual report, strategy doc, etc)", type="pdf")
question = st.text_input("Ask a question about the document")

st.caption("Or try one of these:")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("💰 Revenue breakdown"):
        question = "What was the revenue breakdown by business segment in 2024?"
with col2:
    if st.button("⚠️ Key risks"):
        question = "What are the top strategic risks mentioned in this report?"
with col3:
    if st.button("🤖 AI strategy"):
        question = "What is the company's AI and generative AI strategy?"

if st.button("Get Answer") and uploaded_file and question and api_key:
    with st.spinner("Reading document and thinking..."):
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(uploaded_file.read())
            tmp_path = f.name

        # Load and split the PDF
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        # Create embeddings and search index
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-004", google_api_key=api_key)
        try:
            db = FAISS.from_documents(chunks, embeddings)
        except Exception as e:
        st.error(f"Embedding error: {str(e)}")
        st.stop()
        
        # Find the most relevant chunks for the question
        relevant_docs = db.similarity_search(question, k=4)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Show source pages
        page_numbers = [str(doc.metadata.get('page', 'unknown') + 1) for doc in relevant_docs]
        st.caption(f"📄 Answer sourced from pages: {', '.join(set(page_numbers))}")

        # Ask Gemini with the context
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
        prompt = f"""You are a senior business analyst. Use ONLY the document context below to answer the question.
        If the answer is not in the context, say "I couldn't find that in the document."

Context:
{context}

Question: {question}

Answer in a structured, professional way using bullet points where appropriate:"""
        
        response = llm.invoke(prompt)
        st.success(response.content)
        st.download_button("📥 Download this answer", response.content, file_name="analysis.txt")
        os.unlink(tmp_path)
