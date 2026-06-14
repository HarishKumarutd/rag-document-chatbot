import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import tempfile, os

st.set_page_config(page_title="Business Document Analyst", page_icon="📄")
st.title("📄 Business Document Analyst")
st.caption("Upload any business PDF and ask questions about it")

api_key = st.text_input("Paste your Gemini API key here", type="password")
uploaded_file = st.file_uploader("Upload a PDF (annual report, strategy doc, etc)", type="pdf")
question = st.text_input("Ask a question about the document")

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
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        db = FAISS.from_documents(chunks, embeddings)

        # Find the most relevant chunks for the question
        relevant_docs = db.similarity_search(question, k=4)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Ask Gemini with the context
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
        prompt = f"""You are a business analyst. Use ONLY the document context below to answer the question.
If the answer is not in the context, say "I couldn't find that in the document."

Context:
{context}

Question: {question}

Answer:"""
        
        response = llm.invoke(prompt)
        st.success(response.content)
        os.unlink(tmp_path)
