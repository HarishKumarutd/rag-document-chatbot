import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile, os

st.set_page_config(page_title="Business Document Analyst", page_icon="📄")
st.title("📄 Business Document Analyst")
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
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                f.write(uploaded_file.read())
                tmp_path = f.name

            # Load and split the PDF
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = splitter.split_documents(docs)

            # Find relevant chunks using simple keyword matching
            question_words = set(question.lower().split())
            scored_chunks = []
            for chunk in chunks:
                chunk_words = set(chunk.page_content.lower().split())
                score = len(question_words & chunk_words)
                scored_chunks.append((score, chunk))

            # Sort by relevance score and take top 6
            scored_chunks.sort(key=lambda x: x[0], reverse=True)
            top_chunks = [c[1] for c in scored_chunks[:6]]
            context = "\n\n".join([chunk.page_content for chunk in top_chunks])

            # Show source pages
            page_numbers = [str(chunk.metadata.get('page', 0) + 1) for chunk in top_chunks]
            st.caption(f"📄 Answer sourced from pages: {', '.join(set(page_numbers))}")

            # Ask Gemini
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

        except Exception as e:
            st.error(f"Error: {str(e)}")
