import os
import streamlit as st

from dotenv import load_dotenv
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
    )
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(
    page_title="RAG Document Retrieval System",
    page_icon="📚",
    layout="wide"
)

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("Gemini API Key not found.")
    st.stop()

@st.cache_resource(show_spinner=False)
def process_pdf(pdf_bytes: bytes, filename: str):
    """Process a PDF and cache its FAISS vector store."""

    os.makedirs("data", exist_ok=True)

    pdf_path = os.path.join("data", filename)

    with open(pdf_path, "wb") as file:
        file.write(pdf_bytes)

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    if not documents:
        raise ValueError("No pages were extracted from the PDF.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200
    )

    chunks = text_splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No readable text was extracted from the PDF.")

    embeddings =GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key= os.getenv("GEMINI_API_KEY")
    )

    vector_store = FAISS.from_documents(chunks, embeddings)

    return vector_store, len(documents), len(chunks)


@st.cache_resource
def get_gemini_model():
    return ChatGoogleGenerativeAI(
        model = "gemini-3.8-flash",
        google_api_key = os.getenv("GEMINI_API_KEY")
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.button("🗑️ Clear Chat"):
    st.session_state.message = []
    st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

st.title("📚 RAG-Based Document Retrieval System")
st.write("Upload a PDF and ask questions about its content.")


uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

if uploaded_file is not None:
    st.success(f"PDF Uploaded: {uploaded_file.name}")

    try:
        with st.spinner("Processing PDF and building search index..."):
            vector_store, page_count, chunk_count = process_pdf(
                uploaded_file.getvalue(),
                uploaded_file.name
            )

        st.write(f"Number of pages: {page_count}")
        st.write(f"Number of chunks: {chunk_count}")

        retriever = vector_store.as_retriever(
            search_kwargs={"k":3}
        )

        model = get_gemini_model()

    except Exception as e:
        st.error(f"Could not process the PDF: {e}")
        st.stop()

    st.divider()

    
    question = st.chat_input("Ask a question about your document")
    submitted = question is not None

    if submitted:

        if not question.strip():
            st.warning("Please enter a question.")

        else:
            with st.chat_message("user"):
                st.markdown(question)

            st.session_state.messages.append(
                {"role":"user", "content":question}
            )
        
            results = retriever.invoke(question)

            context = "\n\n".join(
                document.page_content
                for document in results
            )

            prompt = ChatPromptTemplate.from_template(
                """
    You are a helpful document assistant.

    Answer the user's question using only the provided context.

    If the answer cannot be found in the context, say:
    "I could not find the answer in the provided document."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
                )
            formatted_prompt = prompt.invoke(
                {
                    "context": context,
                    "question": question
                }
            )

            with st.spinner("Generating answer..."):

                try:
                    response = model.invoke(formatted_prompt)

                except Exception as e:
                    error_message = str(e)

                    if "503" in error_message or "UNAVAILABLE" in error_message:
                        st.warning("Gemini is temporarily unavailable due to high demand." 
                                   
                                   "Please wait a few minutes and try again.")
                        
                    elif "429" in error_message or "RESOURCE_EXHAUSTED" in error_message:
                        st.warning("Gemini API quota or rate limit reached."
                                   "Please check your quota and retry later.")
                        
                    else:
                        st.error("Unable to generate an answer. Please try again later.")

                    st.stop()

            content = response.content

            if isinstance(content, list):

                answer_parts = []

                for item in content:

                    if(isinstance(item, dict) and item.get("type") == "text"):

                        answer_parts.append(item.get("text", ""))

                answer = "".join(answer_parts)

            else:
                answer = str(content)

            with st.chat_message("assistant"):
                st.markdown(answer)

            st.session_state.messages.append(
                {"role":"assistant", "content":answer}
            )