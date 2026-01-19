import streamlit as st
import os
from dotenv import load_dotenv
import fitz  # PyMuPDF
from PIL import Image
import pytesseract

# --- LangChain Imports ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

load_dotenv()

# --- OCR Configuration ---
# Your Tesseract Path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

st.set_page_config(page_title="PSNS: Study Notes", page_icon="📚", layout="wide")
st.title("📚 PSNS: Personal Study Notes Searcher")

# API Key Check
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("⚠️ API Key not found! Please check your .env file.")
    st.stop()

# Session State
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "uploaded_file_path" not in st.session_state:
    st.session_state.uploaded_file_path = None
if "file_type" not in st.session_state:
    st.session_state.file_type = None

# --- 1. Upload Section ---
uploaded_file = st.file_uploader("Upload Lecture Slides (PDF) or Note Images", type=['pdf', 'png', 'jpg', 'jpeg'])

if uploaded_file:
    # Check/Create temp folder
    if not os.path.exists("temp_files"):
        os.makedirs("temp_files")
    
    # Save file
    file_path = os.path.join("temp_files", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.session_state.uploaded_file_path = file_path
    st.session_state.file_type = uploaded_file.type

    if st.button("🧠 Start Processing"):
        with st.spinner("Processing file... (Images might take some time)"):
            try:
                documents = []
                
                # A. PDF Processing
                if uploaded_file.type == "application/pdf":
                    loader = PyPDFLoader(file_path)
                    documents = loader.load()
                    st.info("📄 Processing in PDF mode...")

                # B. Image Processing (OCR)
                else:
                    st.info("📷 Image detected. Extracting text using OCR...")
                    image = Image.open(file_path)
                    extracted_text = pytesseract.image_to_string(image)
                    
                    if not extracted_text.strip():
                        st.warning("⚠️ No text found in the image! Is the image clear?")
                    else:
                        doc = Document(page_content=extracted_text, metadata={"page": 0, "source": uploaded_file.name})
                        documents = [doc]

                # Chunking & Embedding
                if documents:
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    chunks = text_splitter.split_documents(documents)
                    
                    embeddings = OpenAIEmbeddings()
                    vector_store = FAISS.from_documents(chunks, embeddings)
                    
                    st.session_state.vector_store = vector_store
                    st.success(f"✅ Success! Study Brain created. Ready to answer questions.")
                else:
                    st.error("Could not process any text from the file.")

            except Exception as e:
                st.error(f"Error details: {e}")
                st.info("Tip: Is Tesseract installed correctly? Check the file path in code.")

st.write("---")
# --- 2. Q&A Section ---
user_question = st.text_input("Ask a question from your notes:")

if user_question and st.session_state.vector_store:
    
    # --- CHANGE IS HERE (k=10 added) ---
    # আগে এটি শুধু 4টি লাইন পড়ত, এখন 10টি লাইন পড়ে উত্তর দিবে
    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 10})
    
    # --- BETTER PROMPT ---
    template = """You are an advanced university assistant.
    Answer the question based ONLY on the following context.
    If the user asks for a summary, provide a comprehensive and detailed summary using all the available context.
    
    Context:
    {context}
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    with st.spinner("Searching for answer..."):
        try:
            # Retrieve Docs
            relevant_docs = retriever.invoke(user_question)
            context_text = "\n\n".join([d.page_content for d in relevant_docs])
            
            # Generate Answer
            formatted_prompt = prompt.invoke({"context": context_text, "question": user_question})
            response = llm.invoke(formatted_prompt)
            
            st.success("🤖 AI Answer:")
            st.write(response.content)
            
            # Show References
            st.markdown("---")
            st.subheader("📌 References & Source Slides:")
            
            unique_pages = set()
            for doc in relevant_docs:
                page_num = doc.metadata.get('page', 0)
                unique_pages.add(page_num)
            
            # Show Slides/Images
            if st.session_state.uploaded_file_path:
                
                # If original file is PDF
                if "pdf" in st.session_state.file_type:
                    pdf_doc = fitz.open(st.session_state.uploaded_file_path)
                    for page_num in sorted(unique_pages):
                        with st.expander(f"📄 Page {page_num + 1} (Click to View Slide)", expanded=False):
                            st.info(f"Source: Page {page_num + 1}")
                            page = pdf_doc.load_page(page_num)
                            pix = page.get_pixmap(dpi=150)
                            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                            st.image(img, caption=f"Page {page_num + 1}", use_container_width=True)
                
                # If original file is Image
                else:
                    with st.expander("📷 View Original Image", expanded=False):
                        img = Image.open(st.session_state.uploaded_file_path)
                        st.image(img, caption="Uploaded Image", use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}")

elif user_question and not st.session_state.vector_store:
    st.warning("⚠️ Please upload and process a file first!")