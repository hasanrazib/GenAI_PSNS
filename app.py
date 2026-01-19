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
from langchain_core.documents import Document

load_dotenv()

# --- OCR Configuration ---
# Your Tesseract Path (Update if needed)
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
# আগে ছিল শুধু একটা পাথ, এখন হবে পাথ-এর ডিকশনারি {filename: filepath}
if "file_paths" not in st.session_state:
    st.session_state.file_paths = {}

# --- 1. Upload Section (Multiple Files Support) ---
# accept_multiple_files=True যোগ করা হয়েছে
uploaded_files = st.file_uploader("Upload Lecture Slides (PDF) or Note Images", 
                                  type=['pdf', 'png', 'jpg', 'jpeg'], 
                                  accept_multiple_files=True)

if uploaded_files:
    if st.button("🧠 Start Processing All Files"):
        
        # Check/Create temp folder
        if not os.path.exists("temp_files"):
            os.makedirs("temp_files")
            
        all_documents = []
        st.session_state.file_paths = {} # Reset paths
        
        with st.spinner("Processing all files... This may take a moment."):
            try:
                # Loop through each uploaded file
                for uploaded_file in uploaded_files:
                    
                    # Save file locally
                    file_path = os.path.join("temp_files", uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Store path in session state (FileName -> FilePath)
                    st.session_state.file_paths[uploaded_file.name] = file_path
                    
                    # --- A. PDF Processing ---
                    if uploaded_file.type == "application/pdf":
                        loader = PyPDFLoader(file_path)
                        docs = loader.load()
                        # Add source metadata just to be safe
                        for doc in docs:
                            doc.metadata['source'] = uploaded_file.name
                        all_documents.extend(docs)
                        
                    # --- B. Image Processing (OCR) ---
                    else:
                        image = Image.open(file_path)
                        extracted_text = pytesseract.image_to_string(image)
                        
                        if extracted_text.strip():
                            # Create a document object manually
                            doc = Document(page_content=extracted_text, 
                                           metadata={"page": 0, "source": uploaded_file.name})
                            all_documents.append(doc)
                
                # --- Chunking & Embedding (Common for all files) ---
                if all_documents:
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    chunks = text_splitter.split_documents(all_documents)
                    
                    embeddings = OpenAIEmbeddings()
                    vector_store = FAISS.from_documents(chunks, embeddings)
                    
                    st.session_state.vector_store = vector_store
                    st.success(f"✅ Success! Processed {len(uploaded_files)} files. Brain is ready!")
                else:
                    st.error("Could not extract text from any of the uploaded files.")

            except Exception as e:
                st.error(f"Error during processing: {e}")

st.write("---")

# --- 2. Q&A Section ---
user_question = st.text_input("Ask a question from your notes:")

if user_question and st.session_state.vector_store:
    
    # k=10 for better summary
    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 10})
    
    template = """You are an advanced university assistant.
    Answer the question based ONLY on the following context.
    If the user asks for a summary, provide a comprehensive and detailed summary.
    
    Context:
    {context}
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    with st.spinner("Searching across all files..."):
        try:
            relevant_docs = retriever.invoke(user_question)
            context_text = "\n\n".join([d.page_content for d in relevant_docs])
            
            formatted_prompt = prompt.invoke({"context": context_text, "question": user_question})
            response = llm.invoke(formatted_prompt)
            
            st.success("🤖 AI Answer:")
            st.write(response.content)
            
            # --- Source Verification (Multi-file Support) ---
            st.markdown("---")
            st.subheader("📌 References & Source Slides:")
            
            # We need to group pages by source file
            # Example: {'Lecture1.pdf': {1, 5}, 'Note.jpg': {0}}
            sources_map = {}
            for doc in relevant_docs:
                source_name = doc.metadata.get('source')
                page_num = doc.metadata.get('page', 0)
                
                if source_name not in sources_map:
                    sources_map[source_name] = set()
                sources_map[source_name].add(page_num)
            
            # Display images/slides based on source
            for source_name, pages in sources_map.items():
                
                # Get local path from session state
                file_path = st.session_state.file_paths.get(source_name)
                
                if file_path:
                    st.markdown(f"**📂 Source File: `{source_name}`**")
                    
                    # If it's a PDF
                    if source_name.lower().endswith('.pdf'):
                        pdf_doc = fitz.open(file_path)
                        cols = st.columns(len(pages))
                        for idx, page_num in enumerate(sorted(pages)):
                            with st.expander(f"📄 Page {page_num + 1}", expanded=False):
                                page = pdf_doc.load_page(page_num)
                                pix = page.get_pixmap(dpi=150)
                                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                                st.image(img, caption=f"Page {page_num + 1}", use_container_width=True)
                    
                    # If it's an Image
                    else:
                        with st.expander("📷 Original Image", expanded=False):
                            img = Image.open(file_path)
                            st.image(img, caption="Source Note", use_container_width=True)

        except Exception as e:
            st.error(f"Error generating answer: {e}")

elif user_question and not st.session_state.vector_store:
    st.warning("⚠️ Please upload and process files first!")