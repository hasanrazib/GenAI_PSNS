import streamlit as st
import os
from dotenv import load_dotenv
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import platform

# --- LangChain Imports ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

load_dotenv()

# ==========================================
# ⚙️ CONFIGURATION FOR PROFESSOR
# ==========================================
# Set this to True to enable Local Llama 3 (Requires Ollama running)
# Set this to False to use OpenAI GPT-4o (Default, Stable)
ENABLE_LOCAL_MODE = False 
# ==========================================

# --- OCR Configuration ---
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else:
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

st.set_page_config(page_title="PSNS: Study Notes", page_icon="📚", layout="wide")
st.title("📚 PSNS: Personal Study Notes Searcher")

# Show current mode
if ENABLE_LOCAL_MODE:
    st.markdown("### 🟢 Mode: **Local Llama 3 (Privacy Focused)**")
else:
    st.markdown("### 🔵 Mode: **OpenAI GPT-4o (High Speed)**")

# API Key Check
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("⚠️ API Key not found! Please check your .env file.")
    st.stop()

# --- Session State Initialization ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "file_paths" not in st.session_state:
    st.session_state.file_paths = {}
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 1. Upload Section (FIXED VIEW) ---
st.markdown("#### 📂 Upload Documents")
uploaded_files = st.file_uploader("Upload Lecture Slides (PDF) or Note Images", 
                                  type=['pdf', 'png', 'jpg', 'jpeg'], 
                                  accept_multiple_files=True)

if uploaded_files:
    if st.button("🧠 Start Processing All Files"):
        
        if not os.path.exists("temp_files"):
            os.makedirs("temp_files")
            
        all_documents = []
        st.session_state.file_paths = {} 
        
        with st.spinner("Processing files..."):
            try:
                for uploaded_file in uploaded_files:
                    # Save file locally
                    file_path = os.path.join("temp_files", uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    st.session_state.file_paths[uploaded_file.name] = file_path
                    
                    # --- A. PDF Processing ---
                    if uploaded_file.type == "application/pdf":
                        loader = PyPDFLoader(file_path)
                        docs = loader.load()
                        for doc in docs:
                            doc.metadata['source'] = uploaded_file.name
                        all_documents.extend(docs)
                        
                    # --- B. Image Processing (OCR) ---
                    else:
                        image = Image.open(file_path)
                        extracted_text = pytesseract.image_to_string(image)
                        
                        if extracted_text.strip():
                            doc = Document(page_content=extracted_text, 
                                           metadata={"page": 0, "source": uploaded_file.name})
                            all_documents.append(doc)
                
                # --- Chunking & Embedding ---
                if all_documents:
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
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

# --- 2. Q&A Section (Modern Chat UI) ---

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input Field
if user_question := st.chat_input("💬 Ask a question from your notes..."):
    
    # 1. User Message Show
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # 2. Check if DB exists
    if not st.session_state.vector_store:
        st.warning("⚠️ Please upload and process files first!")
        st.stop()

    # 3. Model Setup
    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 2})
    
    template = """You are an advanced university assistant.
    Answer the question based ONLY on the following context.
    If the user asks for a summary, provide a comprehensive and detailed summary.
    
    Context:
    {context}
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    if ENABLE_LOCAL_MODE:
        llm = ChatOllama(model="llama3", temperature=0, base_url="http://127.0.0.1:11434")
        model_name_display = "Local Llama 3"
    else:
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
        model_name_display = "OpenAI GPT-4o"

    # 4. Generate Answer
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner(f"Thinking using {model_name_display}..."):
            try:
                relevant_docs = retriever.invoke(user_question)
                context_text = "\n\n".join([d.page_content for d in relevant_docs])
                
                formatted_prompt = prompt.invoke({"context": context_text, "question": user_question})
                response = llm.invoke(formatted_prompt)
                
                # Show Text Response
                st.markdown(response.content)
                st.session_state.messages.append({"role": "assistant", "content": response.content})
                
                # Show Sources (Images) in Expander
                with st.expander("📌 View Source Slides & References", expanded=False):
                    sources_map = {}
                    for doc in relevant_docs:
                        source_name = doc.metadata.get('source')
                        page_num = doc.metadata.get('page', 0)
                        if source_name not in sources_map:
                            sources_map[source_name] = set()
                        sources_map[source_name].add(page_num)
                    
                    for source_name, pages in sources_map.items():
                        file_path = st.session_state.file_paths.get(source_name)
                        if file_path:
                            st.markdown(f"**📄 Source: `{source_name}`**")
                            if source_name.lower().endswith('.pdf'):
                                try:
                                    pdf_doc = fitz.open(file_path)
                                    cols = st.columns(len(pages)) # Dynamic columns
                                    for idx, page_num in enumerate(sorted(pages)[:3]):
                                        with cols[idx]: # কলামের মধ্যে ইমেজ বসবে
                                            page = pdf_doc.load_page(page_num)
                                            # 🔥 FIX: DPI 200 (High Quality)
                                            pix = page.get_pixmap(dpi=200) 
                                            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                                            # 🔥 FIX: use_container_width=True (Full width, Clear view)
                                            st.image(img, caption=f"Page {page_num + 1}", use_container_width=True)
                                except:
                                    pass
                            else:
                                img = Image.open(file_path)
                                st.image(img, caption="Source Image", use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")