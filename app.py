import streamlit as st
import os
from dotenv import load_dotenv
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import platform

# --- LangChain Imports ---
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
ENABLE_LOCAL_MODE = False # রিপোর্টে লোকাল সিস্টেমের কথা বলা হয়েছে
# ==========================================

# --- OCR Configuration ---
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else:
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

st.set_page_config(page_title="PSNS: Study Notes", page_icon="📚", layout="wide")
st.title("📚 PSNS: Personal Study Notes Searcher")

# মোড ইন্ডিকেটর
if ENABLE_LOCAL_MODE:
    st.markdown("### 🟢 Mode: **Local Llama 3 (Privacy Focused)**")
else:
    st.markdown("### 🔵 Mode: **OpenAI GPT-4o (High Speed)**")

# API Key Check
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("⚠️ API Key not found! Please check your .env file.")
    st.stop()

# --- Session State ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "file_paths" not in st.session_state:
    st.session_state.file_paths = {}
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 1. Upload Section ---
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
                    file_path = os.path.join("temp_files", uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    st.session_state.file_paths[uploaded_file.name] = file_path
                    
                    # --- A. PDF Processing (Fixed to avoid 'bbox' error) ---
                    if uploaded_file.type == "application/pdf":
                        pdf_doc = fitz.open(file_path)
                        for page_num in range(len(pdf_doc)):
                            page = pdf_doc.load_page(page_num)
                            text = page.get_text()
                            if text.strip():
                                all_documents.append(Document(
                                    page_content=text, 
                                    metadata={'source': uploaded_file.name, 'page': page_num}
                                ))
                        
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
                    # রিপোর্টে ১০০০ চাঙ্ক এবং ২০০ ওভারল্যাপের কথা বলা হয়েছে
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    chunks = text_splitter.split_documents(all_documents)
                    
                    embeddings = OpenAIEmbeddings()
                    vector_store = FAISS.from_documents(chunks, embeddings)
                    st.session_state.vector_store = vector_store
                    
                    st.success(f"✅ Success! Processed {len(uploaded_files)} files.")
                    st.info(f"🔢 Total Chunks Indexed: {vector_store.index.ntotal}")
                else:
                    st.error("Could not extract text.")
            except Exception as e:
                st.error(f"Error during processing: {e}")

st.write("---")

# --- 2. Q&A Section ---

# চ্যাট হিস্ট্রি প্রদর্শন
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# চ্যাট ইনপুট
if user_question := st.chat_input("💬 Ask a question from your notes..."):
    
    # ইউজার মেসেজ সেভ ও শো
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    if not st.session_state.vector_store:
        st.warning("⚠️ Please upload and process files first!")
        st.stop()

    # রিট্রিভাল সেটআপ (k=3 রিপোর্টে উল্লিখিত)
    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
    
    template = """Answer the question based ONLY on the context.
    Context: {context}
    Question: {question}"""
    prompt_obj = ChatPromptTemplate.from_template(template)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # প্রাসঙ্গিক ডকুমেন্ট রিট্রিভ করা
                relevant_docs = retriever.invoke(user_question)
                context_text = "\n\n".join([d.page_content for d in relevant_docs])
                
                # মডেল সিলেকশন
                if ENABLE_LOCAL_MODE:
                    llm = ChatOllama(model="llama3", temperature=0, base_url="http://127.0.0.1:11434", num_ctx=4096)
                else:
                    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

                # উত্তর জেনারেট করা
                response = llm.invoke(prompt_obj.invoke({"context": context_text, "question": user_question}))
                response_content = response.content
                
                # উত্তর প্রদর্শন
                st.markdown(response_content)
                
                # সেশন স্টেটে উত্তর যোগ করা
                st.session_state.messages.append({"role": "assistant", "content": response_content})
                
                # সোর্স ইমেজ প্রদর্শন
                with st.expander("📌 View Source Slides"):
                    for doc in relevant_docs:
                        src = doc.metadata.get('source')
                        path = st.session_state.file_paths.get(src)
                        if path and src.lower().endswith('.pdf'):
                            pg = doc.metadata.get('page', 0)
                            pdf = fitz.open(path)
                            pix = pdf.load_page(pg).get_pixmap(dpi=150)
                            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                            st.image(img, caption=f"Source: {src} (Page {pg+1})", use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")

# --- ✅ ফিক্সড সেভ বাটন লজিক (চ্যাট ইনপুটের বাইরে) ---
# এটি নিশ্চিত করে যে শেষ অ্যাসিস্ট্যান্ট মেসেজটি সেভ হবে
if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    last_answer = st.session_state.messages[-1]["content"]
    last_query = st.session_state.messages[-2]["content"] if len(st.session_state.messages) > 1 else "Unknown"
    
    if st.button("💾 Save Insight", key=f"save_btn_{len(st.session_state.messages)}"):
        try:
            file_name = "saved_notes.txt"
            with open(file_name, "a", encoding="utf-8") as f:
                f.write(f"Question: {last_query}\nAnswer: {last_answer}\n" + "-"*30 + "\n")
            
            st.success(f"✅ Insight saved to {file_name}!")
            st.info(f"📁 Full Path: {os.path.abspath(file_name)}")
        except Exception as e:
            st.error(f"Save error: {e}")