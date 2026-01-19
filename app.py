import streamlit as st
import os
from dotenv import load_dotenv
import fitz  # PyMuPDF
from PIL import Image
import pytesseract # নতুন লাইব্রেরি (OCR এর জন্য)

# --- LangChain ইম্পোর্ট ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document # ইমেজ থেকে টেক্সট বানিয়ে ডকুমেন্ট বানানোর জন্য

load_dotenv()

# --- OCR কনফিগারেশন (খুব গুরুত্বপূর্ণ) ---
# তোমার পিসিতে যদি Tesseract অন্য কোথাও ইন্সটল করো, তবে এই লাইনটি আপডেট করতে হবে
# সাধারণ পাথ: C:\Program Files\Tesseract-OCR\tesseract.exe
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

st.set_page_config(page_title="PSNS: Study Notes", page_icon="📚", layout="wide")
st.title("📚 PSNS: PDF & Image Searcher (OCR Enabled)")

# API Key চেক
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("⚠️ API Key পাওয়া যায়নি!")
    st.stop()

# সেশন স্টেট
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "uploaded_file_path" not in st.session_state:
    st.session_state.uploaded_file_path = None
if "file_type" not in st.session_state:
    st.session_state.file_type = None

# --- ১. আপলোড সেকশন (PDF + Image) ---
# এখন png, jpg, jpeg ফাইলও আপলোড করা যাবে
uploaded_file = st.file_uploader("লেকচার স্লাইড (PDF) বা নোটের ছবি আপলোড করো", type=['pdf', 'png', 'jpg', 'jpeg'])

if uploaded_file:
    # টেম্প ফোল্ডার চেক
    if not os.path.exists("temp_files"):
        os.makedirs("temp_files")
    
    # ফাইল সেভ করা
    file_path = os.path.join("temp_files", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.session_state.uploaded_file_path = file_path
    st.session_state.file_type = uploaded_file.type

    if st.button("🧠 প্রসেস শুরু করুন (OCR/PDF)"):
        with st.spinner("ফাইল পড়া হচ্ছে... (Images may take time)"):
            try:
                documents = []
                
                # A. যদি PDF হয়
                if uploaded_file.type == "application/pdf":
                    loader = PyPDFLoader(file_path)
                    documents = loader.load()
                    st.info("📄 PDF মোডে প্রসেস হচ্ছে...")

                # B. যদি ছবি (Image) হয় - OCR ব্যবহার হবে
                else:
                    st.info("📷 ছবি শনাক্ত হয়েছে। OCR দিয়ে টেক্সট বের করা হচ্ছে...")
                    image = Image.open(file_path)
                    # ছবি থেকে টেক্সট বের করা (Tesseract)
                    extracted_text = pytesseract.image_to_string(image)
                    
                    if not extracted_text.strip():
                        st.warning("⚠️ ছবি থেকে কোনো লেখা পাওয়া যায়নি! ছবি ক্লিয়ার তো?")
                    else:
                        # LangChain এর ফরম্যাটে ডকুমেন্ট বানানো
                        # ছবিতে পেজ নম্বর থাকে না, তাই আমরা Page 1 ধরে নিচ্ছি
                        doc = Document(page_content=extracted_text, metadata={"page": 0, "source": uploaded_file.name})
                        documents = [doc]

                # চাংকিং এবং এম্বেডিং (সবার জন্য কমন)
                if documents:
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    chunks = text_splitter.split_documents(documents)
                    
                    embeddings = OpenAIEmbeddings()
                    vector_store = FAISS.from_documents(chunks, embeddings)
                    
                    st.session_state.vector_store = vector_store
                    st.success(f"✅ সম্পন্ন! ব্রেইন তৈরি হয়েছে।")
                else:
                    st.error("কোনো টেক্সট প্রসেস করা সম্ভব হয়নি।")

            except Exception as e:
                st.error(f"Error details: {e}")
                st.info("টিপস: তোমার পিসিতে কি Tesseract ইন্সটল করা আছে? পাথ ঠিক আছে তো?")

st.write("---")

# --- ২. প্রশ্ন ও উত্তর সেকশন ---
user_question = st.text_input("প্রশ্ন করো:")

if user_question and st.session_state.vector_store:
    
    retriever = st.session_state.vector_store.as_retriever()
    
    template = """Answer the question based ONLY on the following context:
    {context}
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    with st.spinner("উত্তর খুঁজছি..."):
        try:
            # ডকুমেন্ট খোঁজা
            relevant_docs = retriever.invoke(user_question)
            context_text = "\n\n".join([d.page_content for d in relevant_docs])
            
            # উত্তর জেনারেট
            formatted_prompt = prompt.invoke({"context": context_text, "question": user_question})
            response = llm.invoke(formatted_prompt)
            
            st.success("🤖 AI উত্তর:")
            st.write(response.content)
            
            # রেফারেন্স দেখানো
            st.markdown("---")
            st.subheader("📌 রেফারেন্স:")
            
            unique_pages = set()
            for doc in relevant_docs:
                page_num = doc.metadata.get('page', 0)
                unique_pages.add(page_num)
            
            # স্লাইড/ছবি দেখানো (PDF হলে পেজ, ইমেজ হলে পুরো ইমেজ)
            if st.session_state.uploaded_file_path:
                
                # যদি অরিজিনাল ফাইল PDF হয়
                if "pdf" in st.session_state.file_type:
                    pdf_doc = fitz.open(st.session_state.uploaded_file_path)
                    for page_num in sorted(unique_pages):
                        with st.expander(f"📄 Page {page_num + 1} (Click to View)", expanded=False):
                            page = pdf_doc.load_page(page_num)
                            pix = page.get_pixmap(dpi=150)
                            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                            st.image(img, caption=f"Page {page_num + 1}", use_container_width=True)
                
                # যদি অরিজিনাল ফাইল ছবি হয়
                else:
                    with st.expander("📷 অরিজিনাল ছবি দেখুন", expanded=False):
                        img = Image.open(st.session_state.uploaded_file_path)
                        st.image(img, caption="Uploaded Image", use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}")

elif user_question and not st.session_state.vector_store:
    st.warning("⚠️ আগে ফাইল প্রসেস করো!")