import streamlit as st
import os
from dotenv import load_dotenv
import fitz  # PyMuPDF
from PIL import Image

# --- LangChain ইম্পোর্ট ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

st.set_page_config(page_title="PSNS: Study Notes", page_icon="📚", layout="wide")
st.title("📚 Personal Study Notes Searcher")

# API Key চেক
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("⚠️ API Key পাওয়া যায়নি!")
    st.stop()

# সেশন স্টেট (মেমোরি)
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "uploaded_file_path" not in st.session_state:
    st.session_state.uploaded_file_path = None

# --- ১. আপলোড সেকশন ---
uploaded_file = st.file_uploader("লেকচার স্লাইড (PDF) আপলোড করো", type=['pdf'])

if uploaded_file:
    # টেম্প ফোল্ডার চেক
    if not os.path.exists("temp_files"):
        os.makedirs("temp_files")
    
    # ফাইল সেভ করা
    file_path = os.path.join("temp_files", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # ফাইল পাথ সেশনে রাখা
    st.session_state.uploaded_file_path = file_path

    if st.button("🧠 প্রসেস শুরু করুন"):
        with st.spinner("ব্রেইন তৈরি হচ্ছে..."):
            try:
                loader = PyPDFLoader(file_path)
                pages = loader.load()

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_documents(pages)
                
                embeddings = OpenAIEmbeddings()
                vector_store = FAISS.from_documents(chunks, embeddings)
                
                st.session_state.vector_store = vector_store
                st.success(f"✅ সম্পন্ন! {len(pages)} পেজ পড়া হয়েছে।")

            except Exception as e:
                st.error(f"Error: {e}")

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
            # A. ডকুমেন্ট খোঁজা
            relevant_docs = retriever.invoke(user_question)
            
            # B. টেক্সট বানানো
            context_text = "\n\n".join([d.page_content for d in relevant_docs])
            
            # C. উত্তর জেনারেট
            formatted_prompt = prompt.invoke({"context": context_text, "question": user_question})
            response = llm.invoke(formatted_prompt)
            
            # D. উত্তর দেখানো
            st.success("🤖 AI উত্তর:")
            st.write(response.content)
            
            # E. সোর্স এবং ছবি দেখানো (লুকানো অবস্থায় থাকবে)
            st.markdown("---")
            st.subheader("📌 রেফারেন্স (প্রয়োজন হলে ক্লিক করে দেখুন):")
            
            # ইউনিক পেজ নম্বর বের করা
            unique_pages = set()
            for doc in relevant_docs:
                page_num = doc.metadata.get('page', 0)
                unique_pages.add(page_num)
            
            if st.session_state.uploaded_file_path:
                pdf_doc = fitz.open(st.session_state.uploaded_file_path)
                
                cols = st.columns(len(unique_pages))
                
                for idx, page_num in enumerate(sorted(unique_pages)):
                    # CHANGE HERE: expanded=False করে দেওয়া হয়েছে
                    with st.expander(f"📄 Page {page_num + 1} (Click to View Slide)", expanded=False):
                        st.info(f"Source: Page {page_num + 1}")
                        
                        page = pdf_doc.load_page(page_num)
                        pix = page.get_pixmap(dpi=150)
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        
                        st.image(img, caption=f"Slide Page: {page_num + 1}", use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}")

elif user_question and not st.session_state.vector_store:
    st.warning("⚠️ আগে ফাইল প্রসেস করো!")