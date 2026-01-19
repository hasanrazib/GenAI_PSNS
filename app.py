import streamlit as st
import os
from dotenv import load_dotenv

# --- আধুনিক ইম্পোর্ট (LCEL - Modern Approach) ---
# এই লাইব্রেরিগুলো তোমার পিসিতে অলরেডি ঠিকভাবে ইন্সটল আছে
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# .env লোড করা
load_dotenv()

st.set_page_config(page_title="PSNS: Study Notes Searcher", page_icon="📚")
st.title("📚 Personal Study Notes Searcher")

# API Key চেক
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("⚠️ .env ফাইলে API Key পাওয়া যায়নি!")
    st.stop()

# মেমোরি স্টেট
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# --- ১. ফাইল আপলোড সেকশন ---
uploaded_file = st.file_uploader("লেকচার স্লাইড (PDF) আপলোড করো", type=['pdf'])

if uploaded_file:
    # টেম্প ফাইল সেভ করা
    if not os.path.exists("temp_files"):
        os.makedirs("temp_files")
    
    file_path = os.path.join("temp_files", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # প্রসেসিং বাটন
    if st.button("🧠 প্রসেস শুরু করুন"):
        with st.spinner("ব্রেইন তৈরি হচ্ছে... (Smart LCEL Mode)"):
            try:
                # PDF পড়া
                loader = PyPDFLoader(file_path)
                pages = loader.load()

                # টেক্সট টুকরো করা
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_documents(pages)
                
                # ভেক্টর ডেটাবেস তৈরি
                embeddings = OpenAIEmbeddings()
                vector_store = FAISS.from_documents(chunks, embeddings)
                
                # সেশনে সেভ
                st.session_state.vector_store = vector_store
                st.success(f"✅ সম্পন্ন! {len(pages)} পেজ পড়া হয়েছে। এখন নিচে প্রশ্ন করো।")

            except Exception as e:
                st.error(f"Error: {e}")

st.write("---")

# --- ২. প্রশ্ন ও উত্তর সেকশন (LCEL চেইন) ---
user_question = st.text_input("প্রশ্ন করো:")

if user_question and st.session_state.vector_store:
    # A. রিট্রিভার (Retriever) - তথ্য খোঁজার জন্য
    retriever = st.session_state.vector_store.as_retriever()
    
    # B. প্রম্পট টেমপ্লেট (AI কে নির্দেশ)
    template = """You are a helpful assistant for university students.
    Answer the question based ONLY on the following context:
    {context}
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    # C. LLM (GPT Model)
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    
    # D. ফরম্যাটিং ফাংশন
    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])
    
    # E. চেইন তৈরি (LCEL পাইপলাইন)
    # Retriever -> Format -> Prompt -> LLM -> Output Parser
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    with st.spinner("উত্তর খুঁজছি..."):
        try:
            response = rag_chain.invoke(user_question)
            st.success("উত্তর:")
            st.write(response)
        except Exception as e:
            st.error(f"Error: {e}")

elif user_question and not st.session_state.vector_store:
    st.warning("⚠️ আগে ফাইল আপলোড করে 'প্রসেস' বাটনে চাপ দাও!")