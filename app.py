import streamlit as st
import os
from dotenv import load_dotenv

# --- ইম্পোর্ট ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

st.set_page_config(page_title="PSNS: Study Notes", page_icon="📚")
st.title("📚 Personal Study Notes Searcher")

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("⚠️ API Key পাওয়া যায়নি!")
    st.stop()

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# --- ১. আপলোড সেকশন (আগের মতোই) ---
uploaded_file = st.file_uploader("লেকচার স্লাইড (PDF) আপলোড করো", type=['pdf'])

if uploaded_file:
    if not os.path.exists("temp_files"):
        os.makedirs("temp_files")
    
    file_path = os.path.join("temp_files", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

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
# --- ২. প্রশ্ন ও উত্তর সেকশন (Fixed Version) ---
user_question = st.text_input("প্রশ্ন করো:")

if user_question and st.session_state.vector_store:
    
    # A. রিট্রিভার তৈরি
    retriever = st.session_state.vector_store.as_retriever()
    
    # B. উত্তর জেনারেট করার টেমপ্লেট
    template = """Answer the question based ONLY on the following context:
    {context}
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    with st.spinner("উত্তর খুঁজছি এবং সোর্স বের করছি..."):
        try:
            # ধাপ ১: আগে ডকুমেন্টগুলো (Chunks) খুঁজে বের করি
            relevant_docs = retriever.invoke(user_question)
            
            # ধাপ ২: ডকুমেন্টগুলোকে টেক্সটে কনভার্ট করি (Manual Formatting)
            # আমরা এখানে সরাসরি Python List Comprehension ব্যবহার করছি, যা অনেক সেফ
            context_text = "\n\n".join([d.page_content for d in relevant_docs])
            
            # ধাপ ৩: AI-এর জন্য প্রম্পট রেডি করা
            formatted_prompt = prompt.invoke({"context": context_text, "question": user_question})
            
            # ধাপ ৪: উত্তর জেনারেট করা
            response = llm.invoke(formatted_prompt)
            
            # আউটপুট দেখানো
            st.success("উত্তর:")
            st.write(response.content) # .content দিলে শুধু টেক্সট আসবে
            
            # ধাপ ৫: সোর্স দেখানো (Page Numbers)
            st.warning("📌 রেফারেন্স (Sources):")
            
            unique_pages = set()
            for doc in relevant_docs:
                # পেজ নম্বর চেক করা (যদি না থাকে তবে 0 ধরবে)
                page_num = doc.metadata.get('page', 0) + 1
                unique_pages.add(page_num)
            
            for page in sorted(unique_pages):
                st.write(f"📄 তথ্যটি **Page {page}** থেকে নেওয়া হয়েছে।")
                
            # ডিবাগিং (অপশনাল)
            with st.expander("🔍 বিস্তারিত সোর্স টেক্সট দেখুন"):
                for i, doc in enumerate(relevant_docs):
                    st.caption(f"Source {i+1} (Page {doc.metadata.get('page', 0) + 1})")
                    st.text(doc.page_content[:200] + "...")

        except Exception as e:
            st.error(f"Error: {e}")

elif user_question and not st.session_state.vector_store:
    st.warning("⚠️ আগে ফাইল প্রসেস করো!")