import streamlit as st
from dotenv import load_dotenv
import pdfplumber
import docx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.prompts import PromptTemplate
import os
import hashlib
import pickle
# import concurrent.futures

# --- CUSTOM CSS FOR AMAZING UI ---
CUSTOM_CSS = """
<style>
body {
    background-color: #f4f6fa;
}
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #2d3a4a;
    margin-bottom: 0.5rem;
    letter-spacing: 1px;
}
.chat-bubble {
    padding: 1rem;
    border-radius: 1.2rem;
    margin-bottom: 1rem;
    max-width: 80%;
    font-size: 1.1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    color: #222;
}
.user-bubble {
    background: linear-gradient(90deg, #e0e7ff 0%, #f3f4f6 100%);
    align-self: flex-end;
    margin-left: auto;
    color: #222;
}
.bot-bubble {
    background: linear-gradient(90deg, #f0fdfa 0%, #e0f2fe 100%);
    align-self: flex-start;
    margin-right: auto;
    color: #222;
}
.sidebar .sidebar-content {
    background: #fff;
    border-radius: 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    padding: 1.5rem 1rem;
}
</style>
"""


# --- PAGE CONFIG ---
load_dotenv()
st.set_page_config(page_title="Fast RAG PDF Chat", page_icon=":books:", layout="wide")

# --- CONFIGURATION ---
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# --- PROMPT TEMPLATE FOR FOCUSED ANSWERS ---
QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant. Use ONLY the following context to answer the question. 
If the answer is not in the context, say 'I don't know.'

Context:
{context}

Question: {question}
Answer:
"""
)

# --- TEXT CHUNKING (Optimized for RAG) ---
# --- TEXT CHUNKING (Optimized for RAG) ---
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len
    )
    return text_splitter.split_text(text)


# --- PDF TEXT EXTRACTION (missing in your code) ---

# --- GENERIC TEXT EXTRACTION FOR PDF, TXT, DOCX ---
def extract_text_from_file(file):
    if file.name.lower().endswith('.pdf'):
        with pdfplumber.open(file) as pdf_reader:
            text = ""
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            return text
    elif file.name.lower().endswith('.txt'):
        file.seek(0)
        return file.read().decode(errors='ignore')
    elif file.name.lower().endswith('.docx'):
        file.seek(0)
        doc = docx.Document(file)
        return '\n'.join([para.text for para in doc.paragraphs])
    else:
        return ""

def get_files_text(files):
    text = ""
    for file in files:
        text += extract_text_from_file(file) + "\n"
    return text

# --- VECTORSTORE WITH CACHING ---
def get_vectorstore(text_chunks):
    if not text_chunks:
        return None
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    chunks_hash = hashlib.md5("".join(text_chunks).encode()).hexdigest()
    vector_cache_file = os.path.join(CACHE_DIR, f"vectorstore_{chunks_hash}.pkl")
    if os.path.exists(vector_cache_file):
        with open(vector_cache_file, "rb") as f:
            return pickle.load(f)
    else:
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        with open(vector_cache_file, "wb") as f:
            pickle.dump(vectorstore, f)
        return vectorstore

# --- HUGGINGFACE LLM FOR FAST, ACCURATE RESPONSES ---
def get_conversation_chain(vectorstore):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    model_name = "google/flan-t5-base"  # You can change to any supported HuggingFace model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=256,
        temperature=0.3,
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT}
    )

# --- HANDLE USER INPUT ---

# --- HANDLE USER INPUT WITH MODERN CHAT BUBBLES ---
def handle_user(user_question):
    with st.spinner("Thinking..."):
        response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]
    chat_placeholder = st.container()
    with chat_placeholder:
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.markdown(f"<div class='chat-bubble user-bubble'>{message.content}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='chat-bubble bot-bubble'>{message.content}</div>", unsafe_allow_html=True)

# --- MAIN STREAMLIT APP ---

def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Header with icon
    st.markdown("<div class='main-header'>⚡ Fast RAG PDF Chat</div>", unsafe_allow_html=True)
    st.caption("Ask questions about your files and get instant, accurate answers.")

    # Layout: Sidebar for upload, main for chat
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("### 📂 Upload Files")
        files = st.file_uploader(
            "Upload your files (PDF, TXT, DOCX)",
            accept_multiple_files=True,
            type=["pdf", "txt", "docx"]
        )
        if st.button("Process"):
            if files:
                with st.spinner("Processing your files..."):
                    raw_text = get_files_text(files)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.session_state.chat_history = []
                    st.success("Files processed and chat ready!")

    with col2:
        user_question = st.text_input("Ask a question about the documents", key="user_question")
        if user_question and st.session_state.conversation:
            handle_user(user_question)




if __name__ == "__main__":
    main()
