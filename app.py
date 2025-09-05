import streamlit as st
from dotenv import load_dotenv
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.prompts import PromptTemplate
import os
import hashlib
import pickle
import concurrent.futures
# If you have these templates and css, import them:
from htmltemplates import css, bot_template, user_template


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
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        with pdfplumber.open(pdf) as pdf_reader:
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
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
        vectorstore = Chroma.from_texts(texts=text_chunks, embedding=embeddings)
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
def handle_user(user_question):
    with st.spinner("Thinking..."):
        response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

# --- MAIN STREAMLIT APP ---
def main():
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.title("⚡ Fast RAG PDF Chat")
    st.caption("Ask questions about your PDFs and get instant, accurate answers.")

    user_question = st.text_input("Ask a question about the documents")
    if user_question and st.session_state.conversation:
        handle_user(user_question)

    with st.sidebar:
        st.header("Your Documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs",
            accept_multiple_files=True,
            type=["pdf"]
        )
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing your PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.session_state.chat_history = None
                    st.success("PDFs processed and chat ready!")




if __name__ == "__main__":
    main()
