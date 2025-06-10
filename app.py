import streamlit as st
from dotenv import load_dotenv
import pdfplumber
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from htmltemplates import css, bot_template, user_template
import os
import pickle  # Added for caching

# Load text from PDF

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        with pdfplumber.open(pdf) as pdf_reader:
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    return text
# Split text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,  # Reduced chunk size for faster processing
        chunk_overlap=150,  # Slightly reduced overlap
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Create FAISS vectorstore with embeddings
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Save vectorstore to disk
def save_vectorstore(vectorstore, path="vectorstore.pkl"):
    with open(path, "wb") as f:
        pickle.dump(vectorstore, f)

# Load vectorstore from disk
def load_vectorstore(path="vectorstore.pkl"):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

# Use HuggingFacePipeline with optimized settings
def get_conversation_chain(vectorstore):
    model_name = "google/flan-t5-small"  # Switched to smaller model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=128,  # Reduced max_length for faster generation
        temperature=0.5,
        num_beams=1,  # Disabled beam search for speed
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),  # Reduced k for faster retrieval
        memory=memory,
    )
    return conversational_chain

# Handle user input and generate response
def handle_user(user_question):  # Fixed typo
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

# Main Streamlit app
def main():
    load_dotenv()
    st.set_page_config(
        page_title="Chat with Multiple PDFs",
        page_icon=":books:",
        layout="wide"
    )
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.title("Chat with Multiple PDFs")

    user_question = st.text_input("Ask a question about the documents")
    if user_question and st.session_state.conversation:
        handle_user(user_question)
    user_question =None

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
                    # Check for cached vectorstore
                    vectorstore = load_vectorstore()
                    if vectorstore is None:
                        raw_text = get_pdf_text(pdf_docs)
                        text_chunks = get_text_chunks(raw_text)
                        vectorstore = get_vectorstore(text_chunks)
                        save_vectorstore(vectorstore)  # Cache vectorstore
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.success("PDFs processed and chat ready!")

if __name__ == "__main__":
    main()