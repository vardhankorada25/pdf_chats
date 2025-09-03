import streamlit as st
from dotenv import load_dotenv
import pdfplumber
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from htmltemplates import css, bot_template, user_template

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
        chunk_size=800,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# **REMOVED THE @st.cache_resource DECORATOR**
def get_conversation_chain(vectorstore):
    # Changed model to a larger, more capable one for better accuracy
    model_name = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=128,
        temperature=0.5,
        num_beams=1,
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        # Increased the number of retrieved chunks for more context
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
    )
    return conversational_chain

# Handle user input and generate response
def handle_user(user_question):
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
                    # Clear chat history when new documents are processed
                    st.session_state.chat_history = None
                    st.success("PDFs processed and chat ready!")

if __name__ == "__main__":
    main()
