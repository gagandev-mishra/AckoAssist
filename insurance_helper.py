from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
import os 

load_dotenv()

st.image(r"INSURANCE_CHAT_BOT\acko_logo.png", width=148)
st.title("Why Should Medical Insurance Be Complicated? Ask Acko Assist! 🧞")

def create_vector():
    # load file
    files = ["D:\\Learning Materials\\CampusX\\LangChain\\RAG\\INSURANCE_CHAT_BOT\\acko_clause.pdf", 
             "D:\\Learning Materials\\CampusX\\LangChain\\RAG\\INSURANCE_CHAT_BOT\\acko_policy.pdf"]
    for file_name in files:
        loader = PyPDFLoader(file_name)
        pdf_document = loader.load_and_split()
        print(f"{len(pdf_document)} Pages Loaded")

        # Split the file into chunk
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 100, 
            chunk_overlap=20,
            separators=["\n\n", "\n", " ", ""]
        )

        split_documents = text_splitter.split_documents(pdf_document)
        print(f"Split into {len(split_documents)} Documents...")

        # Upload Chunks into Embedding form
        embedding = HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5')
        # Save embedding form into VectorDB
        db = FAISS.from_documents(split_documents, embedding)
        db.save_local("faiss_index")

def build_chat_history(chat_history_list):
    chat_history = []
    for message in chat_history_list:
        chat_history.append(HumanMessage(content=message[0]))
        chat_history.append(AIMessage(content=message[1]))

    return chat_history

def query(question, chat_history):
    chat_history = build_chat_history(chat_history)
    embedding = HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5')
    new_db = FAISS.load_local(r"INSURANCE_CHAT_BOT\faiss_index", 
                              embedding,
                              allow_dangerous_deserialization=True)
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite",
                                google_api_key=os.getenv("GOOGLE_API_KEY"),
                                temperature=0.5)
    
    condense_question_system_template = (
        "Given a chat history and the latest user question"
        "Which might refrence context in the chat history,"
        "formulate a standalone questions which can be understood"
        "without the chat history. Do NOT answer the question,"
        "just reformulate it if needed and othewise return it as it."
        "Try to explain thing in detail, and find best solutions"     
    )

    condense_question_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", condense_question_system_template),
            ("placeholder", "{chat_history}"),
            ("human", "{input}")
        ]
    )

    history_aware_retriever =  create_history_aware_retriever(
        model, new_db.as_retriever(), condense_question_prompt
    )

    system_prompt = (
        "You are well experienced insurance in medical domain, and you are working for Acko Health Insurance."
        "You're name is Genie"
        "You know how everthing works in hospital, medical insurance company, and people looking before buying health insurance"
        "For you refrence use the retrieved context to the user answer"
        "If you don't know the answer, then gently deny the request"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )

    qa_chain = create_stuff_documents_chain(llm=model, prompt=qa_prompt)
    convo_qa_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    return convo_qa_chain.invoke(
        {
            "input": question,
            "chat_history": chat_history
        }
    )

def chat_ui():
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_history = []

    # Chat message from history on app re-run
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    # Accept user prompt
    if prompt := st.chat_input("What question do you have?"):
        with st.spinner("Let read your query..."):
            response = query(question=prompt, chat_history=st.session_state.chat_history)
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                st.markdown(response['answer'])

            st.session_state.messages.append({
                'role': 'user',
                'content': prompt
            })

            st.session_state.messages.append({
                'role': 'assistant',
                'content': response['answer']
            })

            st.session_state.chat_history.extend([
                prompt,
                response['answer']
            ])

if __name__ == "__main__":
    chat_ui()
    #create_vector()