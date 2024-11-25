import streamlit as st
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os

# Set page configuration
st.set_page_config(page_title="Study Buddy", layout="wide")

# Initialize session states
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Function to load and process documents
@st.cache_resource
def initialize_vectorstore(directory_path):
    # Load documents from directory
    loader = DirectoryLoader(directory_path, glob="**/*.pdf")  # Adjust file pattern as needed
    documents = loader.load()
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    texts = text_splitter.split_documents(documents)
    
    # Create embeddings and vectorstore
    embeddings = OllamaEmbeddings(model="llama3.2")
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    return vectorstore

# Main app
def main():
    st.title("📚 Study Buddy - Your Personal Study Assistant")
    
    # Sidebar for document upload and model selection
    with st.sidebar:
        st.header("Configuration")
        documents_path = st.text_input("Enter documents directory path:", "path/to/your/documents")
        model_name = st.selectbox("Select Model:", ["llama3.2", "mistral"])
        temperature = st.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.7)
    
    # Initialize the conversation chain
    if st.sidebar.button("Initialize Study Buddy"):
        with st.spinner("Processing your study materials..."):
            try:
                vectorstore = initialize_vectorstore(documents_path)
                llm = Ollama(model=model_name, temperature=temperature)
                
                memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True
                )
                
                st.session_state.conversation = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=vectorstore.as_retriever(),
                    memory=memory,
                    verbose=True
                )
                
                st.sidebar.success("Study Buddy is ready to help!")
            except Exception as e:
                st.sidebar.error(f"Error initializing: {str(e)}")
    
    # Chat interface
    if st.session_state.conversation is None:
        st.info("Please initialize the Study Buddy first using the sidebar.")
    else:
        # Chat interface
        for message in st.session_state.chat_history:
            if isinstance(message, dict):
                role = message.get("role", "")
                content = message.get("content", "")
                with st.chat_message(role):
                    st.write(content)
        
        # User input
        user_question = st.chat_input("Ask me anything about your study materials!")
        if user_question:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.write(user_question)
            
            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.conversation.invoke({"question": user_question})
                    st.write(response["answer"])
                    st.session_state.chat_history.append({"role": "assistant", "content": response["answer"]})

if __name__ == "__main__":
    main()
