import streamlit as st
import ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# ---------------------------
# Streamlit page configuration
# ---------------------------
st.set_page_config(
    page_title="Chat with Webpage",
    page_icon="🌐",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Custom CSS styling
# ---------------------------
CUSTOM_CSS = """
<style>
/* Main container background */
main {
    background-color: #f8f9fa;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #dedede;
}

/* Chat message containers */
.user-message, .assistant-message {
    background: #ffffff;
    border-radius: 10px;
    padding: 15px;
    margin-bottom: 10px;
}

.user-message {
    border: 2px solid #1e88e5;
}

.assistant-message {
    border: 2px solid #9c27b0;
}

/* Title color */
h1 {
    color: #2e4053;
}

/* Subtitle or caption color */
div[data-testid="stCaptionContainer"] {
    color: #555555;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------
# Title and Intro
# ---------------------------
st.title("Chat with Webpage 🌐")
st.caption("This app allows you to chat with a webpage using local Llama and RAG, **with context memory**.")

# ---------------------------
# Sidebar for Webpage URL Input
# ---------------------------
st.sidebar.title("Webpage Loader")
webpage_url = st.sidebar.text_input("Enter Webpage URL", value="")

# Initialize session states
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------
# Load the webpage and build vectorstore
# ---------------------------
def load_webpage(url):
    """Loads the webpage from the given URL, splits, and creates a Chroma vector store."""
    loader = WebBaseLoader(url)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
    splits = text_splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model="llama3.2")  # Adjust model name as needed

    # Create a new persistent Chroma DB
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="my_new_chroma_db"  # or any unique name per session
    )
    return vectorstore

# If URL is provided, load data (only once to prevent reloading on each interaction)
if webpage_url and st.session_state.vectorstore is None:
    st.session_state.vectorstore = load_webpage(webpage_url)
    st.success(f"Loaded {webpage_url} successfully!")

# ---------------------------
# Helper Functions
# ---------------------------
def stream_parser(stream):
    """Generator to read streaming response from Ollama."""
    for chunk in stream:
        yield chunk['message']['content']

def combine_docs(docs):
    """Combine retrieved docs into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def ollama_llm(question, context, conversation_history):
    """
    Call the Ollama LLM with the question, the retrieved context, 
    and the recent conversation_history for better continuity.
    """
    # You can adjust how many recent turns of conversation you want to include
    # in the prompt to limit the context size. Here we use the last 3 messages.
    formatted_prompt = f"""
Conversation so far:
{conversation_history}

You are a helpful assistant. Use the context below and the conversation above to answer the user's last question.

Context:
{context}

User's last question: {question}

Please provide a concise, simple, and helpful answer.
"""
    response = ollama.chat(
        model='llama3.2',  # same model name as above
        messages=[{'role': 'user', 'content': formatted_prompt}],
        stream=True
    )
    return response

def rag_chain(question, conversation_history):
    """
    Retrieve relevant documents from vectorstore and pass them, 
    along with conversation history, to the LLM.
    """
    if st.session_state.vectorstore is None:
        return ["No webpage loaded. Please enter a URL on the sidebar first."]
    
    retriever = st.session_state.vectorstore.as_retriever()
    retrieved_docs = retriever.invoke(question)
    formatted_context = combine_docs(retrieved_docs)
    return ollama_llm(question, formatted_context, conversation_history)

# ---------------------------
# Display existing messages
# ---------------------------
for message in st.session_state.messages:
    if message["role"] == "user":
        # Display user messages
        st.markdown(
            f"<div class='user-message'><strong>You:</strong><br/>{message['content']}</div>",
            unsafe_allow_html=True
        )
    else:
        # Display assistant messages
        st.markdown(
            f"<div class='assistant-message'><strong>Assistant:</strong><br/>{message['content']}</div>",
            unsafe_allow_html=True
        )

# ---------------------------
# Chat input
# ---------------------------
user_prompt = st.chat_input("What would you like to ask?")

if user_prompt:
    # 1. Display user's message
    st.markdown(
        f"<div class='user-message'><strong>You:</strong><br/>{user_prompt}</div>",
        unsafe_allow_html=True
    )
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    # 2. Gather conversation history (last N messages)
    #    (We can adjust how many we want to keep; here we use up to 6 for a short context.)
    recent_messages = st.session_state.messages[-6:]
    conversation_history_str = ""
    for msg in recent_messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        conversation_history_str += f"{role}: {msg['content']}\n"

    # 3. Generate response with RAG + conversation history
    with st.spinner('Generating response...'):
        llm_stream = rag_chain(user_prompt, conversation_history_str)
        # 4. Stream the response
        partial_response = []
        for chunk in stream_parser(llm_stream):
            partial_response.append(chunk)
            # We could live-update in the UI if desired with st.write, 
            # but standard streaming in Streamlit is limited. 
            # We'll just gather everything and display at once.
        final_response = "".join(partial_response)

    # 5. Display assistant's message
    st.markdown(
        f"<div class='assistant-message'><strong>Assistant:</strong><br/>{final_response}</div>",
        unsafe_allow_html=True
    )
    st.session_state.messages.append({"role": "assistant", "content": final_response})
