from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_community.vectorstores import FAISS
import streamlit as st
import os
from dotenv import load_dotenv

# for custom deployemnt
'''
load_dotenv()

os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
##Langsmith Tracking
os.environ["LANGCHAIN_TRACING"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
'''
# for streamlit deployment
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
##Langsmith Tracking
os.environ["LANGCHAIN_TRACING"]="true"
os.environ["LANGCHAIN_API_KEY"]= st.secrets["LANGCHAIN_API_KEY"]

prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context.
You are supposed to obtain insights about information from the
documents, compare numbers.Think step by step before providing a detailed answer.
You will be reward you positvely if the user finds the answer helpful.
<context>
{context}
</context>
Question: {input}""")

loader = PyPDFDirectoryLoader("./../data")
#chunking
text_docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
documents = text_splitter.split_documents(text_docs)

#Vector Embeddings
embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True}
)

#Vector Store
db = FAISS.from_documents(documents,embeddings)
##Interface
st.title('Ask Questions!')

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

#llm 
client = ChatGroq(model="llama3-8b-8192")
#Chain
document_chain = create_stuff_documents_chain(client, prompt)
retriver = db.as_retriever()
retrieval_chain = create_retrieval_chain(retriver,document_chain)

#User input
if prompt := st.chat_input("Should I invest in..."):
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    response = retrieval_chain.invoke({'input':prompt})

    st.session_state.messages.append({"role":"assistant","content":response['answer']})
    with st.chat_message("assistant"):
        st.markdown(response['answer'])
