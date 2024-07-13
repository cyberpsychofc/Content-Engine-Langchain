from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
##Langsmith Tracking
os.environ["LANGCHAIN_TRACING"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are an assistant. Please respond to the user query."),
        ("user","Question:{question}")
    ]
)

##Interface
st.title('Search the Content Engine')

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

#llm 
client = ChatGroq(model="llama3-8b-8192")
output_parser = StrOutputParser()
chain = prompt | client | output_parser

#User input
if prompt := st.chat_input("Should I invest in..."):
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    response = chain.invoke({'question':prompt})

    st.session_state.messages.append({"role":"assistant","content":response})
    
    with st.chat_message("assistant"):
        st.markdown(response)