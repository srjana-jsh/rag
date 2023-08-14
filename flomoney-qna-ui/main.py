import streamlit as st
from bs4 import BeautifulSoup
import os
import requests
import openai
import base64
# Note: The openai-python library support for Azure OpenAI is in preview.
import sys
import glob
import re
import importlib
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
sys.path.append(os.path.join(os.getcwd(), 'scripts'))
import langchain_qna as lc_qna
importlib.reload(lc_qna)
import datetime

# Set OpenAI API parameters
os.environ["OPENAI_API_KEY"] = "6cdb659e5a9d402e80c212fe8ea26483"
openai.api_type = "azure"
openai.api_base = "https://test-chatgpt-flomoney.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = os.getenv("OPENAI_API_KEY")



def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def save_uploaded_files(uploaded_files, save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    for file in uploaded_files:
        file_path = os.path.join(save_folder, file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())
        st.success(f"Saved file: {file_path}")


st.title("Document-based Chatbot")

# User input options: Text-based content, PDF upload, or URL upload
input_option = st.radio("Select Input Option", ["Text-based Content", "PDF Upload"])

chunking_interface = RecursiveCharacterTextSplitter
embedding_model = OpenAIEmbeddings
chunk_size = 8000
chunk_overlap = 0
vectorstore_engine = 'Finbot-embedding'
llm_model = 'text-davinci-002'
llm_engine = 'finbot-gpt'
temperature = 0
search_type = 'mmr'
retrieval_kwargs = {'k': 6, 'lambda_mult': 0.5}

langchain_qna = lc_qna.LangchainQnA(chunking_interface, embedding_model)

if input_option == "Text-based Content":
    user_input = st.text_area("Enter text content")
elif input_option == "PDF Upload":
    user_input = None
    if 'formatted_datetime' not in st.session_state:
        current_datetime = datetime.datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H-%M-%S")
        st.session_state.formatted_datetime = formatted_datetime
        save_folder = 'data/output/' + formatted_datetime
    else:
        formatted_datetime = st.session_state.formatted_datetime
        save_folder = 'data/output/' + formatted_datetime
    uploaded_files = st.file_uploader("Upload multiple PDF files", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        save_button = st.button("Save Files")
        if save_button:
            st.success(f"Formatted datetime: {formatted_datetime}")
            save_folder = 'data/output/' + formatted_datetime
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            save_uploaded_files(uploaded_files, save_folder)
        

            
        
            pdf_list = [_ for _ in glob.glob(os.path.join(os.path.join(os.getcwd(), save_folder), '*.pdf')) ]
            print (pdf_list)
            web_list = []

            qna_chain = langchain_qna.main_function(
                            pdf_list, web_list, chunk_size, chunk_overlap, vectorstore_engine,
                            llm_model, llm_engine, temperature, search_type, **retrieval_kwargs
                            )

            chatbot_responses = []  # Store chatbot responses

            # Initialize a list to store chat history
            chat_history = []

            # Function to display chat messages
            def display_chat_message(sender, message):
                if sender == "user":
                    with st.chat_message(name = "You", avatar="👨‍🎓"):
                        st.write(message)
                elif sender == "bot":
                    with st.chat_message(name = "FinBot", avatar="🤖"):
                        st.write(message)

            # Initialize chat history
            if "messages" not in st.session_state:
                st.session_state.messages = []

            # Display chat messages from history on app rerun
            for message in st.session_state.messages:
                display_chat_message(message["role"], message["content"])

            # Respond to user input
            if user_input := st.chat_input("What would you like to ask FinBot?"):
                # Display user message in chat container
                display_chat_message("user", user_input)
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": user_input})   

                response = qna_chain({"query": user_input})
                response = response['result']
                # Display chatbot response in chat container
                display_chat_message("bot", response)
                # Add chatbot response to chat history
                st.session_state.messages.append({"role": "bot", "content": response})

    