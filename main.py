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
import datetime
from scripts import langchain_qna as lc_qna
from scripts import constants as c
from scripts import helpers as h

# Set OpenAI API parameters
os.environ["OPENAI_API_KEY"] = "6cdb659e5a9d402e80c212fe8ea26483"
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"
os.environ["OPENAI_API_BASE"] = "https://test-chatgpt-flomoney.openai.azure.com/"

st.title("Document-based Chatbot")

# User input options: Text-based content, PDF upload, or URL upload
input_option = st.radio("Select Input Option", ["Text-based Content", "PDF Upload"])

langchain_qna = lc_qna.LangchainQnA(c.chunking_interface, c.embedding_model)

if input_option == "Text-based Content":
    user_input = st.text_area("Enter text content")
elif input_option == "PDF Upload":
    user_input = None
    if "formatted_datetime" not in st.session_state:
        current_datetime = datetime.datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H-%M-%S")
        st.session_state.formatted_datetime = formatted_datetime
        save_folder = "data/output/" + formatted_datetime
    else:
        formatted_datetime = st.session_state.formatted_datetime
        save_folder = "data/output/" + formatted_datetime
    uploaded_files = st.file_uploader(
        "Upload multiple PDF files", type=["pdf"], accept_multiple_files=True
    )
    if uploaded_files:
        save_button = st.button("Save Files")
        if save_button:
            st.success(f"Formatted datetime: {formatted_datetime}")
            save_folder = "data/output/" + formatted_datetime
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            h.save_uploaded_files(uploaded_files, save_folder)

            # prompt_template_file = os.path.join(os.getcwd(), 'scripts/prompt_template.txt')
            prompt_template_file = None
            pdf_list = [
                _ for _ in glob.glob(os.path.join(os.getcwd(), save_folder, "*.pdf"))
            ]
            web_list = []

            qna_chain = langchain_qna.main_function(
                pdf_list, web_list, prompt_template_file
            )
            st.session_state.qna_chain = qna_chain

chatbot_responses = []  # Store chatbot responses

# Initialize a list to store chat history
chat_history = []

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    h.display_chat_message(message["role"], message["content"])

# Respond to user input
if user_input := st.chat_input("What would you like to ask FinBot?"):
    # Display user message in chat container
    h.display_chat_message("user", user_input)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    response = st.session_state.qna_chain(
        {"query": f"Based on the provided content, {user_input}"}
    )
    response = response["result"]
    # Display chatbot response in chat container
    h.display_chat_message("bot", response)
    # Add chatbot response to chat history
    st.session_state.messages.append({"role": "bot", "content": response})
    