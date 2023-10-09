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
from scripts import qna_memory as qna_m
from scripts import constants as c
from scripts import helpers as h
from scripts import scraper as s

# Set OpenAI API parameters
os.environ["OPENAI_API_KEY"] = "6cdb659e5a9d402e80c212fe8ea26483"
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"
os.environ["OPENAI_API_BASE"] = "https://test-chatgpt-flomoney.openai.azure.com/"

st.title("User-input based Chatbot")

# User input options: Text-based content, PDF upload, or URL upload
input_option = st.radio(
    "Select Input Option", 
    ["Text-based Content", "PDF Upload", "URL Upload"]
)
#prompt templates
qna_prompt_template = os.path.join(
    os.getcwd(), 'scripts/qna_prompt_template.txt'
)
# prompt_template_file = None
condense_question_template = os.path.join(
    os.getcwd(), 'scripts/condense_question_template.txt'
)
# condense_question_template = None

#inputs from user
web_list = []
pdf_list = []
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
            os.makedirs(save_folder, exist_ok=True)
            h.save_uploaded_files(uploaded_files, save_folder)
            #pdfs for question-answering
            pdf_list = [
                _ for _ in glob.glob(os.path.join(os.getcwd(), save_folder, "*.pdf"))
            ]            
elif input_option == "URL Upload":
    user_input = st.text_area("Enter URL")
    save_button = st.button("Save URL")
    #web pages for question-answering
    if save_button:
        web_list = s.scrape_site(user_input, c.header_template)
#chain for question-answering
if len(web_list) > 0 or len(pdf_list) > 0:
    qna_with_memory = qna_m.LangchainQnA(c.chunking_interface, c.embedding_model)
    qna_chain = qna_with_memory.main_function(
        pdf_list, web_list, qna_prompt_template, condense_question_template
    )
    web_list, pdf_list = [], []
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
    result = st.session_state.qna_chain({"question": user_input})
    response = result["answer"]
    # Display chatbot response in chat container
    h.display_chat_message("bot", response)
    # Add chatbot response to chat history
    st.session_state.messages.append({"role": "bot", "content": response})