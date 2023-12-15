import os
import datetime
import json
import glob
import warnings
import logging
import sys
from scripts import qna_memory as qna_m
from scripts import constants as c
from scripts import helpers as h
from scripts import scraper as s
from flask import Flask, flash, request, redirect, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
#environment-variables
load_dotenv()
#Logging
warnings.filterwarnings("ignore")
logger = h.set_logging(logging.getLogger(__name__), __name__)
#App
app = Flask(__name__)
app.config.from_object('scripts.config.AppConfig')
app.secret_key = os.environ["FLASK_KEY"]
file_directory = os.path.join(app.config['UPLOAD_FOLDER'], 'finance', 'flom_0')
web_list = []
pdf_list = [_ for _ in glob.glob(f'{file_directory}/*.pdf')]
qna_with_memory = qna_m.LangchainQnA(c.CHUNKING_INTERFACE, c.EMBEDDING_MODEL)
qna_chain = qna_with_memory.main_function(pdf_list, web_list, app.config['PROMPT_TEMPLATE'], app.config['CONDENSE_TEMPLATE'])
answer = qna_chain({"question": "Should I invest in Tesla"})["answer"]
print(answer)