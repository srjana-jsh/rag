import os
import datetime
import json
import glob
import warnings
import logging
from scripts import qna_memory as qna_m
from scripts import constants as c
from scripts import helpers as h
from scripts import scraper as s
from flask import Flask, flash, request, redirect, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# environment-variables
load_dotenv()

# Logging
warnings.filterwarnings("ignore")
logger = h.set_logging(logging.getLogger(__name__), __name__)

# App
app = Flask(__name__)
app.config.from_object("scripts.config.AppConfig")
app.secret_key = os.environ["FLASK_KEY"]


@app.route("/vectordb", methods=["POST"])
def set_vectordb():
    """
	Endpoint for uploading documents and URLs for a user id

	Sample cURL request 
	-------------------
	- For PDF Documents:
	curl -X POST \
	     -F "user_id=flom_0" \
	     -F "chatbot_type=finance" \
	     -F "upload_type=pdf" \
	     -F "url=" \
	     -F "files[]=@path_to_files/pdf_1.pdf" \
	     -F "files[]=@path_to_files/pdf_2.pdf" \
	     http://127.0.0.1:5000/vectordb

	- For URLs:
	curl -X POST \
	     -F "user_id=flom_0" \
	     -F "chatbot_type=finance" \
	     -F "upload_type=url" \
	     -F "url=https://www.economist.com/" \
	     -F "files[]=" \
	     http://127.0.0.1:5000/vectordb
	"""
    user_id = request.form.get("user_id")
    chatbot_type = request.form.get("chatbot_type")
    upload_type = request.form.get("upload_type")
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    file_directory = os.path.join(app.config["UPLOAD_FOLDER"], chatbot_type, user_id)
    if not os.path.isdir(file_directory):
        os.mkdir(file_directory)

    if upload_type == "pdf":
        if "files[]" not in request.files:
            flash("No files provided")
            return redirect(request.url)

        files = request.files.getlist("files[]")
        if all(
            [
                h.allowed_file(_.filename, app.config["ALLOWED_EXTENSIONS"])
                for _ in files
            ]
        ):
            for file in files:
                filename = secure_filename(file.filename)
                file.save(os.path.join(file_directory, filename))
            flash("File(s) successfully uploaded")
            return redirect("/")
        else:
            return jsonify(
                message=json.dumps(
                    f"Allowed file types are {', '.join(app.config['ALLOWED_EXTENSIONS'])}"
                )
            )

    if upload_type == "url":
        web_list = s.scrape_site(request.form.get("url"), app.config["HEADER_TEMPLATE"])
        if len(web_list) > 0:
            with open(os.path.join(file_directory, "web_test.txt"), "w") as output:
                output.write(str(web_list))
            flash("URL successfully uploaded")
            return redirect("/")
        else:
            flash("Site cannot be scraped")
            return redirect(request.url)


@app.route("/qna", methods=["POST"])
def set_qna():
    """
	Endpoint for answering a question based on below user inputs in API call:
	{"user_id":"", "chatbot_type": "", "question": ""}	

	Sample cURL request
	-------------------
	curl -X POST \
		-H "Content-Type: application/json" \
		-d '{"user_id":"flom_0", "chatbot_type": "finance", "question": "<question>"}' \
		http://127.0.0.1:5000/qna

	Returns
	-------
	{"message" : "<Answer to question>"}		
	"""
    qna_details = request.get_json()
    file_directory = os.path.join(
        app.config["UPLOAD_FOLDER"], qna_details["chatbot_type"], qna_details["user_id"]
    )
    if not os.path.isdir(file_directory):
        flash("You have not uploaded any content for this type of chatbot")
        return jsonify(
            message=json.dumps(
                "You have not uploaded any content for this type of chatbot"
            )
        )
        # return redirect(request.url)
    else:
        # content for question-answering
        web_list = []
        pdf_list = [_ for _ in glob.glob(f"{file_directory}/*.pdf")]
        qna_with_memory = qna_m.LangchainQnA(c.CHUNKING_INTERFACE, c.EMBEDDING_MODEL)
        qna_chain = qna_with_memory.main_function(
            pdf_list,
            web_list,
            app.config["PROMPT_TEMPLATE"],
            app.config["CONDENSE_TEMPLATE"],
        )
        answer = qna_chain({"question": qna_details["question"]})["answer"]
        logger.info(f'Answer to question : {qna_details["question"]} is : \n {answer}')
        return jsonify(message=json.dumps(answer))


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False, threaded=True)
