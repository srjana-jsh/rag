import os
import datetime
import json
import qna_memory as qna_m
import constants as c
import helpers as h
import scraper as s
from flask import Flask, flash, request, redirect, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config.from_object('config.AppConfig')
app.secret_key = os.environ["FLASK_KEY"]

@app.route('/vectordb', methods=['POST'])
def set_vectordb():
	user_id = request.form.get('user_id')
	chatbot_type = request.form.get('chatbot_type')	
	upload_type = request.form.get('upload_type')
	 
	if upload_type == 'pdf':
		if 'files[]' not in request.files:
		    flash('No files provided')
		    return redirect(request.url)		 

		files = request.files.getlist('files[]')
		if all([h.allowed_file(_.filename, app.config['ALLOWED_EXTENSIONS']) for _ in files]):	
			for file in files:
				filename = secure_filename(file.filename)
				file_directory = os.path.join(app.config['UPLOAD_FOLDER'], chatbot_type, user_id)
				if not os.path.isdir(file_directory):
				    os.mkdir(file_directory)				
				file.save(os.path.join(file_directory, filename))					
			flash('File(s) successfully uploaded')
			return redirect('/')		        
		else:
			return jsonify(message=f"Allowed file types are: {', '.join(app.config['ALLOWED_EXTENSIONS'])}")	

@app.route('/qna', methods=['POST'])
def set_qna():
	qna_details = request.get_json()
	file_directory = os.path.join(app.config['UPLOAD_FOLDER'], qna_details['chatbot_type'], qna_details['chatbot_type'])
	if not os.path.isdir(file_directory):
		flash('You have not uploaded any content for this type of chatbot')
		return redirect(request.url)
	else :	
		#content for question-answering	
		web_list = []
		pdf_list = [_ for _ in glob.glob(file_directory)]
		qna_with_memory = qna_m.LangchainQnA(c.CHUNKING_INTERFACE, c.EMBEDDING_MODEL)
		qna_chain = qna_with_memory.main_function(
			pdf_list, web_list, app.config['PROMPT_TEMPLATE'], app.config['CONDENSE_TEMPLATE']
		)
		answer = qna_chain({"question": qna_details["question"]})
		return jsonify(message=answer)		

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)		        	
