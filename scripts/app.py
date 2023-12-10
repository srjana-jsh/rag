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

@app.route('/vectordb', methods=['POST'])
def set_vectordb():
	payload = request.get_json()

	if 'chatbot_type' not in payload:
	    flash('No chatbot type provided')
	    return redirect(request.url)	
	 
	if payload["upload_type"] == 'pdf':
		if 'files[]' not in request.files:
		    flash('No files provided')
		    return redirect(request.url)		 

		files = request.files.getlist('files[]')
		if all([h.allowed_file(_, app.config['ALLOWED_EXTENSIONS']) for _ in files]):	
			for file in files:
		        filename = secure_filename(file.filename)
		        file.save(
		        	os.path.join(
		        		app.config['UPLOAD_FOLDER'], payload['chatbot_type'], payload['user_id'], filename
		        	)
		        )
			flash('File(s) successfully uploaded')
			return redirect('/')		        
		else:
			return jsonify(message=f'Allowed file types are: {', '.join(app.config['ALLOWED_EXTENSIONS'])}')	

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)		        	
