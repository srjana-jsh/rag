import os 

class AppConfig:
	ALLOWED_EXTENSIONS = ['pdf']
	UPLOAD_FOLDER = os.path.join(os.getcwd(), '../user_uploads')