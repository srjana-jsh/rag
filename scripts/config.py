import os 

class AppConfig:
	ALLOWED_EXTENSIONS = ['pdf']
	UPLOAD_FOLDER = os.path.join(os.getcwd(), 'user_uploads')
	PROMPT_TEMPLATE = os.path.join(os.getcwd(), 'scripts/qna_prompt_template.txt')
	CONDENSE_TEMPLATE = os.path.join(os.getcwd(), 'scripts/condense_question_template.txt')	
