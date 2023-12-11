# Finbot
---------------------
A RAG pipeline to answer questions based on a set of user provided documents or url


### Setup
-------------
 ```
 pip install -r requirements.txt
 ```


### Credentials
---------------
Add **OPENAI_API_KEY** and **FLASK_KEY**

```
touch .env
FLASK_KEY="<flask-key>"
OPENAI_API_KEY="<you-api-key>"
OPENAI_API_TYPE="azure"
OPENAI_API_VERSION="2023-03-15-preview"
OPENAI_API_BASE="https://test-chatgpt-flomoney.openai.azure.com/"
source .env
```

### Execution
-------------
*Deploy endpoints on local*
```
python scripts/app.py
```

*Sample cURL Requests*
```
#Files for VectorDB - Documents
curl -X POST \
     -F "user_id=flom_0" \
     -F "chatbot_type=finance" \
     -F "upload_type=pdf" \
	 -F "url=" \
     -F "files[]=@./sample_data/pdfs/microsoft_investorcom_1.pdf" \
     -F "files[]=@./sample_data/pdfs/tesla_investorcom_1.pdf" \
     http://127.0.0.1:5000/vectordb


#Files for VectorDB - URLs
curl -X POST \
     -F "user_id=flom_0" \
     -F "chatbot_type=finance" \
     -F "upload_type=url" \
	 -F "url=https://www.economist.com/" \
     -F "files[]=" \
     http://127.0.0.1:5000/vectordb

#QnA 
curl -X POST \
	-H "Content-Type: application/json" \
	-d '{"user_id":"flom_0", "chatbot_type": "finance", "question": "What are the advantages of investing in nvidia"}' \
	http://127.0.0.1:5000/qna
```