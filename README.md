# Web QnA
---------------------
A pipeline to answer questions based on information present in website(s)


### Setup
-------------
 ```
 pip install -r requirements.txt
 ```


### Environment
---------------
Add **OPENAI_API_KEY** and **FLASK_KEY**

```
touch .env
OPENAI_API_KEY="<your_open_api_key>"
OPENAI_API_TYPE="openai"
FLASK_KEY="rag"
FLASK_APP=app.py
FLASK_DEBUG=1
FLASK_RUN_HOST=0.0.0.0
source .env
```

### Execution
-------------
*Deploy endpoints on local*
```
flask run
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
	-d '{"user_id":"flom_0", "chatbot_type": "finance", "question": "What are the advantages of investing in tesla"}' \
	http://127.0.0.1:5000/qna
```
