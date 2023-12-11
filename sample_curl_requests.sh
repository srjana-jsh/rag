#Files for VectorDB
curl -X POST \
     -F "user_id=flom_0" \
     -F "chatbot_type=finance" \
     -F "upload_type=pdf" \
     -F "files[]=@./data/pdfs/microsoft_investorcom_1.pdf" \
     -F "files[]=@./data/pdfs/nvidia_investorcom_1.pdf" \
     http://127.0.0.1:5000/vectordb

#QnA 
curl -X POST \
	-H "Content-Type: application/json" \
	-d '{"user_id":"flom_0", "chatbot_type": "finance", "question": ""}' \
	http://127.0.0.1:5000/qna


