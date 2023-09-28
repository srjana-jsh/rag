import glob
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceHubEmbeddings

chunking_interface = RecursiveCharacterTextSplitter
embedding_model = OpenAIEmbeddings
prompt_max = 48000
chunk_overlap = 0
vectorstore_engine = 'Finbot-embedding-2'
llm_model = 'text-davinci-002'
# llm_model = 'gpt-3.5-turbo'
llm_engine = 'finbot-gpt'
temperature = 0
search_type = 'mmr'
chunks_max = 15
answer_max_tokens = 512
source_documents=True
retrieval_kwargs = {'k': 4, 'lambda_mult': 0.5, 'fetch_k':10}
prompt_role = 'financial advisor'
qna_prompt_input = ['context', 'question']
condense_question_input = ['chat_history', 'question']
history_tokens = 2000
debug_mode = True
