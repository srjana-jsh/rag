import glob
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceHubEmbeddings

chunking_interface = RecursiveCharacterTextSplitter
embedding_model = OpenAIEmbeddings
pdf_list = [
    _ for _ in glob.glob(
        os.path.join(os.path.join(os.getcwd(), '../data/pdfs'), '*.pdf')
    ) if '' in _
]
web_list = []
prompt_max = 96000
chunk_overlap = 0
vectorstore_engine = 'Finbot-embedding-2'
llm_model = 'text-davinci-002'
llm_engine = 'finbot-gpt'
temperature = 0
search_type = 'mmr'
chunks_max = 15
answer_max_tokens = 512
source_documents=True
retrieval_kwargs = {'k': 4, 'lambda_mult': 0.5, 'fetch_k':10}
prompt_template_file = os.path.join(os.getcwd(),'../scripts/prompt_template.txt')
prompt_role = 'financial advisor'
prompt_input_variables = ['context', 'question']
