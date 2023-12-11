import glob
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceHubEmbeddings

#Constructing Chain
CHUNKING_INTERFACE = RecursiveCharacterTextSplitter
EMBEDDING_MODEL = OpenAIEmbeddings
PROMPT_MAX = 24000
CHUNK_OVERLAP = 0
VECTORSTORE_ENGINE = "Finbot-embedding-2"
LLM_MODEL = "text-davinci-002"
LLM_ENGINE = "finbot-gpt"
TEMPERATURE = 0
SEARCH_TYPE = "mmr"
CHUNKS_MAX = 15
ANSWER_MAX_TOKENS = 512
SOURCE_DOCUMENTS = True
DEBUG_MODE = False
RETRIEVAL_KWARGS = {"k": 4, "lambda_mult": 0.75, "fetch_k": 10}
#Prompt Templates - to use chat history, context and question
PROMPT_ROLE = "financial advisor"
QNA_PROMPT_INPUT = ["context", "question"]
CONDENSE_QUESTION_INPUT = ["chat_history", "question"]
HISTORY_TOKENS = 2000
#HTTP requests
HEADER_TEMPLATE = {
    "User-Agent": "",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*"
    ";q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.google.com/",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}
