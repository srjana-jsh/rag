import os
import sys
import glob
import re
import importlib
from langchain.document_loaders import WebBaseLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceHubEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.llms import AzureOpenAI
from typing import (
    AbstractSet,
    Any,
    Callable,
    Collection,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypedDict,
    TypeVar,
    Union,
    cast,
)

class LangchainQnA:
    
    def __init__(
        self,
        chunking_interface: Union[RecursiveCharacterTextSplitter, TextSplitter],
        embedding_model: Union[OpenAIEmbeddings, HuggingFaceHubEmbeddings],        
    ):
        """
        """
        os.environ["OPENAI_API_TYPE"] = "azure"
        os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"
        os.environ["OPENAI_API_BASE"] = "https://test-chatgpt-flomoney.openai.azure.com/"
        self.chunking_interface = chunking_interface
        self.embedding_model = embedding_model

    def get_loaded_data(
        self,
        pdf_list: Optional[List[str]],
        web_list: Optional[List[str]],         
    ) -> List[Document]:
        """
        """
        loaded_data = []        
        if len(pdf_list) > 0:
            for pdf in pdf_list :            
                loaded_data.extend(UnstructuredPDFLoader(pdf).load())        
        if len(web_list) > 0:
            for page in web_list :
                loaded_data.extend(WebBaseLoader(page).load())                     
        return loaded_data

    def get_chunked_data(
        self,
        loaded_data,
        chunk_size: int,
        chunk_overlap: int,
     ) -> List[Document]:
        """
        """
        if self.chunking_interface == RecursiveCharacterTextSplitter:   
            data_splitter = RecursiveCharacterTextSplitter(
                chunk_size = chunk_size, chunk_overlap = chunk_overlap
            )           
        if self.chunking_interface == TextSplitter:         
            data_splitter = TextSplitter(
                chunk_size = chunk_size, chunk_overlap = chunk_overlap
            )
        chunked_data = data_splitter.split_documents(loaded_data)          
        return chunked_data
    
    def chunk_list(self, lst, chunk_size):
        for i in range(0, len(lst), chunk_size):
            yield lst[i:i + chunk_size]



    
    def get_vectorstore(
        self,
        chunked_data: List[Document],
        vectorstore_engine: str,
    ) -> Chroma:
        """
        """
        if self.embedding_model == OpenAIEmbeddings :
            if os.getenv('OPENAI_API_TYPE') == "azure":  
                embedding_model = OpenAIEmbeddings(deployment=vectorstore_engine)
            if os.getenv('OPENAI_API_TYPE') == "openai":                
                embedding_model = OpenAIEmbeddings()
        if self.embedding_model == HuggingFaceHubEmbeddings:  
            embedding_model = HuggingFaceHubEmbeddings()    
        
        vectorstore = Chroma.from_documents(
                                            documents=chunked_data, 
                                            embedding=embedding_model
                                        )
        return vectorstore
    
    def get_qna_chain(
        self,
        vectorstore: Chroma,
        llm_model: str,
        llm_engine: str,        
        temperature: int,
        search_type: str,
        **retrieval_kwargs: Any,
    ) -> RetrievalQA:
        """
        """
        if os.getenv('OPENAI_API_TYPE') == "azure":        
            base_llm = AzureOpenAI(
                engine=llm_engine, model_name=llm_model, temperature=temperature
            )
        if os.getenv('OPENAI_API_TYPE') == "openai":  
            base_llm = ChatOpenAI(model_name=llm_model, temperature=temperature)
        qna_chain = RetrievalQA.from_chain_type(
            base_llm, 
            retriever=vectorstore.as_retriever(
                search_type=search_type, search_kwargs=retrieval_kwargs
            )
        )
        return qna_chain

    def main_function(
        self, 
        pdf_list: Optional[List[str]] = [], 
        web_list: Optional[List[str]] = [], 
        chunk_size: int = 2000, 
        chunk_overlap: int = 0, 
        vectorstore_engine: str = "Finbot-embedding",        
        llm_model: str = "text-davinci-002",
        llm_engine: str = "finbot-gpt",
        temperature: int = 0,
        search_type: str = 'mmr',
        **retrieval_kwargs: Any,
    ):
        """
        Main function to the chain for answering questions
        """
        loaded_data = self.get_loaded_data(pdf_list, web_list)
        chunked_data = self.get_chunked_data(loaded_data, chunk_size, chunk_overlap)
        vectorstore = self.get_vectorstore(chunked_data, vectorstore_engine)
        qna_chain = self.get_qna_chain(
            vectorstore, llm_model, llm_engine, temperature, search_type, **retrieval_kwargs
        )
        return qna_chain

if __name__ == '__main__':
    chunking_interface = sys.argv[1]
    embedding_model = sys.argv[2]
    pdf_list = sys.argv[3]
    web_list = sys.argv[4]
    chunk_size = sys.argv[5]
    chunk_overlap = sys.argv[6]
    vectorstore_engine = sys.argv[7]
    llm_model = sys.argv[8]
    llm_engine  = sys.argv[9]    
    temperature = sys.argv[10]
    search_type = sys.argv[11]
    retrieval_kwargs = sys.argv[12]
    #QnA Chain
    langchain_qna = LangchainQnA(chunking_interface, embedding_model)
    qna_chain = langchain_qna.main_function(
        pdf_list, web_list, chunk_size, chunk_overlap, vectorstore_engine,
        llm_model, llm_engine, temperature, search_type, **retrieval_kwargs
    )
    