import os
import sys
import glob
import re
import importlib
import helpers as h
import langchain
from langchain.document_loaders import WebBaseLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceHubEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
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
        if isinstance(
            self.chunking_interface, langchain.text_splitter.RecursiveCharacterTextSplitter
        ):   
            data_splitter = RecursiveCharacterTextSplitter(
                chunk_size = chunk_size, chunk_overlap = chunk_overlap
            )           
        if isinstance(
            self.chunking_interface, langchain.text_splitter.TextSplitter
        ):   
            data_splitter = TextSplitter(
                chunk_size = chunk_size, chunk_overlap = chunk_overlap
            )
        chunked_data = data_splitter.split_documents(loaded_data)          
        return chunked_data
    
    def get_vectorstore(
        self,
        chunked_data,
    ) -> Chroma:
        """
        """
        if isinstance(self.embedding_model, OpenAIEmbeddings):  
            embedding_model = OpenAIEmbeddings()
        if isinstance(self.embedding_model, HuggingFaceHubEmbeddings):  
            embedding_model = HuggingFaceHubEmbeddings()            
        vectorstore = Chroma.from_documents(
            documents=chunked_data, embedding=embedding_model
        )
        return vectorstore
    
    def get_qna_chain(
        self,
        vectorstore: Chroma,
        llm_model: str,
        temperature: int,
    ) -> RetrievalQA:
        """
        """
        base_llm = ChatOpenAI(model_name=llm_model, temperature=temperature)            
        qna_chain = RetrievalQA.from_chain_type(
            base_llm, retriever=vectorstore.as_retriever()
        )
        return qna_chain

    def main_function(
        self, 
        pdf_list: Optional[List[str]] = [], 
        web_list: Optional[List[str]] = [], 
        chunk_size: int = 2000, 
        chunk_overlap: int = 0, 
        llm_model: str = "gpt-3.5-turbo", 
        temperature: int = 0,  
    ):
        """
        Main function to the chain for answering questions
        """
        loaded_data = self.get_loaded_data(pdf_list, web_list)
        chunked_data = self.get_chunked_data(loaded_data, chunk_size, chunk_overlap)
        vectorstore = self.get_vectorstore (chunked_data)
        qna_chain = self.qna_chain(vectorstore, llm_model, temperature)
        return qna_chain

if __name__ == '__main__':
    chunking_interface = sys.argv[1]
    embedding_model = sys.argv[2]
    pdf_list = sys.argv[3]
    web_list = sys.argv[4]
    chunk_size = sys.argv[5]
    chunk_overlap = sys.argv[6]
    llm_model = sys.argv[7]
    temperature = sys.argv[8]
    langchain_qna = LangchainQnA(chunking_interface, embedding_model)
    qna_chain = langchain_qna.main_function(
        pdf_list, web_list, chunk_size, chunk_overlap, llm_model, temperature
    )
    