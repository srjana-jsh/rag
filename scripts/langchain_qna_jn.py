import os
import sys
import glob
import re
import langchain

sys.path.append(os.path.join(os.getcwd(), "../scripts"))
import constants as c
from langchain.document_loaders import WebBaseLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceHubEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.llms import AzureOpenAI
from langchain.prompts import PromptTemplate
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
        """ """
        self.chunking_interface = chunking_interface
        self.embedding_model = embedding_model

    def get_loaded_data(
        self,
        pdf_list: Optional[List[str]],
        web_list: Optional[List[str]],
    ) -> List[Document]:
        """ """
        loaded_data = []
        if len(pdf_list) > 0:
            for pdf in pdf_list:
                loaded_data.extend(UnstructuredPDFLoader(pdf).load())
        if len(web_list) > 0:
            for page in web_list:
                loaded_data.extend(WebBaseLoader(page).load())
        return loaded_data

    def get_chunked_data(
        self,
        loaded_data,
        chunk_size: int,
        chunk_overlap: int,
    ) -> List[Document]:
        """ """
        if self.chunking_interface == RecursiveCharacterTextSplitter:
            data_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
        if self.chunking_interface == TextSplitter:
            data_splitter = TextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
        chunked_data = data_splitter.split_documents(loaded_data)
        return chunked_data

    def get_vectorstore(
        self,
        chunked_data: List[Document],
        vectorstore_engine: str,
        chunks_max: int,
    ) -> Chroma:
        """ """
        if self.embedding_model == OpenAIEmbeddings:
            if os.getenv("OPENAI_API_TYPE") == "azure":
                embedding_model = OpenAIEmbeddings(deployment=vectorstore_engine)
            if os.getenv("OPENAI_API_TYPE") == "openai":
                embedding_model = OpenAIEmbeddings()
        if self.embedding_model == HuggingFaceHubEmbeddings:
            embedding_model = HuggingFaceHubEmbeddings()
        for _ in range(0, len(chunked_data), chunks_max):
            vectorstore = Chroma.from_documents(
                documents=chunked_data[_ : _ + chunks_max], embedding=embedding_model
            )
        return vectorstore

    def get_qna_chain(
        self,
        vectorstore: Chroma,
        llm_model: str,
        llm_engine: str,
        temperature: int,
        search_type: str,
        answer_max_tokens: int,
        source_documents: bool,
        prompt_template_file: str,
        prompt_input_variables: List[str],
        prompt_role: str,
        **retrieval_kwargs: Any,
    ) -> RetrievalQA:
        """ """
        if prompt_template_file is not None:
            chain_prompt = PromptTemplate.from_file(
                prompt_template_file,
                input_variables=prompt_input_variables,
                #                 partial_variables={'role':prompt_role}
            )
        else:
            chain_prompt = None
        if os.getenv("OPENAI_API_TYPE") == "azure":
            base_llm = AzureOpenAI(
                engine=llm_engine,
                model_name=llm_model,
                temperature=temperature,
                max_tokens=answer_max_tokens,
            )
        if os.getenv("OPENAI_API_TYPE") == "openai":
            base_llm = ChatOpenAI(model_name=llm_model, temperature=temperature)
        qna_chain = RetrievalQA.from_chain_type(
            base_llm,
            retriever=vectorstore.as_retriever(
                search_type=search_type, search_kwargs=retrieval_kwargs
            ),
            chain_type_kwargs={"prompt": chain_prompt},
            return_source_documents=source_documents,
        )
        print(f"prompttemplate : {qna_chain.combine_documents_chain.llm_chain.prompt}")
        return qna_chain

    def main_function(self, pdf_list, web_list, prompt_template_file):
        """
        Main function to the chain for answering questions
        """
        chunk_size = c.prompt_max // c.retrieval_kwargs["k"]
        loaded_data = self.get_loaded_data(pdf_list, web_list)
        chunked_data = self.get_chunked_data(loaded_data, chunk_size, c.chunk_overlap)
        vectorstore = self.get_vectorstore(
            chunked_data, c.vectorstore_engine, c.chunks_max
        )
        qna_chain = self.get_qna_chain(
            vectorstore,
            c.llm_model,
            c.llm_engine,
            c.temperature,
            c.search_type,
            c.answer_max_tokens,
            c.source_documents,
            prompt_template_file,
            c.prompt_input_variables,
            c.prompt_role,
            **c.retrieval_kwargs,
        )
        return qna_chain


if __name__ == "__main__":
    # QnA Chain
    langchain_qna = LangchainQnA(chunking_interface, embedding_model)
    qna_chain = langchain_qna.main_function(pdf_list, web_list, prompt_template_file)
