import os
import sys
import glob
import re
import importlib
import langchain
import logging
import warnings
# sys.path.append(os.path.join(os.getcwd(), "../scripts"))
# import constants as c
# import helpers as h
from scripts import constants as c
from scripts import helpers as h
from langchain.document_loaders import WebBaseLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceHubEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.llms import AzureOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
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
from dotenv import load_dotenv

# environment-variables
load_dotenv()

# Logging
warnings.filterwarnings("ignore")
logger = h.set_logging(logging.getLogger(__name__), __name__)


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
        unfetched_urls = []
        if len(pdf_list) > 0:
            for pdf in pdf_list:
                loaded_data.extend(UnstructuredPDFLoader(pdf).load())
        if len(web_list) > 0:
            for url in web_list:
                try:
                    loaded_data.extend(WebBaseLoader(url).load())
                except:
                    unfetched_urls.append(url)
                    continue
            logger.info(
                f"Proportion of unfetched urls : {len(unfetched_urls)/len(web_list)}"
            )
            logger.info(f"Unfetched urls : {unfetched_urls}")
        logger.info(f"Length of loaded data : {len(loaded_data)}")
        return loaded_data, unfetched_urls

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
        logger.info(f"Number of chunks : {len(chunked_data)}")
        return chunked_data

    def get_vectorstore(
        self,
        chunked_data: List[Document],
        vectorstore_engine: str,
        chunks_max: int,
    ) -> Chroma:
        """ """
        logger.info(f'OPENAI_API_TYPE : {os.getenv("OPENAI_API_TYPE")}')
        logger.info(f"Vectorstore Engine : {vectorstore_engine}")
        if self.embedding_model == OpenAIEmbeddings:
            if os.getenv("OPENAI_API_TYPE") == "azure":
                embedding_model = OpenAIEmbeddings(deployment=vectorstore_engine)
            if os.getenv("OPENAI_API_TYPE") == "openai":
                embedding_model = OpenAIEmbeddings()
        if self.embedding_model == HuggingFaceHubEmbeddings:
            embedding_model = HuggingFaceHubEmbeddings()
        logger.info(f"Embedding model : {embedding_model}")
        for _ in range(0, len(chunked_data), chunks_max):
            vectorstore = Chroma.from_documents(
                documents=chunked_data[_ : _ + chunks_max], embedding=embedding_model
            )
        logger.info(f"Vectorstore created : {vectorstore}")
        return vectorstore

    def get_qna_chain(
        self,
        vectorstore: Chroma,
        qna_prompt_template: str,
        condense_question_template: str,
        llm_model: str,
        llm_engine: str,
        temperature: int,
        search_type: str,
        answer_max_tokens: int,
        source_documents: bool,       
        qna_prompt_input: List[str],
        prompt_role: str,
        condense_question_input: List[str],
        history_tokens: int,
        debug_mode: bool,
        **retrieval_kwargs: Any,
    ) -> RetrievalQA:
        """ """
        langchain.debug = debug_mode
        # llm to use
        if os.getenv("OPENAI_API_TYPE") == "azure":
            base_llm = AzureOpenAI(
                engine=llm_engine,
                model_name=llm_model,
                temperature=temperature,
                max_tokens=answer_max_tokens,
            )
        if os.getenv("OPENAI_API_TYPE") == "openai":
            base_llm = ChatOpenAI(temperature=temperature)
        logger.info(f"Base LLM chosen : {base_llm}")
        # custom prompt to pass to llm
        if qna_prompt_template is not None:
            qna_prompt = PromptTemplate.from_file(
                qna_prompt_template,
                input_variables=qna_prompt_input,
                partial_variables={"role": prompt_role},
            )
        else:
            qna_prompt = None
        # if chat history is to be used
        if condense_question_template is not None:
            # prompt to condense question into standalone question
            chat_prompt = PromptTemplate.from_file(
                condense_question_template, input_variables=condense_question_input
            )
            # memory type and token limit
            memory = ConversationSummaryBufferMemory(
                llm=base_llm,
                memory_key="chat_history",
                return_messages=True,
                max_token_limit=history_tokens,
            )
            # retrieval chain with memory
            qna_chain = ConversationalRetrievalChain.from_llm(
                base_llm,
                retriever=vectorstore.as_retriever(
                    search_type=search_type, search_kwargs=retrieval_kwargs
                ),
                memory=memory,
                condense_question_prompt=chat_prompt,
                combine_docs_chain_kwargs=dict(prompt=qna_prompt),
                #                 return_source_documents=source_documents,
            )
        else:
            # retrieval chain without memory
            qna_chain = RetrievalQA.from_chain_type(
                base_llm,
                retriever=vectorstore.as_retriever(
                    search_type=search_type, search_kwargs=retrieval_kwargs
                ),
                chain_type_kwargs={"prompt": qna_prompt},
                return_source_documents=source_documents,
            )
        return qna_chain

    def main_function(
        self, pdf_list, web_list, qna_prompt_template, condense_question_template
    ):
        """
        Main function to the chain for answering questions
        """
        chunk_size = c.PROMPT_MAX // c.RETRIEVAL_KWARGS["k"]
        loaded_data = self.get_loaded_data(pdf_list, web_list)[0]
        chunked_data = self.get_chunked_data(loaded_data, chunk_size, c.CHUNK_OVERLAP)
        vectorstore = self.get_vectorstore(
            chunked_data, c.VECTORSTORE_ENGINE, c.CHUNKS_MAX
        )
        qna_chain = self.get_qna_chain(
            vectorstore,
            qna_prompt_template,
            condense_question_template,
            c.LLM_MODEL,
            c.LLM_ENGINE,
            c.TEMPERATURE,
            c.SEARCH_TYPE,
            c.ANSWER_MAX_TOKENS,
            c.SOURCE_DOCUMENTS,
            c.QNA_PROMPT_INPUT,
            c.PROMPT_ROLE,
            c.CONDENSE_QUESTION_INPUT,
            c.HISTORY_TOKENS,
            c.DEBUG_MODE,
            **c.RETRIEVAL_KWARGS,
        )
        return qna_chain


if __name__ == "__main__":
    # QnA Chain
    langchain_qna = LangchainQnA(chunking_interface, embedding_model)
    qna_chain = langchain_qna.main_function(
        pdf_list, web_list, qna_prompt_template, condense_question_template
    )
