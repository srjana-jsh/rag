import fitz
import re
import os
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from datetime import datetime
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


def merge_pdfs(pdf_paths: List[str], output_path: str) -> None:
    """ """
    pdf_document = fitz.open()

    for pdf_path in pdf_paths:
        pdf_to_merge = fitz.open(pdf_path)
        pdf_document.insert_pdf(pdf_to_merge)
        pdf_to_merge.close()

    pdf_document.save(output_path)
    pdf_document.close()


def get_clean_content(input_string: str) -> str:
    """ """
    pattern = r"[^\w\s!?,.]|(?<=[^\?])\?"
    clean_string = re.sub(pattern, "", input_string)
    return clean_string


def _text_splitter(chunk_size: int, chunk_overlap: int) -> TextSplitter:
    """ """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )


def extract_text_from_pdf(pdf_file: str) -> str:
    """ """
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def save_uploaded_files(uploaded_files: str, save_folder: str) -> None:
    """
    Function to save uploaded files to disk
    """
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for file in uploaded_files:
        file_path = os.path.join(save_folder, file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())
        # st.success(f"Saved file: {file_path}")


def display_chat_message(sender: str, message: str) -> None:
    """
    Function to display chat messages
    """
    if sender == "user":
        with st.chat_message(name="You", avatar="👨‍🎓"):
            st.write(message)
    elif sender == "bot":
        with st.chat_message(name="FinBot", avatar="🤖"):
            st.caption(message)


def set_logging(logger, module_name) -> logging.Logger:
    """
    Generates log with level INFO
    """
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(
        f"./logs/{datetime.now().strftime(f'{module_name}_%d_%m_%Y_%H_%M.log')}",
        mode="a",
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[INFO] - %(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def allowed_file(filename: str, allowed_extensions: List[str]) -> bool:
    """ """
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions
