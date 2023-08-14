import fitz
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
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
    """
    """
    pdf_document = fitz.open()
    
    for pdf_path in pdf_paths:
        pdf_to_merge = fitz.open(pdf_path)
        pdf_document.insert_pdf(pdf_to_merge)
        pdf_to_merge.close()

    pdf_document.save(output_path)
    pdf_document.close()
    
def get_clean_content(input_string: str) -> str:
    """
    """
    pattern = r"[^\w\s!?,.]|(?<=[^\?])\?"
    clean_string = re.sub(pattern, "", input_string)
    return clean_string    

def _text_splitter(chunk_size: int, chunk_overlap: int) -> TextSplitter:
    """
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )