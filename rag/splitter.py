# rag/splitter.py

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    논문 문서를 의미 단위로 분할합니다.

    Parameters
    ----------
    documents : List[Document]
        원본 Document 리스트
    chunk_size : int
        각 chunk의 최대 길이
    chunk_overlap : int
        인접 chunk 간 중복 길이

    Returns
    -------
    List[Document]
        분할된 Document 리스트
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    split_docs = splitter.split_documents(documents)
    return split_docs
