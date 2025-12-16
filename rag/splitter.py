# rag/splitter.py
"""
RecursiveCharacterTextSplitter 기본 사용법 - 08-TextSplitter/02-RecursiveCharacterTextSplitter.ipynb
chunk_size, chunk_overlap 파라미터 설정
의미 단위로 재귀적 분할

Separators 개념 - 08-TextSplitter/02-RecursiveCharacterTextSplitter.ipynb
단락 → 문장 → 단어 순서로 분할
splitter.py는 "."(마침표)를 추가하여 문장 단위 분할을 더욱 강화

RAG 파이프라인의 2단계 - 12-RAG/01-RAG-Basic-PDF.ipynb
로드된 문서를 검색 가능한 작은 청크로 분할
벡터 임베딩 전 필수 단계
특히 33줄의 separators=["\n\n", "\n", ".", " ", ""]는 논문처럼 문장 구조가 중요한 문서에 최적화된 설정입니다!
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter # 08-TextSplitter/02-RecursiveCharacterTextSplitter.ipynb
from langchain_core.documents import Document
from typing import List


def split_documents(
    documents: List[Document], # 08-TextSplitter/02-RecursiveCharacterTextSplitter.ipynb
    chunk_size: int = 1000, # 교재에서는 250으로 설정하였지만 실제로는 1000으로 설정 (챗지피티가 추천, 바이브 코딩)
    chunk_overlap: int = 200 # 교재에서는 50으로 설정하였지만 실제로는 200으로 설정 (챗지피티가 추천, 바이브 코딩)
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
    splitter = RecursiveCharacterTextSplitter( # 08-TextSplitter/02-RecursiveCharacterTextSplitter.ipynb
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    split_docs = splitter.split_documents(documents) # 12-RAG/01-RAG-Basic-PDF.ipynb
    return split_docs
