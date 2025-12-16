# rag/vectorstore.py

"""
build_vectorstore 함수 - RAG 3-4단계

09-Embeddings/01-OpenAIEmbeddings.ipynb - 임베딩 모델 설정

10-VectorStore/02-FAISS.ipynb - FAISS 벡터스토어 생성

12-RAG/01-RAG-Basic-PDF.ipynb (Cell 16-17) - 전체 통합
build_retriever 함수 - RAG 5단계

11-Retriever/01-VectorStoreRetriever.ipynb - 검색기 생성
search_kwargs={"k": k} - 검색할 문서 개수 지정

12-RAG/01-RAG-Basic-PDF.ipynb (Cell 19) - RAG 파이프라인 통합

특히 28줄의 text-embedding-3-small 모델 선택은 비용 대비 성능이 우수하여 실전에서 많이 사용되는 설정입니다!
51-53줄의 search_kwargs={"k": k}는 질문에 가장 관련 있는 상위 k개 문서를 검색하는 핵심 파라미터입니다!
"""

from langchain_openai import OpenAIEmbeddings # 09-Embeddings/01-OpenAIEmbeddings.ipynb
from langchain_community.vectorstores import FAISS # 10-VectorStore/02-FAISS.ipynb
from langchain_core.documents import Document
from typing import List


def build_vectorstore(
    documents: List[Document],
    embedding_model: str = "text-embedding-3-small" # 09-Embeddings/01-OpenAIEmbeddings.ipynb / 가성비 좋은 모델
) -> FAISS:
    """
    Document 리스트로부터 FAISS VectorStore를 생성합니다.

    Parameters
    ----------
    documents : List[Document]
        분할된 문서 chunk
    embedding_model : str
        OpenAI embedding 모델 이름

    Returns
    -------
    FAISS
        생성된 VectorStore 객체
    """
    embeddings = OpenAIEmbeddings(model=embedding_model) # 09-Embeddings/01-OpenAIEmbeddings.ipynb / 12-RAG/01-RAG-Basic-PDF.ipynb / RAG의 3단계: 임베딩 생성
    vectorstore = FAISS.from_documents(documents, embeddings) # 10-VectorStore/02-FAISS.ipynb / 12-RAG/01-RAG-Basic-PDF.ipynb
    return vectorstore # RAG의 4단계: 임베딩된 Chunk를 DB에 저장


def build_retriever( # 11-Retriever/01-VectorStoreRetriever.ipynb
    vectorstore: FAISS,
    k: int = 4 # 11-Retriever/01-VectorStoreRetriever.ipynb / 검색할 문서 개수
):
    """
    VectorStore로부터 Retriever를 생성합니다.

    Parameters
    ----------
    vectorstore : FAISS
        FAISS VectorStore
    k : int
        검색할 문서 개수

    Returns
    -------
    BaseRetriever
    """
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": k}
    )
    return retriever # RAG의 5단계: 검색기 생성
