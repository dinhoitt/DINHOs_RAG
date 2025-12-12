# rag/vectorstore.py

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from typing import List


def build_vectorstore(
    documents: List[Document],
    embedding_model: str = "text-embedding-3-small"
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
    embeddings = OpenAIEmbeddings(model=embedding_model)
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore


def build_retriever(
    vectorstore: FAISS,
    k: int = 4
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
    return retriever
