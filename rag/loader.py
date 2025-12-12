# rag/loader.py

import os
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document


def load_pdfs_from_dir(data_dir: str) -> List[Document]:
    """
    지정한 디렉토리 내 모든 PDF 파일을 로드하여
    하나의 Document 리스트로 반환합니다.

    각 Document에는 source(pdf 파일명) metadata가 포함됩니다.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"디렉토리를 찾을 수 없습니다: {data_dir}")

    all_documents: List[Document] = []

    pdf_files = [
        f for f in os.listdir(data_dir)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        raise RuntimeError(f"{data_dir} 안에 PDF 파일이 없습니다.")

    for pdf_name in pdf_files:
        pdf_path = os.path.join(data_dir, pdf_name)
        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()

        # PDF 출처 정보 추가 (발표 시 매우 중요)
        for d in docs:
            d.metadata["source"] = pdf_name

        all_documents.extend(docs)

    return all_documents
