# rag/loader.py

"""
loader.py의 load_pdfs_from_dir 함수는 다음을 조합한 것입니다:

PyMuPDFLoader 사용법 - 07-DocumentLoader/01-PDF-Loader.ipynb
PDF 파일 로드
metadata 자동 추가

디렉토리 로더 개념 - 07-DocumentLoader/11-Directory-Loader.ipynb
여러 파일 탐색
파일 필터링 (glob 패턴)

PyPDFDirectoryLoader 개념 - 07-DocumentLoader/01-PDF-Loader.ipynb
디렉토리의 모든 PDF를 한 번에 로드
loader.py는 이를 수동으로 구현하여 더 세밀한 제어 가능

특히 34-36줄의 metadata["source"] = pdf_name 추가는 RAG 시스템에서 답변의 출처를 추적하는 데 매우 중요한 부분입니다!
"""

import os
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader # 07-DocumentLoader/01-PDF-Loader.ipynb
from langchain_core.documents import Document # 07-DocumentLoader/01-PDF-Loader.ipynb


def load_pdfs_from_dir(data_dir: str) -> List[Document]: 
    """
    지정한 디렉토리 내 모든 PDF 파일을 로드하여
    하나의 Document 리스트로 반환합니다.

    각 Document에는 source(pdf 파일명) metadata가 포함됩니다.    
    """
    if not os.path.isdir(data_dir): # 07-DocumentLoader/11-Directory-Loader.ipynb / 파일 시스템에서 파일 읽고 디렉토리 유효성 검사
        raise FileNotFoundError(f"디렉토리를 찾을 수 없습니다: {data_dir}")

    all_documents: List[Document] = []

    pdf_files = [ # 07-DocumentLoader/11-Directory-Loader.ipynb / glob 패턴으로 파일 필터링, 특정 확장자 파일만 선택
        f for f in os.listdir(data_dir)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        raise RuntimeError(f"{data_dir} 안에 PDF 파일이 없습니다.")

    for pdf_name in pdf_files: # 07-DocumentLoader/01-PDF-Loader.ipynb / PyMuPDFLoader 초기화 후 load() 호출
        pdf_path = os.path.join(data_dir, pdf_name)
        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()

        # PDF 출처 정보 추가하도록 설계
        # 07-DocumentLoader/01-PDF-Loader.ipynb / metadata는 딕셔너리 형태, source, page 등의 정보 포함
        # 10-VectorStore/02-FAISS.ipynb / metadata에 source 정보 추가하여 출처 추적
        for d in docs:
            d.metadata["source"] = pdf_name

        all_documents.extend(docs) # 07-DocumentLoader/11-Directory-Loader.ipynb / 여러 파일의 문서를 하나의 리스트로 통합

    return all_documents
