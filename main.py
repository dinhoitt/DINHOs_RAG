# main.py
"""
RAG 기본 파이프라인 구현

pdf 파일을 로드하여 RAG 파이프라인을 구현하기 때문에
12장 01-RAG-Basic-PDF.ipynb를 가장 많이 참고하였습니다.
"""
import os
import sys
import traceback
from typing import Optional

import os
from dotenv import load_dotenv
load_dotenv() # 01-Basic/01-OpenAI-APIKey.ipynb

print("LANGSMITH_TRACING:", os.getenv("LANGSMITH_TRACING")) # 01-Basic/01-OpenAI-APIKey.ipynb
print("LANGSMITH_PROJECT:", os.getenv("LANGSMITH_PROJECT")) # 01-Basic/01-OpenAI-APIKey.ipynb
print("LANGSMITH_API_KEY exists:", os.getenv("LANGSMITH_API_KEY") is not None) # 01-Basic/01-OpenAI-APIKey.ipynb


from rag.loader import load_pdfs_from_dir
from rag.splitter import split_documents
from rag.vectorstore import build_vectorstore, build_retriever
from rag.chain import build_rag_chain


def _print_env_hint():
    print("\n[ENV CHECK]")
    print("필수: OPENAI_API_KEY")
    print("권장(LangSmith): LANGCHAIN_TRACING_V2=true, LANGCHAIN_API_KEY, LANGSMITH_PROJECT")
    print("예시(.env):")
    print('  OPENAI_API_KEY="..."')
    print('  LANGCHAIN_API_KEY="..."')
    print("  LANGCHAIN_TRACING_V2=true")
    print('  LANGSMITH_PROJECT="RAG_PaperPresenter_Taesoo"\n')


def validate_env(): # 01-Basic/01-OpenAI-APIKey.ipynb
    """필수/권장 환경변수 점검"""
    if not os.getenv("OPENAI_API_KEY"):
        _print_env_hint()
        raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다. .env 또는 시스템 환경변수를 확인하세요.")

    tracing = os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
    has_langsmith_key = bool(os.getenv("LANGCHAIN_API_KEY"))
    if not (tracing and has_langsmith_key):
        print("[WARN] LangSmith 추적이 비활성화되어 있을 수 있습니다.")
        print("       과제 요건(추적 결과 발표)을 위해 .env에 LANGCHAIN_TRACING_V2=true 및 LANGCHAIN_API_KEY 설정을 권장합니다.\n")



def ask_question() -> str:
    """질문 입력(빈 문자열 방지)"""
    while True:
        q = input("question > ").strip()
        if q:
            return q
        print("질문이 비어 있습니다. 질문을 입력해 주세요.")



def build_pipeline( #12-RAG/01-RAG-Basic-PDF.ipynb
    data_dir: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    k: int = 4
    ):
    print(f"[1/4] PDF 폴더 로드 중... ({data_dir})")
    docs = load_pdfs_from_dir(data_dir) # 07-DocumentLoader/01-PDF-Loader.ipynb / 07-DocumentLoader/11-Directory-Loader.ipynb

    print("[2/4] 문서 분할 중...")
    chunks = split_documents( # 08-TextSplitter/02-RecursiveCharacterTextSplitter.ipynb
        docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    print("[3/4] 벡터스토어 생성 중...")
    vs = build_vectorstore(chunks) #10-VectorStore 

    print("[4/4] Retriever / Chain 구성 중...")
    retriever = build_retriever(vs, k=k) # 11-Retriever/01-VectorStoreRetriever.ipynb
    chain = build_rag_chain(retriever) #12-RAG/01-RAG-Basic-PDF.ipynb

    return chain


def run_once(chain):
    """사용자 입력 1회 실행"""
    question = ask_question()

    print("\n[RUN] RAG 실행 중...\n")

    result = chain.invoke({"question": question}) #12-RAG/01-RAG-Basic-PDF.ipynb

    print("========== RESULT ==========")
    print(result)
    print("============================\n")




def main():
    load_dotenv()  # .env 자동 로드
    validate_env()

    # 기본 데이터 경로 (필요하면 환경변수 DATA_DIR로 변경 가능)
    data_dir = os.getenv("DATA_DIR", "data")

    # 파라미터는 환경변수로 오버라이드 가능 
    chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
    top_k = int(os.getenv("TOP_K", "4"))

    try:
        chain = build_pipeline(
            data_dir=data_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            k=top_k
        )
    except Exception as e:
        print("\n[ERROR] 파이프라인 구성 중 오류가 발생했습니다.")
        print(f"원인: {e}")
        print("\n[TRACEBACK]")
        traceback.print_exc()
        sys.exit(1)

    print("\n[READY] 질문을 입력하면 'ppt/script' 모드로 답변합니다.")
    print("종료하려면 Ctrl+C\n")

    try:
        while True:
            run_once(chain)
    except KeyboardInterrupt:
        print("\n[EXIT] 종료합니다.")
    except Exception as e:
        print("\n[ERROR] 실행 중 오류가 발생했습니다.")
        print(f"원인: {e}")
        print("\n[TRACEBACK]")
        traceback.print_exc()


if __name__ == "__main__":
    main()
