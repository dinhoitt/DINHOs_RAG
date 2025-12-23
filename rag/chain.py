# rag/chain.py

"""
모델 생성: 04-Model/01-Chat-Models.ipynb - ChatOpenAI 설정
출력 파서: 03-OutputParser/00-concept.ipynb - 구조화된 출력
프롬프트: 02-Prompt/01-PromptTemplate.ipynb - 템플릿 작성
LCEL 문법: 01-Basic/03-LCEL.Ipynb - 파이프 연산자 체인
사용자 정의 함수: 13-LangChain-Expression-Language/03-RunnableLambda.ipynb - format_docs 구현
RAG 통합: 12-RAG/01-RAG-Basic-PDF.ipynb - retriever와 prompt 연결
특히 41-49줄의 체인 구성은 12-RAG/01-RAG-Basic-PDF.ipynb의 Cell 24, 28과 거의 동일한 구조로, RAG의 핵심 파이프라인을 구현하고 있습니다!
"""

from langchain_openai import ChatOpenAI # 04-Model/01-Chat-Models.ipynb
from langchain_core.runnables import RunnableLambda # 13-LangChain-Expression-Language/03-RunnableLambda.ipynb
from langchain_core.output_parsers import StrOutputParser # 03-OutputParser/00-concept.ipynb

from rag.prompts import INTEGRATED_PROMPT, PresentationOutput #02-Prompt/01-PromptTemplate.ipynb, 03-OutputParser/01-PydanticOuputParser.ipynb


def format_docs(docs): # 13-LangChain-Expression-Language/03-RunnableLambda.ipynb
    """
    Retriever 결과를 context 문자열로 변환
    (출처 PDF와 페이지 정보 포함)

    13-LangChain-Expression-Language/03-RunnableLambda.ipynb
    사용자 정의 함수 작성 방법
    함수는 단일 인자만 받아야 함

    12-RAG/01-RAG-Basic-PDF.ipynb
    docs[0].metadata 구조 확인
    page_content 및 metadata 활용
    """
    blocks = []
    for i, d in enumerate(docs, start=1):
        source = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "?")
        content = d.page_content.strip().replace("\n", " ")
        # 너무 긴 chunk는 잘라서 인용 안정화
        content = content[:1200]
        blocks.append(f"[E{i}] [{source} | page {page}] {content}")

    joined = "\n".join(blocks)
    return joined[:12000]  # 토큰 과다 방지


def build_rag_chain(retriever):
    """
    단일 RAG 체인:
    - 하나의 검색
    - 하나의 프롬프트
    - PPT + SCRIPT 동시 출력
    
    06-Chains/03-Structured-Output-Chain.ipynb
    with_structured_output()을 사용하여 구조화된 출력 강제
    """

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2
    )
    
    # 03-OutputParser/01-PydanticOuputParser.ipynb
    # 06-Chains/03-Structured-Output-Chain.ipynb
    # with_structured_output()을 사용하여 PresentationOutput 모델 형식으로 출력 강제
    # OpenAI Function Calling을 활용하여 JSON Schema 기반 출력 제어
    structured_llm = llm.with_structured_output(PresentationOutput)

    chain = ( #13-LangChain-Expression-Language/03-RunnableLambda.ipynb / 12-RAG/01-RAG-Basic-PDF.ipynb

        # 람다 함수로 입력 데이터 변환
        # RunnableLambda로 사용자 정의 함수 통합

        # retriever를 딕셔너리 키로 사용
        # context에 검색 결과 주입
        # 프롬프트 → LLM → 구조화된 출력 순서
        # LCEL 문법으로 연결

        {
            "question": lambda x: x["question"],
            "context": (lambda x: x["question"]) | retriever | RunnableLambda(format_docs),
        }
        | INTEGRATED_PROMPT
        | structured_llm
        # StrOutputParser() 제거 - with_structured_output()이 자동으로 파싱
    )

    return chain
